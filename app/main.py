"""FastAPI backend for DocPilot Web."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import structlog
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    FileStatus,
    StepEnum,
    TaskResult,
    TaskStatus,
    UploadResponse,
)

logger = structlog.get_logger()

# ── Config ──
MAX_FILES = 50
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp",
    ".docx", ".xlsx",
}
TASK_TTL_SECONDS = 30 * 60  # 30 minutes

# ── In-memory storage ──
tasks: dict[str, TaskStatus] = {}
task_results: dict[str, TaskResult] = {}
task_output_dirs: dict[str, Path] = {}
task_upload_dirs: dict[str, Path] = {}
task_created: dict[str, float] = {}

# ── App ──
app = FastAPI(title="DocPilot", version="1.0.0")


@app.on_event("startup")
async def startup():
    """Pre-warm EasyOCR reader on startup."""
    import threading

    def warm_up():
        from app.processor import get_reader

        logger.info("warming_up_easyocr")
        get_reader()
        logger.info("easyocr_ready")

    threading.Thread(target=warm_up, daemon=True).start()

    # Start cleanup loop
    asyncio.create_task(_cleanup_loop())


async def _cleanup_loop():
    """Remove expired tasks every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        now = time.time()
        expired = [
            tid
            for tid, created in task_created.items()
            if now - created > TASK_TTL_SECONDS
        ]
        for tid in expired:
            tasks.pop(tid, None)
            task_results.pop(tid, None)
            task_created.pop(tid, None)

            upload_dir = task_upload_dirs.pop(tid, None)
            if upload_dir and upload_dir.exists():
                shutil.rmtree(upload_dir, ignore_errors=True)

            output_dir = task_output_dirs.pop(tid, None)
            if output_dir and output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

            logger.info("task_cleaned", task_id=tid)


def _format_size(size_bytes: int) -> str:
    if size_bytes > 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    return f"{size_bytes // 1024} KB"


# ── Routes ──


@app.get("/api/health")
async def health():
    return {"status": "ok", "tasks": len(tasks)}


@app.post("/api/upload", response_model=UploadResponse)
async def upload(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    """Upload files and start processing."""
    if len(files) > MAX_FILES:
        return JSONResponse(
            status_code=400,
            content={"error": f"Максимум {MAX_FILES} файлів"},
        )

    if len(files) < 2:
        return JSONResponse(
            status_code=400,
            content={"error": "Потрібно щонайменше 2 файли"},
        )

    task_id = str(uuid.uuid4())
    upload_dir = Path(tempfile.mkdtemp(prefix=f"docpilot_upload_{task_id[:8]}_"))

    file_statuses: list[FileStatus] = []
    saved_paths: list[Path] = []

    for f in files:
        ext = Path(f.filename or "unknown").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Непідтримуваний формат: {ext}"},
            )

        content = await f.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Файл {f.filename} перевищує {MAX_FILE_SIZE_MB} МБ"
                },
            )

        safe_name = f.filename or f"file_{len(saved_paths)}{ext}"
        file_path = upload_dir / safe_name
        file_path.write_bytes(content)
        saved_paths.append(file_path)

        file_statuses.append(
            FileStatus(
                name=safe_name,
                ext=ext.lstrip(".").upper(),
                size=_format_size(len(content)),
            )
        )

    status = TaskStatus(
        task_id=task_id,
        step=StepEnum.UPLOADING,
        progress=0,
        files=file_statuses,
        message="Файли завантажено",
    )
    tasks[task_id] = status
    task_created[task_id] = time.time()
    task_upload_dirs[task_id] = upload_dir

    background_tasks.add_task(_run_pipeline_wrapper, task_id, saved_paths)

    return UploadResponse(task_id=task_id, file_count=len(saved_paths))


def _run_pipeline_wrapper(task_id: str, file_paths: list[Path]) -> None:
    """Wrapper to run pipeline (called via BackgroundTasks)."""
    from app.processor import run_pipeline

    run_pipeline(task_id, tasks, file_paths, task_results, task_output_dirs)


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Get processing status for polling."""
    status = tasks.get(task_id)
    if not status:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return status.model_dump()


@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    """Get final processing result."""
    result = task_results.get(task_id)
    if not result:
        status = tasks.get(task_id)
        if status and status.step != StepEnum.DONE:
            return JSONResponse(
                status_code=202,
                content={"error": "Processing not complete"},
            )
        return JSONResponse(status_code=404, content={"error": "Result not found"})
    return result


@app.get("/api/download/{task_id}")
async def download(task_id: str):
    """Download ZIP with all generated PDFs."""
    output_dir = task_output_dirs.get(task_id)
    if not output_dir or not output_dir.exists():
        return JSONResponse(status_code=404, content={"error": "Files not found"})

    zip_path = output_dir.parent / f"docpilot_{task_id[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pdf_file in output_dir.glob("*.pdf"):
            zf.write(pdf_file, pdf_file.name)

    return FileResponse(
        path=str(zip_path),
        filename="DocPilot_результат.zip",
        media_type="application/zip",
    )


# Serve static files (index.html) — mount LAST
app.mount("/", StaticFiles(directory="static", html=True), name="static")
