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

from pydantic import BaseModel

from app.models import (
    FileStatus,
    StepEnum,
    TaskResult,
    TaskStatus,
    UploadResponse,
)

logger = structlog.get_logger()

# ── Config ──
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {
    ".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp",
    ".docx", ".xlsx",
}
TASK_TTL_SECONDS = 30 * 60  # 30 minutes


class TaskCreateRequest(BaseModel):
    instruction: str = ""
    mode: str = "simple"


class TaskStartRequest(BaseModel):
    instruction: str = ""
    mode: str = "simple"


# ── In-memory storage ──
tasks: dict[str, TaskStatus] = {}
task_results: dict[str, TaskResult] = {}
task_output_dirs: dict[str, Path] = {}
task_upload_dirs: dict[str, Path] = {}
task_created: dict[str, float] = {}
task_instructions: dict[str, str] = {}
task_modes: dict[str, str] = {}

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
            task_instructions.pop(tid, None)
            task_modes.pop(tid, None)

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
    import os
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    return {
        "status": "ok",
        "tasks": len(tasks),
        "gemini": "configured" if len(gemini_key) > 10 else "missing",
    }


@app.get("/api/gemini-test")
async def gemini_test():
    """Quick Gemini API connectivity test."""
    import json
    import os
    import urllib.error
    import urllib.request

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {"status": "error", "detail": "GEMINI_API_KEY not set"}

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = json.dumps({
        "contents": [{"parts": [{"text": "Say OK"}]}],
        "generationConfig": {"maxOutputTokens": 5},
    }).encode()

    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"status": "ok", "response": text.strip()}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        return {"status": "error", "code": e.code, "detail": body}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.post("/api/upload", response_model=UploadResponse)
async def upload(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    """Upload files and start processing (single request, for small batches)."""
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

    background_tasks.add_task(_run_pipeline_wrapper, task_id, saved_paths, "")

    return UploadResponse(task_id=task_id, file_count=len(saved_paths))


# ── Batch upload endpoints (for large file counts) ──


@app.post("/api/task/create")
async def create_task(body: TaskCreateRequest | None = None):
    """Create a task for batch uploading."""
    task_id = str(uuid.uuid4())
    upload_dir = Path(tempfile.mkdtemp(prefix=f"docpilot_upload_{task_id[:8]}_"))

    tasks[task_id] = TaskStatus(
        task_id=task_id,
        step=StepEnum.UPLOADING,
        progress=0,
        files=[],
        message="Завантажуємо файли...",
    )
    task_created[task_id] = time.time()
    task_upload_dirs[task_id] = upload_dir
    if body and body.instruction:
        task_instructions[task_id] = body.instruction
    if body and body.mode:
        task_modes[task_id] = body.mode

    return {"task_id": task_id}


@app.post("/api/task/{task_id}/files")
async def upload_batch(task_id: str, files: list[UploadFile] = File(...)):
    """Upload a batch of files to an existing task."""
    status = tasks.get(task_id)
    if not status:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    upload_dir = task_upload_dirs.get(task_id)
    if not upload_dir or not upload_dir.exists():
        return JSONResponse(status_code=404, content={"error": "Upload dir not found"})

    saved_count = 0
    for f in files:
        ext = Path(f.filename or "unknown").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            continue

        content = await f.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            continue

        safe_name = f.filename or f"file_{len(status.files)}{ext}"
        # Avoid name collisions
        file_path = upload_dir / safe_name
        counter = 1
        while file_path.exists():
            stem = Path(safe_name).stem
            file_path = upload_dir / f"{stem}_{counter}{ext}"
            counter += 1
        file_path.write_bytes(content)

        status.files.append(
            FileStatus(
                name=file_path.name,
                ext=ext.lstrip(".").upper(),
                size=_format_size(len(content)),
            )
        )
        saved_count += 1

    return {"saved": saved_count, "total": len(status.files)}


@app.post("/api/task/{task_id}/start")
async def start_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    body: TaskStartRequest | None = None,
):
    """Start processing after all batches are uploaded."""
    status = tasks.get(task_id)
    if not status:
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    upload_dir = task_upload_dirs.get(task_id)
    if not upload_dir or not upload_dir.exists():
        return JSONResponse(status_code=404, content={"error": "Upload dir not found"})

    file_paths = sorted(upload_dir.iterdir())
    if len(file_paths) < 2:
        return JSONResponse(
            status_code=400,
            content={"error": "Потрібно щонайменше 2 файли"},
        )

    instruction = ""
    if body and body.instruction:
        instruction = body.instruction
    elif task_id in task_instructions:
        instruction = task_instructions[task_id]

    mode = "simple"
    if body and body.mode:
        mode = body.mode
    elif task_id in task_modes:
        mode = task_modes[task_id]

    status.message = "Починаємо обробку..."
    background_tasks.add_task(_run_pipeline_wrapper, task_id, file_paths, instruction, mode)

    return {"task_id": task_id, "file_count": len(file_paths)}


def _run_pipeline_wrapper(
    task_id: str,
    file_paths: list[Path],
    instruction: str = "",
    mode: str = "simple",
) -> None:
    """Wrapper to run pipeline (called via BackgroundTasks)."""
    from app.processor import run_pipeline

    run_pipeline(task_id, tasks, file_paths, task_results, task_output_dirs, instruction, mode)


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
_static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
