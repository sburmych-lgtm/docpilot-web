"""Pipeline processor — OCR, dedup, classify, build PDFs."""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog
import torch
from PyPDF2 import PdfReader

from app.dedup import find_duplicates
from app.gemini import classify_batch, classify_text
from app.models import (
    FileStatusEnum,
    Group,
    GroupFile,
    StepEnum,
    TaskResult,
    TaskStatus,
)
from app.pdf_builder import build_pdf, safe_filename

logger = structlog.get_logger()

# ── Singleton EasyOCR Reader ──
_reader: Any = None
_reader_lock = threading.Lock()

GPU_AVAILABLE = torch.cuda.is_available()
MAX_OCR_WORKERS = 2 if GPU_AVAILABLE else 4  # fewer threads on GPU to avoid VRAM OOM


def get_reader() -> Any:
    """Get or create the EasyOCR Reader singleton (thread-safe)."""
    global _reader  # noqa: PLW0603
    if _reader is None:
        with _reader_lock:
            if _reader is None:  # double-check after acquiring lock
                import easyocr  # type: ignore[import-untyped]

                _reader = easyocr.Reader(
                    ["uk", "en"], gpu=GPU_AVAILABLE, verbose=False,
                )
                gpu_label = "GPU" if GPU_AVAILABLE else "CPU"
                logger.info("easyocr_reader_initialized", device=gpu_label)
    return _reader


# ── Keyword map (from legal_ua plugin) ──
KEYWORD_MAP: dict[str, str] = {
    "позовна заява": "Позовна заява",
    "рішення суду": "Рішення суду",
    "ухвала": "Ухвала",
    "клопотання": "Клопотання",
    "апеляційна скарга": "Апеляційна скарга",
    "касаційна скарга": "Касаційна скарга",
    "постанова": "Постанова",
    "судовий наказ": "Судовий наказ",
    "виконавчий лист": "Виконавчий лист",
    "протокол судового засідання": "Протокол",
    "довіреність": "Довіреність",
    "договір": "Договір",
    "акт звірки": "Акт звірки",
}

# Regex: case number pattern
CASE_NUMBER_PATTERN = re.compile(r"\d{1,5}/\d{1,5}/\d{2,4}")

# Group colors palette
GROUP_COLORS = [
    "#22c55e", "#3b82f6", "#a855f7", "#f59e0b",
    "#ef4444", "#06b6d4", "#ec4899", "#14b8a6",
    "#f97316", "#8b5cf6",
]

MIN_TEXT_LENGTH = 20  # minimum chars to consider PDF text valid
SESSION_GAP_MINUTES = 30  # gap between photo sessions


def format_file_size(size_bytes: int) -> str:
    """Format file size to human-readable string."""
    if size_bytes > 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    return f"{size_bytes // 1024} KB"


# ── Metadata extraction (Simple mode) ──


def extract_metadata_date(file_path: Path) -> datetime:
    """Extract date/time from file metadata.

    Priority: EXIF DateTimeOriginal > EXIF DateTime > PDF CreationDate > file mtime.
    Always returns a datetime (file mtime as ultimate fallback).
    """
    suffix = file_path.suffix.lower()

    # Images: try EXIF
    if suffix in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}:
        try:
            from PIL import Image  # type: ignore[import-untyped]
            from PIL.ExifTags import Base as ExifBase  # type: ignore[import-untyped]

            with Image.open(file_path) as img:
                exif = img.getexif()
                if exif:
                    # DateTimeOriginal (tag 36867) > DateTime (tag 306)
                    for tag_id in (ExifBase.DateTimeOriginal, ExifBase.DateTime):
                        val = exif.get(tag_id)
                        if val:
                            try:
                                return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                            except (ValueError, TypeError):
                                continue
        except Exception:
            logger.debug("exif_read_failed", path=file_path.name)

    # PDFs: try CreationDate
    if suffix == ".pdf":
        try:
            reader = PdfReader(str(file_path))
            info = reader.metadata
            if info and info.creation_date:
                return info.creation_date.replace(tzinfo=None)
        except Exception:
            logger.debug("pdf_date_read_failed", path=file_path.name)

    # Fallback: file modification time
    return datetime.fromtimestamp(os.path.getmtime(file_path))


def classify_by_metadata(
    file_paths: list[Path],
    gap_minutes: int = SESSION_GAP_MINUTES,
) -> dict[str, list[tuple[Path, int, str]]]:
    """Group files into sessions based on metadata timestamps.

    Files photographed within `gap_minutes` of each other belong to the same session.
    Returns groups dict: { "session_name": [(path, confidence, method), ...] }
    """
    # Extract dates for all files
    dated: list[tuple[Path, datetime]] = []
    for p in file_paths:
        dt = extract_metadata_date(p)
        dated.append((p, dt))

    # Sort by date
    dated.sort(key=lambda x: x[1])

    if not dated:
        return {}

    # Split into sessions by gap
    sessions: list[list[tuple[Path, datetime]]] = []
    current_session: list[tuple[Path, datetime]] = [dated[0]]

    for i in range(1, len(dated)):
        gap = (dated[i][1] - dated[i - 1][1]).total_seconds() / 60
        if gap > gap_minutes:
            sessions.append(current_session)
            current_session = [dated[i]]
        else:
            current_session.append(dated[i])
    sessions.append(current_session)

    # Build groups with readable names
    groups: dict[str, list[tuple[Path, int, str]]] = {}
    for session in sessions:
        first_dt = session[0][1]
        last_dt = session[-1][1]

        if first_dt.date() == last_dt.date():
            # Same day: "05.03.2026 13:10-13:45"
            name = (
                f"{first_dt.strftime('%d.%m.%Y')} "
                f"{first_dt.strftime('%H:%M')}-{last_dt.strftime('%H:%M')}"
            )
        else:
            # Different days
            name = (
                f"{first_dt.strftime('%d.%m.%Y %H:%M')} — "
                f"{last_dt.strftime('%d.%m.%Y %H:%M')}"
            )

        items = [(p, 90, "metadata") for p, _ in session]
        groups[name] = items

    logger.info(
        "metadata_classification",
        total_files=len(file_paths),
        sessions=len(sessions),
    )
    return groups


# ── Text extraction ──


def extract_text_from_pdf(file_path: Path) -> str | None:
    """Try extracting text from PDF using PyPDF2 (no OCR needed).

    Returns text if the PDF has a text layer, None otherwise.
    """
    try:
        reader = PdfReader(str(file_path))
        texts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        full_text = "\n".join(texts).strip()
        if len(full_text) >= MIN_TEXT_LENGTH:
            logger.debug("pdf_text_extracted", path=file_path.name, chars=len(full_text))
            return full_text
        return None
    except Exception:
        logger.debug("pdf_text_extraction_failed", path=file_path.name)
        return None


def ocr_image(file_path: Path) -> str:
    """Run OCR on a single image file. Returns extracted text."""
    reader = get_reader()
    try:
        # Use numpy for Cyrillic path support on Windows
        raw = np.fromfile(str(file_path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            return ""
        results = reader.readtext(img, detail=0)
        text = "\n".join(results) if results else ""
        # Free image memory immediately
        del img, raw
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
        return text
    except Exception:
        logger.warning("ocr_failed", path=str(file_path), exc_info=True)
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
        return ""


def extract_text(file_path: Path) -> str:
    """Extract text from any supported file.

    For PDFs: try text layer first, fall back to OCR.
    For images: use OCR directly.
    """
    if file_path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(file_path)
        if text is not None:
            return text
        # PDF is a scan — fall through to OCR
        logger.debug("pdf_needs_ocr", path=file_path.name)

    return ocr_image(file_path)


# ── Classification ──


def classify_by_regex(text: str) -> tuple[str, str, int] | None:
    """Try regex classification. Returns (category, method, confidence%) or None."""
    match = CASE_NUMBER_PATTERN.search(text)
    if match:
        return f"Справа {match.group(0)}", "regex", 95
    return None


def classify_by_keyword(text: str) -> tuple[str, str, int] | None:
    """Try keyword classification. Returns (category, method, confidence%) or None."""
    text_lower = text.lower()
    for keyword, category in KEYWORD_MAP.items():
        if keyword in text_lower:
            return category, "keyword", 85
    return None


def classify_by_gemini(text: str) -> tuple[str, str, int] | None:
    """Try Gemini classification. Returns (category, method, confidence%) or None."""
    result = classify_text(text)
    if result and result.get("doc_type"):
        doc_type = result["doc_type"]
        case_number = result.get("case_number", "")
        confidence = int(float(result.get("confidence", 0.7)) * 100)
        confidence = max(40, min(confidence, 95))

        if case_number and case_number != "null" and case_number.strip():
            category = f"{doc_type} (Справа {case_number})"
        else:
            category = doc_type
        return category, "gemini", confidence
    return None


# ── Pipeline ──


def run_pipeline(
    task_id: str,
    tasks: dict[str, TaskStatus],
    file_paths: list[Path],
    task_results: dict,
    task_output_dirs: dict,
    instruction: str = "",
    mode: str = "simple",
) -> None:
    """Run the full processing pipeline.

    Args:
        mode: "simple" (metadata-based, no OCR) or "ai" (OCR + classify).

    Updates tasks[task_id] in-place with progress.
    """
    start_time = time.time()
    status = tasks[task_id]
    progress_lock = threading.Lock()

    try:
        total = len(file_paths)

        if mode == "simple":
            # ═══════════════════════════════════════════
            # SIMPLE MODE: metadata → dedup → sessions → PDF
            # ═══════════════════════════════════════════
            status.step = StepEnum.METADATA
            status.message = "Читаємо метадані файлів..."
            status.progress = 10

            # Mark all files as processing then done quickly
            for fs in status.files:
                fs.status = FileStatusEnum.PROCESSING
            status.progress = 20

            for fs in status.files:
                fs.status = FileStatusEnum.DONE
            status.progress = 30

            # ── Dedup ──
            status.step = StepEnum.DEDUP
            status.message = "Шукаємо дублікати..."
            status.progress = 35

            unique_paths, dup_paths = find_duplicates(file_paths)
            duplicates_removed = len(dup_paths)
            active_paths = [p for p in file_paths if p not in set(dup_paths)]
            status.progress = 45

            # ── Classify by metadata ──
            status.step = StepEnum.CLASSIFYING
            status.message = "Групуємо за датами..."
            status.progress = 50

            groups = classify_by_metadata(active_paths)
            unclassified: list[Path] = []

            status.progress = 70

        else:
            # ═══════════════════════════════════════════
            # AI MODE: OCR → dedup → classify → PDF
            # ═══════════════════════════════════════════

            # ── Step 1: Extract text (PDF text layer + OCR) ──
            status.step = StepEnum.OCR
            status.message = "Читаємо текст документів..."

            ocr_results: dict[Path, str] = {}
            completed_count = 0

            def process_file(path: Path) -> tuple[Path, str]:
                """Extract text from a single file (runs in thread pool)."""
                with progress_lock:
                    for fs in status.files:
                        if fs.name == path.name:
                            fs.status = FileStatusEnum.PROCESSING
                            break

                text = extract_text(path)

                with progress_lock:
                    for fs in status.files:
                        if fs.name == path.name:
                            fs.status = FileStatusEnum.DONE
                            break

                return path, text

            workers = 1 if (GPU_AVAILABLE and total > 50) else MAX_OCR_WORKERS
            logger.info("ocr_start", total=total, workers=workers, gpu=GPU_AVAILABLE)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(process_file, p): p for p in file_paths}
                for future in as_completed(futures):
                    path, text = future.result()
                    ocr_results[path] = text
                    completed_count += 1
                    with progress_lock:
                        status.progress = int(completed_count / total * 40)

            # ── Step 2: Dedup ──
            status.step = StepEnum.DEDUP
            status.message = "Шукаємо дублікати..."
            status.progress = 45

            unique_paths, dup_paths = find_duplicates(file_paths)
            duplicates_removed = len(dup_paths)

            for dp in dup_paths:
                ocr_results.pop(dp, None)

            status.progress = 50

            # ── Step 3: Classify ──
            status.step = StepEnum.CLASSIFYING
            status.message = "Класифікуємо документи..."

            groups: dict[str, list[tuple[Path, int, str]]] = {}
            unclassified: list[Path] = []

            if instruction:
                # ── Instruction-based: Gemini batch classification ──
                status.message = "AI класифікує за інструкцією..."
                texts_for_gemini: dict[str, str] = {}
                path_map: dict[str, Path] = {}

                for path, text in ocr_results.items():
                    if text.strip():
                        fid = path.name
                        texts_for_gemini[fid] = text
                        path_map[fid] = path
                    else:
                        unclassified.append(path)

                batch_size = 15
                fids = list(texts_for_gemini.keys())
                classified_fids: set[str] = set()

                for batch_start in range(0, len(fids), batch_size):
                    batch_fids = fids[batch_start:batch_start + batch_size]
                    batch_texts = {fid: texts_for_gemini[fid] for fid in batch_fids}

                    batch_result = classify_batch(batch_texts, instruction)
                    if batch_result:
                        for fid, info in batch_result.items():
                            path = path_map.get(fid)
                            if path:
                                group_name = info.get("group", "Невідомі")
                                confidence = int(float(info.get("confidence", 0.7)) * 100)
                                confidence = max(40, min(confidence, 95))
                                groups.setdefault(group_name, []).append(
                                    (path, confidence, "gemini")
                                )
                                classified_fids.add(fid)
                    else:
                        logger.warning(
                            "gemini_batch_failed_fallback",
                            batch_start=batch_start,
                            batch_size=len(batch_fids),
                        )
                        status.message = "Gemini недоступний, класифікуємо локально..."
                        for fid in batch_fids:
                            if fid in classified_fids:
                                continue
                            path = path_map.get(fid)
                            if not path:
                                continue
                            text = texts_for_gemini[fid]
                            result = classify_by_regex(text)
                            if result is None:
                                result = classify_by_keyword(text)
                            if result is None:
                                result = classify_by_gemini(text)
                            if result:
                                cat_name, method, conf = result
                                groups.setdefault(cat_name, []).append((path, conf, method))
                            else:
                                unclassified.append(path)
                            classified_fids.add(fid)

                    done_ratio = min(batch_start + batch_size, len(fids)) / max(len(fids), 1)
                    status.progress = 50 + int(done_ratio * 20)

                for fid in fids:
                    if fid not in classified_fids:
                        path = path_map.get(fid)
                        if path:
                            unclassified.append(path)
            else:
                # ── No instruction: waterfall classification ──
                for path, text in ocr_results.items():
                    if not text.strip():
                        unclassified.append(path)
                        continue

                    result = classify_by_regex(text)
                    if result is None:
                        result = classify_by_keyword(text)
                    if result is None:
                        result = classify_by_gemini(text)

                    if result:
                        cat_name, method, confidence = result
                        groups.setdefault(cat_name, []).append((path, confidence, method))
                    else:
                        unclassified.append(path)

            if unclassified:
                groups["Невідомі документи"] = [
                    (p, 0, "none") for p in unclassified
                ]

            status.progress = 70

        # ── Step 4: Build PDFs ──
        status.step = StepEnum.BUILDING_PDF
        status.message = "Створюємо PDF файли..."

        output_dir = Path(tempfile.mkdtemp(prefix=f"docpilot_{task_id[:8]}_"))
        result_groups: list[Group] = []
        color_idx = 0

        for cat_name, items in groups.items():
            paths = [p for p, _, _ in items]
            confidences = [c for _, c, _ in items]
            methods = [m for _, _, m in items]
            avg_confidence = sum(confidences) // len(confidences) if confidences else 0

            try:
                pdf_path = output_dir / f"{safe_filename(cat_name)}.pdf"
                build_pdf(paths, output_path=pdf_path)
            except Exception:
                logger.warning("pdf_build_failed", category=cat_name, exc_info=True)

            group_files = []
            for p in paths:
                group_files.append(GroupFile(
                    name=p.name,
                    ext=p.suffix.lstrip(".").upper(),
                    size=format_file_size(p.stat().st_size),
                ))

            result_groups.append(Group(
                name=cat_name,
                color=GROUP_COLORS[color_idx % len(GROUP_COLORS)],
                files=group_files,
                file_count=len(paths),
                confidence=avg_confidence,
                method=methods[0] if methods else "none",
            ))
            color_idx += 1

        status.progress = 95

        # ── Done ──
        status.step = StepEnum.DONE
        status.progress = 100
        status.message = "Готово!"

        elapsed = time.time() - start_time
        logger.info(
            "pipeline_complete",
            task_id=task_id,
            files=total,
            time=round(elapsed, 1),
            gpu=GPU_AVAILABLE,
        )

        task_results[task_id] = TaskResult(
            task_id=task_id,
            groups=result_groups,
            unclassified_count=len(unclassified),
            total_files=total,
            duplicates_removed=duplicates_removed,
            processing_time=round(elapsed, 1),
        )
        task_output_dirs[task_id] = output_dir

    except Exception:
        logger.error("pipeline_error", task_id=task_id, exc_info=True)
        status.step = StepEnum.ERROR
        status.message = "Помилка обробки"
