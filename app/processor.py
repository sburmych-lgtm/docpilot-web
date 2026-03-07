"""Pipeline processor — OCR, dedup, classify, build PDFs."""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog

from app.dedup import find_duplicates
from app.gemini import classify_text
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


def get_reader() -> Any:
    """Get or create the EasyOCR Reader singleton."""
    global _reader  # noqa: PLW0603
    if _reader is None:
        import easyocr  # type: ignore[import-untyped]

        _reader = easyocr.Reader(["uk", "en"], gpu=False, verbose=False)
        logger.info("easyocr_reader_initialized")
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


def format_file_size(size_bytes: int) -> str:
    """Format file size to human-readable string."""
    if size_bytes > 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    return f"{size_bytes // 1024} KB"


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
        return "\n".join(results) if results else ""
    except Exception:
        logger.warning("ocr_failed", path=str(file_path), exc_info=True)
        return ""


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


def run_pipeline(
    task_id: str,
    tasks: dict[str, TaskStatus],
    file_paths: list[Path],
    task_results: dict,
    task_output_dirs: dict,
) -> None:
    """Run the full processing pipeline.

    Updates tasks[task_id] in-place with progress.
    """
    start_time = time.time()
    status = tasks[task_id]

    try:
        total = len(file_paths)

        # ── Step 1: OCR ──
        status.step = StepEnum.OCR
        status.message = "Читаємо текст документів..."

        ocr_results: dict[Path, str] = {}
        for i, path in enumerate(file_paths):
            for fs in status.files:
                if fs.name == path.name:
                    fs.status = FileStatusEnum.PROCESSING
                    break

            text = ocr_image(path)
            ocr_results[path] = text

            for fs in status.files:
                if fs.name == path.name:
                    fs.status = FileStatusEnum.DONE
                    break

            status.progress = int((i + 1) / total * 40)  # OCR = 0-40%

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

        for path, text in ocr_results.items():
            if not text.strip():
                unclassified.append(path)
                continue

            # Waterfall: Regex → Keyword → Gemini
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
