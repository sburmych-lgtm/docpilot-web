"""PDF assembly — build organized PDFs from classified image groups."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import img2pdf  # type: ignore[import-untyped]
import structlog
from PyPDF2 import PdfReader, PdfWriter

logger = structlog.get_logger()

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
_PDF_EXTENSION = ".pdf"


def safe_filename(name: str) -> str:
    """Generate a filesystem-safe filename."""
    safe = re.sub(r'[/\\:*?"<>|]', "-", name)
    safe = safe.strip(" .-")
    return safe if safe else "невідомо"


def build_pdf(file_paths: list[Path], *, output_path: Path) -> Path:
    """Assemble a PDF from image and PDF files.

    Args:
        file_paths: List of source files (images or PDFs).
        output_path: Path for the output PDF file.

    Returns:
        Path to the created PDF.

    Raises:
        ValueError: If no valid pages could be assembled.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    pdf_paths: list[Path] = []

    for path in file_paths:
        ext = path.suffix.lower()
        if ext in _IMAGE_EXTENSIONS:
            image_paths.append(path)
        elif ext == _PDF_EXTENSION:
            pdf_paths.append(path)
        else:
            logger.warning("unsupported_file_type", path=str(path), ext=ext)

    writer = PdfWriter()

    # Convert and add images
    for img_path in image_paths:
        try:
            pdf_bytes = img2pdf.convert(str(img_path))
            if pdf_bytes:
                reader = PdfReader(BytesIO(pdf_bytes))
                for page in reader.pages:
                    writer.add_page(page)
        except Exception:
            logger.warning(
                "image_conversion_failed", path=str(img_path), exc_info=True
            )

    # Add existing PDF pages
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
            for page in reader.pages:
                writer.add_page(page)
        except Exception:
            logger.warning("pdf_read_failed", path=str(pdf_path), exc_info=True)

    if len(writer.pages) == 0:
        raise ValueError("No valid pages could be assembled")

    with open(output_path, "wb") as f:
        writer.write(f)

    logger.info("pdf_built", pages=len(writer.pages), output=str(output_path))
    return output_path
