"""File and perceptual hashing for deduplication."""

from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import structlog
from PIL import Image

logger = structlog.get_logger()


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of file contents."""
    md5 = hashlib.md5()  # noqa: S324
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def compute_perceptual_hash(path: Path) -> str:
    """Compute perceptual hash (pHash) for near-duplicate detection."""
    img = Image.open(path)
    return str(imagehash.phash(img))


def find_duplicates(
    file_paths: list[Path],
    *,
    threshold: int = 8,
) -> tuple[list[Path], list[Path]]:
    """Find and separate duplicate files.

    Uses MD5 for exact matches, then perceptual hash for near-duplicates.

    Args:
        file_paths: List of file paths to check.
        threshold: Hamming distance threshold for perceptual hash similarity.

    Returns:
        Tuple of (unique_paths, duplicate_paths).
    """
    if not file_paths:
        return [], []

    unique: list[Path] = []
    duplicates: list[Path] = []

    seen_file_hashes: set[str] = set()
    seen_perceptual: list[tuple[Path, imagehash.ImageHash]] = []

    for path in file_paths:
        try:
            file_hash = compute_file_hash(path)

            # Exact duplicate by MD5
            if file_hash in seen_file_hashes:
                duplicates.append(path)
                logger.debug("exact_duplicate", path=str(path))
                continue

            # Try perceptual hash for images
            try:
                phash_str = compute_perceptual_hash(path)
                current_phash = imagehash.hex_to_hash(phash_str)

                is_near_dup = False
                for _, existing_phash in seen_perceptual:
                    if current_phash - existing_phash <= threshold:
                        is_near_dup = True
                        break

                if is_near_dup:
                    duplicates.append(path)
                    logger.debug("near_duplicate", path=str(path))
                    continue

                seen_perceptual.append((path, current_phash))
            except Exception:
                # Not an image or phash failed — skip perceptual check
                pass

            seen_file_hashes.add(file_hash)
            unique.append(path)

        except Exception:
            logger.warning("dedup_error", path=str(path), exc_info=True)
            unique.append(path)  # Keep on error

    logger.info(
        "dedup_complete",
        total=len(file_paths),
        unique=len(unique),
        duplicates=len(duplicates),
    )
    return unique, duplicates
