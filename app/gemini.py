"""Gemini API classifier — fallback for documents not matched by regex/keyword."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import structlog

logger = structlog.get_logger()

_CLASSIFY_PROMPT = (
    "Ти класифікуєш українські юридичні документи.\n"
    "OCR текст документа:\n{text}\n\n"
    "Визнач:\n"
    "1. Тип документа (позовна заява, рішення суду, ухвала, клопотання, "
    "апеляційна скарга, касаційна скарга, постанова, судовий наказ, "
    "виконавчий лист, протокол, довіреність, договір, акт звірки, рахунок, або інше)\n"
    "2. Номер справи якщо є (формат: 450/527/18)\n"
    "3. Впевненість від 0 до 1\n\n"
    'Відповідь строго у JSON: {{"doc_type": "...", "case_number": "...", "confidence": 0.9}}'
)

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]


def classify_text(text: str) -> dict | None:
    """Classify document text using Gemini API.

    Args:
        text: OCR text (max 2000 chars will be used).

    Returns:
        Dict with doc_type, case_number, confidence or None on failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("gemini_no_api_key")
        return None

    prompt = _CLASSIFY_PROMPT.format(text=text[:2000])

    for attempt in range(MAX_RETRIES):
        try:
            result = _call_gemini(api_key, prompt)
            if result:
                return result
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.error("gemini_quota_exhausted_single", attempt=attempt + 1)
            else:
                logger.warning("gemini_http_error", code=e.code, attempt=attempt + 1)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
        except Exception:
            logger.warning(
                "gemini_retry",
                attempt=attempt + 1,
                exc_info=True,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])

    logger.warning("gemini_all_retries_failed")
    return None


def classify_batch(
    texts: dict[str, str],
    instruction: str,
) -> dict[str, dict] | None:
    """Classify multiple documents using Gemini with user instruction.

    Args:
        texts: dict mapping file_id to OCR text.
        instruction: User instruction for how to group documents.

    Returns:
        Dict mapping file_id to classification result, or None on failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("gemini_no_api_key_batch")
        return None

    # Build document summaries
    doc_list = []
    file_ids = list(texts.keys())
    for i, fid in enumerate(file_ids):
        snippet = texts[fid][:300].replace("\n", " ").strip()
        doc_list.append(f"[{i}] {snippet}")

    docs_text = "\n".join(doc_list)

    prompt = (
        f"Інструкція користувача: {instruction}\n\n"
        f"Є {len(file_ids)} документів. Ось фрагменти кожного:\n"
        f"{docs_text}\n\n"
        "Класифікуй КОЖЕН документ у відповідну групу згідно інструкції користувача.\n"
        "Кількість груп визначи згідно інструкції.\n"
        "Відповідь строго у JSON масиві:\n"
        '[{"index": 0, "group": "Назва групи", "confidence": 0.9}, ...]\n'
        "Масив має містити рівно " + str(len(file_ids)) + " елементів — по одному на кожен документ."
    )

    for attempt in range(MAX_RETRIES):
        try:
            result = _call_gemini_batch(api_key, prompt)
            if result is not None:
                output: dict[str, dict] = {}
                for item in result:
                    idx = item.get("index", -1)
                    if 0 <= idx < len(file_ids):
                        output[file_ids[idx]] = {
                            "group": item.get("group", "Невідомі"),
                            "confidence": item.get("confidence", 0.7),
                        }
                return output
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.error("gemini_quota_exhausted", attempt=attempt + 1)
            else:
                logger.warning("gemini_batch_http_error", code=e.code, attempt=attempt + 1)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
        except Exception:
            logger.warning("gemini_batch_retry", attempt=attempt + 1, exc_info=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])

    logger.error("gemini_batch_all_retries_failed", docs=len(file_ids))
    return None


def _call_gemini_batch(api_key: str, prompt: str) -> list[dict] | None:
    """Call Gemini API for batch classification."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        data = json.loads(resp.read())

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        logger.warning("gemini_batch_bad_response", data=data)
        return None

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        logger.warning("gemini_batch_json_error", text=text[:200])

    return None


def _call_gemini(api_key: str, prompt: str) -> dict | None:
    """Call Gemini API and parse JSON response."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 256,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read())

    # Extract text from Gemini response
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        logger.warning("gemini_bad_response", data=data)
        return None

    # Parse JSON from response (handle markdown code blocks)
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        result = json.loads(text)
        if "doc_type" in result:
            return result
    except json.JSONDecodeError:
        logger.warning("gemini_json_parse_error", text=text)

    return None
