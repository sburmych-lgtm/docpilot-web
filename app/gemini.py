"""Gemini API classifier — fallback for documents not matched by regex/keyword."""

from __future__ import annotations

import json
import os
import time
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


def _call_gemini(api_key: str, prompt: str) -> dict | None:
    """Call Gemini API and parse JSON response."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
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
