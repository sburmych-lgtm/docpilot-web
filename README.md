# DocPilot

A web app (and standalone desktop GUI) for sorting Ukrainian legal documents: OCR + AI
classification, deduplication, and PDF assembly.

## Stack

- FastAPI + Uvicorn (server), `app/` — OCR (EasyOCR, OpenCV), AI classification (`gemini.py`),
  deduplication (`imagehash`), PDF building (`img2pdf`, `PyPDF2`, Pillow)
- `desktop.py` — standalone Tkinter GUI (no browser needed), same processing pipeline
- Deployed on Railway (`railway.toml`, `Dockerfile`)

## Running locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload   # web server
python desktop.py               # or the standalone desktop GUI
```

Windows convenience scripts: `install.bat`, `DocPilot.bat`.

## Layout

- `app/` — `main.py` (FastAPI app), `processor.py`, `dedup.py`, `gemini.py`, `pdf_builder.py`,
  `models.py`
- `static/` — web UI assets
- `desktop.py` — desktop GUI entrypoint

