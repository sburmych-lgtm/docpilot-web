"""DocPilot Desktop — standalone GUI (no browser needed)."""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Ensure app package is importable
sys.path.insert(0, str(Path(__file__).parent))

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".docx", ".xlsx"}

# ── Color scheme (dark theme matching web UI) ──
BG = "#111111"
BG2 = "#1a1a1a"
BG3 = "#222222"
FG = "#e5e5e5"
FG2 = "#888888"
ACCENT = "#22c55e"
ACCENT2 = "#3b82f6"
RED = "#ef4444"
BORDER = "#333333"


class DocPilotApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DocPilot — Сортувальник документів")
        self.root.geometry("750x600")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.files: list[Path] = []
        self.processing = False

        self._build_ui()

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg=BG, pady=12)
        header.pack(fill="x", padx=20)

        tk.Label(
            header, text="◼ DocPilot", font=("Segoe UI", 16, "bold"),
            bg=BG, fg=FG,
        ).pack(side="left")

        tk.Label(
            header, text="Сортувальник документів", font=("Segoe UI", 10),
            bg=BG, fg=FG2,
        ).pack(side="left", padx=(10, 0))

        # ── Drop area ──
        drop_frame = tk.Frame(self.root, bg=BG3, highlightbackground=BORDER,
                              highlightthickness=1, padx=30, pady=30)
        drop_frame.pack(fill="x", padx=20, pady=(0, 10))

        tk.Label(
            drop_frame, text="Обери файли для сортування",
            font=("Segoe UI", 13, "bold"), bg=BG3, fg=FG,
        ).pack()

        tk.Label(
            drop_frame, text="PDF, DOCX, XLSX, JPG, PNG — без обмежень",
            font=("Segoe UI", 9), bg=BG3, fg=FG2,
        ).pack(pady=(4, 12))

        btn_frame = tk.Frame(drop_frame, bg=BG3)
        btn_frame.pack()

        self.btn_files = tk.Button(
            btn_frame, text="📁  Обрати файли", font=("Segoe UI", 10),
            bg=BG2, fg=FG, activebackground=BG3, activeforeground=FG,
            relief="solid", borderwidth=1, padx=16, pady=6,
            command=self._pick_files, cursor="hand2",
        )
        self.btn_files.pack(side="left", padx=4)

        self.btn_folder = tk.Button(
            btn_frame, text="📂  Обрати папку", font=("Segoe UI", 10),
            bg=BG2, fg=FG, activebackground=BG3, activeforeground=FG,
            relief="solid", borderwidth=1, padx=16, pady=6,
            command=self._pick_folder, cursor="hand2",
        )
        self.btn_folder.pack(side="left", padx=4)

        # ── File count label ──
        self.lbl_count = tk.Label(
            self.root, text="Файлів обрано: 0", font=("Segoe UI", 10),
            bg=BG, fg=FG2,
        )
        self.lbl_count.pack(pady=(4, 4))

        # ── Progress ──
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "green.Horizontal.TProgressbar",
            troughcolor=BG3, background=ACCENT, thickness=6,
        )

        self.progress = ttk.Progressbar(
            self.root, orient="horizontal", mode="determinate",
            style="green.Horizontal.TProgressbar",
        )
        self.progress.pack(fill="x", padx=20, pady=(0, 4))

        self.lbl_status = tk.Label(
            self.root, text="", font=("Segoe UI", 10), bg=BG, fg=FG2,
        )
        self.lbl_status.pack()

        # ── Start button ──
        self.btn_start = tk.Button(
            self.root, text="▶  Почати сортування", font=("Segoe UI", 11, "bold"),
            bg=ACCENT, fg="#000000", activebackground="#16a34a", activeforeground="#000",
            relief="flat", padx=24, pady=8, state="disabled",
            command=self._start_processing, cursor="hand2",
        )
        self.btn_start.pack(pady=10)

        # ── Results area (scrollable) ──
        results_frame = tk.Frame(self.root, bg=BG)
        results_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        self.results_text = tk.Text(
            results_frame, bg=BG2, fg=FG, font=("Consolas", 10),
            relief="flat", borderwidth=0, wrap="word", state="disabled",
            insertbackground=FG,
        )
        scrollbar = tk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.results_text.pack(fill="both", expand=True)

        # Configure text tags for colors
        self.results_text.tag_configure("header", foreground=ACCENT, font=("Consolas", 11, "bold"))
        self.results_text.tag_configure("group", foreground=ACCENT2, font=("Consolas", 10, "bold"))
        self.results_text.tag_configure("file", foreground=FG2)
        self.results_text.tag_configure("info", foreground=FG)
        self.results_text.tag_configure("error", foreground=RED)

    def _pick_files(self):
        paths = filedialog.askopenfilenames(
            title="Обери документи",
            filetypes=[
                ("Документи", "*.pdf *.jpg *.jpeg *.png *.tiff *.tif *.bmp *.docx *.xlsx"),
                ("Всі файли", "*.*"),
            ],
        )
        if paths:
            self.files = [Path(p) for p in paths if Path(p).suffix.lower() in ALLOWED_EXTENSIONS]
            self._update_count()

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Обери папку з документами")
        if folder:
            folder_path = Path(folder)
            self.files = [
                p for p in folder_path.rglob("*")
                if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
            ]
            self._update_count()

    def _update_count(self):
        n = len(self.files)
        self.lbl_count.config(text=f"Файлів обрано: {n}")
        self.btn_start.config(state="normal" if n >= 2 else "disabled")

    def _log(self, text: str, tag: str = "info"):
        self.results_text.config(state="normal")
        self.results_text.insert("end", text + "\n", tag)
        self.results_text.see("end")
        self.results_text.config(state="disabled")

    def _clear_log(self):
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.config(state="disabled")

    def _start_processing(self):
        if self.processing or len(self.files) < 2:
            return

        self.processing = True
        self.btn_start.config(state="disabled", text="⏳  Обробка...")
        self.btn_files.config(state="disabled")
        self.btn_folder.config(state="disabled")
        self._clear_log()
        self.progress["value"] = 0

        threading.Thread(target=self._process, daemon=True).start()

    def _update_ui(self, func):
        """Schedule a function to run on the main thread."""
        self.root.after(0, func)

    def _process(self):
        """Run processing in a background thread."""
        import tempfile

        start_time = time.time()

        try:
            total = len(self.files)
            self._update_ui(lambda: self.lbl_status.config(text="Читаємо текст документів..."))
            self._update_ui(lambda: self._log(f"Обробка {total} файлів...", "header"))

            # ── Step 1: Extract text ──
            from app.processor import extract_text
            ocr_results: dict[Path, str] = {}

            for i, path in enumerate(self.files):
                text = extract_text(path)
                ocr_results[path] = text
                pct = int((i + 1) / total * 40)
                name = path.name
                self._update_ui(lambda p=pct: self.progress.configure(value=p))
                self._update_ui(lambda n=name: self.lbl_status.config(text=f"OCR: {n}"))

            # ── Step 2: Dedup ──
            self._update_ui(lambda: self.lbl_status.config(text="Шукаємо дублікати..."))
            self._update_ui(lambda: self.progress.configure(value=45))

            from app.dedup import find_duplicates
            unique_paths, dup_paths = find_duplicates(self.files)
            for dp in dup_paths:
                ocr_results.pop(dp, None)

            if dup_paths:
                self._update_ui(lambda: self._log(f"Видалено дублікатів: {len(dup_paths)}", "info"))

            self._update_ui(lambda: self.progress.configure(value=50))

            # ── Step 3: Classify ──
            self._update_ui(lambda: self.lbl_status.config(text="Класифікуємо документи..."))

            from app.processor import classify_by_gemini, classify_by_keyword, classify_by_regex

            groups: dict[str, list[tuple[Path, int, str]]] = {}
            unclassified: list[Path] = []

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
                groups["Невідомі документи"] = [(p, 0, "none") for p in unclassified]

            self._update_ui(lambda: self.progress.configure(value=70))

            # ── Step 4: Build PDFs ──
            self._update_ui(lambda: self.lbl_status.config(text="Створюємо PDF файли..."))

            from app.pdf_builder import build_pdf, safe_filename

            output_dir = Path(tempfile.mkdtemp(prefix="docpilot_desktop_"))

            for cat_name, items in groups.items():
                paths = [p for p, _, _ in items]
                try:
                    pdf_path = output_dir / f"{safe_filename(cat_name)}.pdf"
                    build_pdf(paths, output_path=pdf_path)
                except Exception as e:
                    self._update_ui(lambda e=e: self._log(f"  ⚠ Помилка: {e}", "error"))

            self._update_ui(lambda: self.progress.configure(value=95))

            # ── Done — show results ──
            elapsed = time.time() - start_time

            self._update_ui(lambda: self._log("", "info"))
            self._update_ui(lambda: self._log(f"✅ Готово за {elapsed:.1f} сек", "header"))
            self._update_ui(lambda: self._log(f"Файлів оброблено: {total}", "info"))
            if dup_paths:
                self._update_ui(lambda: self._log(f"Дублікатів видалено: {len(dup_paths)}", "info"))
            self._update_ui(lambda: self._log("", "info"))

            for cat_name, items in groups.items():
                n = len(items)
                self._update_ui(lambda c=cat_name, n=n: self._log(f"📁 {c} ({n} файлів)", "group"))
                for path, conf, method in items:
                    self._update_ui(
                        lambda p=path.name, m=method, c=conf: self._log(f"   {p}  [{m} {c}%]", "file")
                    )

            self._update_ui(lambda: self._log("", "info"))
            self._update_ui(lambda d=str(output_dir): self._log(f"PDF збережено: {d}", "info"))

            self._update_ui(lambda: self.progress.configure(value=100))
            self._update_ui(lambda: self.lbl_status.config(text="Готово!"))

            # Ask where to save
            self._update_ui(lambda od=output_dir: self._ask_save(od))

        except Exception as e:
            self._update_ui(lambda e=e: self._log(f"❌ Помилка: {e}", "error"))
            self._update_ui(lambda: self.lbl_status.config(text="Помилка обробки"))

        finally:
            self._update_ui(lambda: self.btn_start.config(state="normal", text="▶  Почати сортування"))
            self._update_ui(lambda: self.btn_files.config(state="normal"))
            self._update_ui(lambda: self.btn_folder.config(state="normal"))
            self.processing = False

    def _ask_save(self, output_dir: Path):
        """Ask user where to save results."""
        save_to = filedialog.askdirectory(title="Куди зберегти результат?")
        if save_to:
            save_path = Path(save_to) / "DocPilot_результат"
            if save_path.exists():
                shutil.rmtree(save_path)
            shutil.copytree(output_dir, save_path)
            self._log(f"💾 Збережено в: {save_path}", "header")
            os.startfile(str(save_path))  # open folder in Explorer
        # Cleanup temp
        shutil.rmtree(output_dir, ignore_errors=True)


def main():
    root = tk.Tk()

    # Try to set icon
    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    app = DocPilotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
