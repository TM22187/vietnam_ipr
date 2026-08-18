"""Giao diện desktop tối giản — Vietnam LPR."""

from __future__ import annotations

import logging
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from app_events import AppEvents
from app_runtime import (
    APP_NAME,
    AppPaths,
    AppSettings,
    install_exception_logging,
    load_settings,
    setup_logging,
)
from app_worker import RecognitionWorker
from lpr_pipeline import LicensePlateRecognizer, find_best_model, find_resource

logger = logging.getLogger(__name__)

BG     = "#f5f5f5"
TEXT   = "#1a1a1a"
MUTED  = "#888888"
BORDER = "#d0d0d0"
ROW_BG = "#ffffff"
ROW_ALT = "#f0f0f0"


class VietnamLPRApp:
    def __init__(self, root: tk.Tk, paths: AppPaths, settings: AppSettings):
        self.root = root
        self.settings = settings
        self.events = AppEvents()
        self.stop_event = threading.Event()
        self.worker = RecognitionWorker(self.events, self.stop_event, settings)
        self.worker_thread: threading.Thread | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_frame: np.ndarray | None = None
        self.closing = False
        self.close_deadline = 0.0

        self.root.title(APP_NAME)
        self.root.geometry("900x560")
        self.root.minsize(680, 420)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        icon = find_resource("assets/app.ico")
        if icon:
            try:
                self.root.iconbitmap(icon)
            except tk.TclError:
                pass

        self._configure_styles()
        self._build_ui()
        self.root.after(35, self._poll_events)

    # ------------------------------------------------------------------ styles

    def _configure_styles(self) -> None:
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TFrame",        background=BG)
        s.configure("Status.TLabel", background=BORDER, foreground=MUTED,
                    font=("Segoe UI", 9), padding=(10, 6))
        s.configure("TButton", font=("Segoe UI", 9), padding=(12, 6),
                    background=ROW_BG, foreground=TEXT, borderwidth=1,
                    relief="solid")
        s.map("TButton",
              background=[("active", ROW_ALT), ("disabled", BG)],
              foreground=[("disabled", MUTED)])
        s.configure("Treeview",
                    background=ROW_BG, fieldbackground=ROW_BG,
                    foreground=TEXT, rowheight=28, borderwidth=0,
                    font=("Consolas", 11))
        s.configure("Treeview.Heading",
                    background=BG, foreground=MUTED,
                    font=("Segoe UI", 8), borderwidth=0)
        s.map("Treeview", background=[("selected", BORDER)])

    # -------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        # ── top bar ──────────────────────────────────────────────────────────
        bar = tk.Frame(self.root, bg=BG, pady=10, padx=16)
        bar.pack(fill="x")

        tk.Label(bar, text=APP_NAME, bg=BG, fg=TEXT,
                 font=("Segoe UI", 12)).pack(side="left")

        self.stop_btn = ttk.Button(bar, text="Dừng",
                                   command=self._stop, state="disabled")
        self.stop_btn.pack(side="right", padx=(4, 0))

        self.cam_btn = ttk.Button(
            bar, text="Camera",
            command=lambda: self._start(
                "camera",
                self.settings.camera_index,
                f"Camera {self.settings.camera_index}",
            ),
        )
        self.cam_btn.pack(side="right", padx=(4, 0))

        self.video_btn = ttk.Button(bar, text="Mở video",
                                    command=self._choose_video)
        self.video_btn.pack(side="right")

        # ── separator ────────────────────────────────────────────────────────
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        # ── main area (grid, tỷ lệ cố định 7:3) ─────────────────────────────
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=16, pady=12)
        content.columnconfigure(0, weight=7)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        # preview (cột 0)
        preview_wrap = tk.Frame(content, bg="#000000")
        preview_wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.preview = tk.Label(
            preview_wrap,
            text="Mở video hoặc bật camera để bắt đầu",
            bg="#000000", fg=MUTED,
            font=("Segoe UI", 10),
        )
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Configure>", lambda _e: self._render_frame())

        # plate list (cột 1)
        right = tk.Frame(content, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        tk.Label(right, text="Biển số", bg=BG, fg=MUTED,
                 font=("Segoe UI", 8)).grid(row=0, column=0, sticky="w", pady=(0, 4))

        self.plate_list = ttk.Treeview(
            right, columns=("plate", "time"),
            show="headings", selectmode="browse",
        )
        self.plate_list.heading("plate", text="SỐ HIỆU")
        self.plate_list.heading("time",  text="GIỜ")
        self.plate_list.column("plate", anchor="w")
        self.plate_list.column("time",  width=72, minwidth=60, anchor="center")
        self.plate_list.grid(row=1, column=0, sticky="nsew")

        # ── separator + status bar ───────────────────────────────────────────
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")
        self.status = ttk.Label(
            self.root, text="Sẵn sàng",
            style="Status.TLabel", anchor="w",
        )
        self.status.pack(fill="x")

    # ----------------------------------------------------------------- control

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("Tất cả", "*.*")],
        )
        if path:
            self._start("video", path, Path(path).name)

    def _start(self, source_type: str, source: str | int, label: str) -> None:
        if self.closing or (self.worker_thread and self.worker_thread.is_alive()):
            return
        self.stop_event.clear()
        self._set_running(True)
        self.worker_thread = threading.Thread(
            target=self.worker.run,
            args=(source_type, source, label),
            name="recognition-worker",
            daemon=True,
        )
        self.worker_thread.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.status.configure(text="Đang dừng…")
        self.stop_btn.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running or self.closing else "normal"
        self.cam_btn.configure(state=state)
        self.video_btn.configure(state=state)
        self.stop_btn.configure(state="normal" if running and not self.closing else "disabled")

    # ------------------------------------------------------------------ events

    def _poll_events(self) -> None:
        frame = self.events.take_frame()
        if frame is not None:
            self.last_frame = frame
            self._render_frame()
        for kind, payload in self.events.drain():
            if kind == "plate":
                self._add_plate(payload)
            elif kind == "status":
                self.status.configure(text=payload)
            elif kind == "error":
                self.status.configure(text=f"Lỗi: {payload}")
            elif kind == "done":
                self._set_running(False)
        if not self.closing:
            self.root.after(35, self._poll_events)

    def _render_frame(self) -> None:
        if self.last_frame is None:
            return
        w = max(64, self.preview.winfo_width()  - 2)
        h = max(64, self.preview.winfo_height() - 2)
        rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        image = ImageOps.contain(Image.fromarray(rgb), (w, h), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")

    def _add_plate(self, result: dict) -> None:
        text = str(result.get("text", ""))
        if not text:
            return
        now = datetime.now().strftime("%H:%M:%S")
        self.plate_list.insert("", 0, values=(text, now))
        for item in self.plate_list.get_children()[200:]:
            self.plate_list.delete(item)

    # ------------------------------------------------------------------- close

    def _close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.close_deadline = time.monotonic() + 4.0
        self.stop_event.set()
        self._set_running(False)
        self._finish_close()

    def _finish_close(self) -> None:
        if (self.worker_thread
                and self.worker_thread.is_alive()
                and time.monotonic() < self.close_deadline):
            self.root.after(50, self._finish_close)
            return
        self.root.destroy()


# --------------------------------------------------------------------------- #

def run_smoke_test() -> None:
    model = find_best_model()
    if not model:
        raise FileNotFoundError("Không tìm thấy model")
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    root.destroy()
    recognizer = LicensePlateRecognizer(model)
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    recognizer.detector.detect(blank)
    recognizer.ocr.read(blank[260:380, 160:480])


def main() -> None:
    paths = AppPaths.discover()
    setup_logging(paths)
    install_exception_logging()
    if "--smoke-test" in sys.argv:
        run_smoke_test()
        return
    settings = load_settings(paths)
    root = tk.Tk()
    VietnamLPRApp(root, paths, settings)
    root.mainloop()


if __name__ == "__main__":
    main()
