"""Điểm vào và giao diện desktop của Vietnam LPR."""

from __future__ import annotations

import logging
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk

from app_events import AppEvents
from app_runtime import (
    APP_NAME,
    APP_VERSION,
    AppPaths,
    AppSettings,
    SingleInstance,
    install_exception_logging,
    load_settings,
    setup_logging,
    verify_model,
)
from app_worker import RecognitionWorker
from history_store import HistoryStore, RecognitionRecord
from lpr_pipeline import LicensePlateRecognizer, find_best_model, find_resource


logger = logging.getLogger(__name__)
BG = "#0f172a"
PANEL = "#172033"
PANEL_ALT = "#111a2c"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
ACCENT = "#22c55e"
ACCENT_HOVER = "#16a34a"


class VietnamLPRApp:
    def __init__(self, root: tk.Tk, paths: AppPaths, settings: AppSettings):
        self.root = root
        self.paths = paths
        self.settings = settings
        self.store = HistoryStore(paths.database_file, settings.history_limit)
        self.events = AppEvents()
        self.stop_event = threading.Event()
        self.worker = RecognitionWorker(self.events, self.stop_event, settings)
        self.worker_thread: threading.Thread | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_frame: np.ndarray | None = None
        self.closing = False
        self.close_deadline = 0.0

        self.root.title(f"{APP_NAME} · Nhận dạng biển số")
        self.root.geometry("1180x720")
        self.root.minsize(920, 600)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        icon = find_resource("assets/app.ico")
        if icon:
            try:
                self.root.iconbitmap(icon)
            except tk.TclError:
                logger.warning("Không tải được icon ứng dụng", exc_info=True)

        self._configure_styles()
        self._build_ui()
        self._load_history()
        self.root.after(35, self._poll_events)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Title.TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI Semibold", 20))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED,
                        font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=PANEL, foreground=TEXT,
                        font=("Segoe UI Semibold", 12))
        style.configure("Status.TLabel", background=PANEL_ALT, foreground=MUTED,
                        font=("Segoe UI", 10), padding=(12, 9))
        style.configure("Action.TButton", font=("Segoe UI Semibold", 10), padding=(14, 9),
                        background=ACCENT, foreground="#052e16", borderwidth=0)
        style.map("Action.TButton", background=[("active", ACCENT_HOVER), ("disabled", "#334155")],
                  foreground=[("disabled", "#94a3b8")])
        style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(13, 9),
                        background="#263449", foreground=TEXT, borderwidth=0)
        style.map("Secondary.TButton", background=[("active", "#334155")])
        style.configure("Danger.TButton", font=("Segoe UI Semibold", 10), padding=(13, 9),
                        background="#7f1d1d", foreground="#fecaca", borderwidth=0)
        style.map("Danger.TButton", background=[("active", "#991b1b")])
        style.configure("Treeview", background=PANEL_ALT, fieldbackground=PANEL_ALT,
                        foreground=TEXT, rowheight=34, borderwidth=0, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#263449", foreground=MUTED,
                        font=("Segoe UI Semibold", 9), borderwidth=0)
        style.map("Treeview", background=[("selected", "#1e3a5f")])

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=(24, 18, 24, 12))
        header.pack(fill="x")
        title_block = ttk.Frame(header)
        title_block.pack(side="left")
        ttk.Label(title_block, text=APP_NAME, style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title_block,
            text=f"Nhận dạng biển số Việt Nam · offline · v{APP_VERSION}",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        controls = ttk.Frame(header)
        controls.pack(side="right")
        self.image_button = ttk.Button(
            controls, text="Mở ảnh", style="Secondary.TButton", command=self._choose_image,
        )
        self.image_button.pack(side="left", padx=4)
        self.video_button = ttk.Button(
            controls, text="Mở video", style="Secondary.TButton", command=self._choose_video,
        )
        self.video_button.pack(side="left", padx=4)
        self.camera_button = ttk.Button(
            controls, text="Bật camera", style="Action.TButton",
            command=lambda: self._start(
                "camera", self.settings.camera_index, f"Camera {self.settings.camera_index}",
            ),
        )
        self.camera_button.pack(side="left", padx=4)
        self.stop_button = ttk.Button(
            controls, text="Dừng", style="Danger.TButton", command=self._stop, state="disabled",
        )
        self.stop_button.pack(side="left", padx=(8, 0))

        content = ttk.Frame(self.root, padding=(24, 0, 24, 18))
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=7)
        content.columnconfigure(1, weight=3)
        content.rowconfigure(0, weight=1)

        preview_panel = ttk.Frame(content, style="Panel.TFrame", padding=12)
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        preview_panel.rowconfigure(1, weight=1)
        preview_panel.columnconfigure(0, weight=1)
        self.source_label = ttk.Label(
            preview_panel, text="Chưa chọn nguồn", style="PanelTitle.TLabel",
        )
        self.source_label.grid(row=0, column=0, sticky="w", pady=(0, 9))
        self.preview = tk.Label(
            preview_panel, text="Chọn ảnh, video hoặc bật camera để bắt đầu",
            bg="#090f1d", fg=MUTED, font=("Segoe UI", 11), compound="center",
        )
        self.preview.grid(row=1, column=0, sticky="nsew")
        self.preview.bind("<Configure>", lambda _event: self._render_frame())

        history_panel = ttk.Frame(content, style="Panel.TFrame", padding=12)
        history_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        history_panel.rowconfigure(1, weight=1)
        history_panel.columnconfigure(0, weight=1)
        history_header = ttk.Frame(history_panel, style="Panel.TFrame")
        history_header.grid(row=0, column=0, sticky="ew", pady=(0, 9))
        ttk.Label(history_header, text="Lịch sử nhận dạng", style="PanelTitle.TLabel").pack(side="left")
        ttk.Button(
            history_header, text="Xuất CSV", style="Secondary.TButton", command=self._export_history,
        ).pack(side="right", padx=(4, 0))
        ttk.Button(
            history_header, text="Xóa", style="Secondary.TButton", command=self._clear_history,
        ).pack(side="right")

        self.history = ttk.Treeview(
            history_panel, columns=("plate", "confidence", "time"),
            show="headings", selectmode="browse",
        )
        self.history.heading("plate", text="BIỂN SỐ")
        self.history.heading("confidence", text="TIN CẬY")
        self.history.heading("time", text="THỜI GIAN")
        self.history.column("plate", width=130, minwidth=95, anchor="w")
        self.history.column("confidence", width=78, minwidth=65, anchor="center")
        self.history.column("time", width=78, minwidth=65, anchor="center")
        self.history.grid(row=1, column=0, sticky="nsew")

        self.status = ttk.Label(self.root, text="Sẵn sàng", style="Status.TLabel", anchor="w")
        self.status.pack(fill="x", side="bottom")

    def _choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Tất cả", "*.*")],
        )
        if path:
            self._start("image", path, Path(path).name)

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
        self.source_label.configure(text=label)
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
        self.status.configure(text="Đang dừng an toàn…")
        self.stop_button.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        source_state = "disabled" if running or self.closing else "normal"
        self.image_button.configure(state=source_state)
        self.video_button.configure(state=source_state)
        self.camera_button.configure(state=source_state)
        self.stop_button.configure(state="normal" if running and not self.closing else "disabled")

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
                messagebox.showerror(APP_NAME, payload)
            elif kind == "done":
                self._set_running(False)
        if not self.closing:
            self.root.after(35, self._poll_events)

    def _render_frame(self) -> None:
        if self.last_frame is None:
            return
        width = max(64, self.preview.winfo_width() - 4)
        height = max(64, self.preview.winfo_height() - 4)
        rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")

    @staticmethod
    def _local_time(captured_at: str) -> str:
        try:
            return datetime.fromisoformat(captured_at).astimezone().strftime("%d/%m %H:%M:%S")
        except ValueError:
            return captured_at

    def _insert_history(self, record: RecognitionRecord, at_top: bool = True) -> None:
        self.history.insert(
            "", 0 if at_top else "end",
            values=(record.plate, f"{record.ocr_confidence:.0%}", self._local_time(record.captured_at)),
        )
        items = self.history.get_children()
        for item in items[100:]:
            self.history.delete(item)

    def _load_history(self) -> None:
        for record in self.store.recent(100):
            self._insert_history(record, at_top=False)

    def _add_plate(self, result: dict) -> None:
        text = str(result.get("text", ""))
        if not text:
            return
        record = RecognitionRecord.create(
            plate=text,
            ocr_confidence=float(result.get("ocr_conf", 0.0)),
            detection_confidence=float(result.get("detection_conf", 0.0)),
            is_valid=bool(result.get("is_valid", False)),
            source_type=str(result.get("source_type", "unknown")),
            source_name=str(result.get("source_name", "unknown")),
        )
        try:
            self.store.add(record)
            self._insert_history(record)
        except Exception:
            logger.exception("Không lưu được lịch sử")
            self.status.configure(text="Cảnh báo: không lưu được lịch sử nhận dạng")

    def _clear_history(self) -> None:
        if not messagebox.askyesno(APP_NAME, "Xóa toàn bộ lịch sử nhận dạng đã lưu?"):
            return
        self.store.clear()
        for item in self.history.get_children():
            self.history.delete(item)
        self.status.configure(text="Đã xóa lịch sử")

    def _export_history(self) -> None:
        destination = filedialog.asksaveasfilename(
            title="Xuất lịch sử",
            defaultextension=".csv",
            initialfile=f"vietnam-lpr-{datetime.now():%Y%m%d-%H%M}.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not destination:
            return
        try:
            count = self.store.export_csv(destination)
            self.status.configure(text=f"Đã xuất {count} bản ghi")
        except OSError as exc:
            logger.exception("Không xuất được CSV")
            messagebox.showerror(APP_NAME, f"Không thể ghi file CSV:\n{exc}")

    def _close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.close_deadline = time.monotonic() + 4.0
        self.stop_event.set()
        self._set_running(False)
        self.status.configure(text="Đang đóng tài nguyên…")
        self._finish_close()

    def _finish_close(self) -> None:
        if (
            self.worker_thread
            and self.worker_thread.is_alive()
            and time.monotonic() < self.close_deadline
        ):
            self.root.after(50, self._finish_close)
            return
        if self.worker_thread and self.worker_thread.is_alive():
            logger.warning("Worker chưa dừng sau timeout")
        self.store.close()
        self.root.destroy()


def run_smoke_test() -> None:
    model = find_best_model()
    if not model:
        raise FileNotFoundError("Không tìm thấy model")
    verify_model(model)
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
    with SingleInstance() as instance:
        if instance.already_running:
            ctypes_message = "Vietnam LPR đang chạy. Hãy mở cửa sổ hiện có."
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, ctypes_message, APP_NAME, 0x40)
            return
        settings = load_settings(paths)
        root = tk.Tk()
        VietnamLPRApp(root, paths, settings)
        root.mainloop()


if __name__ == "__main__":
    main()
