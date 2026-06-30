"""Hạ tầng runtime: cấu hình, đường dẫn, logging và kiểm tra tính toàn vẹn."""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import logging.handlers
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any


APP_NAME = "Vietnam LPR"
APP_ID = "TM22187.VietnamLPR"
APP_VERSION = "1.1.0"
MODEL_FILENAME = "best_vietnam_lpr.onnx"
MODEL_SHA256 = "8893a6333e6fda86a47cc36294a40a9c26422e71f42de75f0046d9f0c7a986e4"


@dataclass(frozen=True)
class AppPaths:
    root: Path
    logs: Path
    data: Path
    config_file: Path
    database_file: Path

    @classmethod
    def discover(cls, base: Path | None = None) -> "AppPaths":
        if base is None:
            local = os.environ.get("LOCALAPPDATA")
            base = Path(local) / "VietnamLPR" if local else Path.home() / ".vietnam-lpr"
        root = Path(base)
        paths = cls(
            root=root,
            logs=root / "logs",
            data=root / "data",
            config_file=root / "config.json",
            database_file=root / "data" / "recognitions.db",
        )
        paths.logs.mkdir(parents=True, exist_ok=True)
        paths.data.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass(frozen=True)
class AppSettings:
    detector_confidence: float = 0.30
    video_target_fps: float = 12.0
    camera_index: int = 0
    history_limit: int = 10_000

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "AppSettings":
        try:
            confidence = min(0.95, max(0.05, float(values.get("detector_confidence", 0.30))))
            video_fps = min(30.0, max(1.0, float(values.get("video_target_fps", 12.0))))
            camera_index = min(16, max(0, int(values.get("camera_index", 0))))
            history_limit = min(100_000, max(100, int(values.get("history_limit", 10_000))))
        except (TypeError, ValueError):
            return cls()
        return cls(confidence, video_fps, camera_index, history_limit)


def load_settings(paths: AppPaths) -> AppSettings:
    if not paths.config_file.is_file():
        settings = AppSettings()
        save_settings(paths, settings)
        return settings
    try:
        values = json.loads(paths.config_file.read_text(encoding="utf-8"))
        return AppSettings.from_mapping(values if isinstance(values, dict) else {})
    except (OSError, json.JSONDecodeError):
        logging.getLogger(__name__).warning("Cấu hình lỗi; dùng giá trị mặc định", exc_info=True)
        try:
            corrupted = paths.config_file.with_suffix(".corrupt.json")
            if paths.config_file.exists():
                paths.config_file.replace(corrupted)
        except OSError:
            logging.getLogger(__name__).warning("Không cách ly được file cấu hình lỗi")
        settings = AppSettings()
        save_settings(paths, settings)
        return settings


def save_settings(paths: AppPaths, settings: AppSettings) -> None:
    temporary = paths.config_file.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(paths.config_file)


def setup_logging(paths: AppPaths) -> Path:
    log_file = paths.logs / "vietnam-lpr.log"
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if not getattr(sys, "frozen", False):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    logging.getLogger(__name__).info("Khởi động %s %s", APP_NAME, APP_VERSION)
    return log_file


def install_exception_logging() -> None:
    logger = logging.getLogger("crash")

    def handle_exception(
        exception_type: type[BaseException],
        exception: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        if issubclass(exception_type, KeyboardInterrupt):
            sys.__excepthook__(exception_type, exception, traceback)
            return
        logger.critical("Ngoại lệ chưa xử lý", exc_info=(exception_type, exception, traceback))

    def handle_thread(args: threading.ExceptHookArgs) -> None:
        logger.critical(
            "Luồng %s dừng do ngoại lệ",
            args.thread.name if args.thread else "unknown",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = handle_exception
    threading.excepthook = handle_thread


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: str | Path, expected_hash: str = MODEL_SHA256) -> None:
    model = Path(path)
    if not model.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {model}")
    actual = sha256_file(model)
    if actual.lower() != expected_hash.lower():
        raise RuntimeError(
            "Model bị sai hoặc hỏng. Hãy cài lại ứng dụng "
            f"(SHA-256 nhận được: {actual[:12]}…)."
        )


class SingleInstance:
    """Named mutex trên Windows để tránh hai tiến trình tranh camera/database."""

    ERROR_ALREADY_EXISTS = 183

    def __init__(self, name: str = APP_ID):
        self.handle: int | None = None
        self._kernel32: Any = None
        self.already_running = False
        if os.name == "nt":
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
            self._kernel32.CreateMutexW.restype = ctypes.c_void_p
            self._kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            self._kernel32.CloseHandle.restype = ctypes.c_bool
            self.handle = self._kernel32.CreateMutexW(None, False, f"Local\\{name}")
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            self.already_running = ctypes.get_last_error() == self.ERROR_ALREADY_EXISTS

    def close(self) -> None:
        if self.handle and self._kernel32 is not None:
            self._kernel32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self) -> "SingleInstance":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
