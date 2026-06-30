"""Kênh sự kiện có back-pressure cho worker và UI."""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np


class AppEvents:
    """Giữ duy nhất frame mới nhất, không để video làm đầy RAM."""

    def __init__(self, capacity: int = 256):
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=capacity)
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None

    def publish(self, kind: str, payload: Any = None) -> None:
        if kind == "frame":
            with self._frame_lock:
                self._latest_frame = payload
            return
        try:
            self._events.put((kind, payload), timeout=1.0)
        except queue.Full:
            # Không được block worker khi UI đang đóng. Status có thể bỏ qua;
            # event quan trọng thay thế event cũ nhất trong tình huống cực hạn.
            if kind == "status":
                return
            try:
                self._events.get_nowait()
                self._events.put_nowait((kind, payload))
            except queue.Empty:
                pass

    def take_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
            return frame

    def drain(self, limit: int = 100) -> list[tuple[str, Any]]:
        drained: list[tuple[str, Any]] = []
        for _ in range(limit):
            try:
                drained.append(self._events.get_nowait())
            except queue.Empty:
                break
        return drained
