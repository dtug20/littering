"""
Load & validate cấu hình YAML cho hệ thống AOD.
Cho phép reload runtime (hot-reload) khi file cấu hình thay đổi, để chỉnh
ROI/threshold/MQTT broker mà không cần restart pipeline DeepStream.
"""
import yaml
import os
import threading
import time
import logging

logger = logging.getLogger("aod.config")


class ConfigLoader:
    def __init__(self, path: str, watch: bool = False, poll_interval: float = 5.0):
        self.path = path
        self._lock = threading.RLock()
        self._mtime = None
        self._data = {}
        self._watch = watch
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = None
        self.reload()
        if watch:
            self._thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._thread.start()

    def reload(self):
        with self._lock:
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
            self._mtime = os.path.getmtime(self.path)
            logger.info(f"[ConfigLoader] Loaded config: {self.path}")

    def _watch_loop(self):
        while not self._stop.is_set():
            try:
                mtime = os.path.getmtime(self.path)
                if mtime != self._mtime:
                    logger.info(f"[ConfigLoader] Change detected in {self.path}, reloading...")
                    self.reload()
            except FileNotFoundError:
                logger.warning(f"[ConfigLoader] File not found: {self.path}")
            time.sleep(self._poll_interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    @property
    def data(self) -> dict:
        with self._lock:
            return self._data

    def get(self, dotted_key: str, default=None):
        """Lấy giá trị theo key dạng 'a.b.c'"""
        node = self.data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node
