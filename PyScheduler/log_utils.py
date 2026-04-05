from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Callable


class Logger:
    def __init__(self, filename: str, logs_dir: str | Path | None = None) -> None:
        base_dir = Path(logs_dir) if logs_dir is not None else Path(__file__).resolve().parent / "logs"
        base_dir.mkdir(parents=True, exist_ok=True)
        self.path = base_dir / filename
        self._lock = Lock()
        self.passed = 0
        self.failed = 0
        self.info(f"Log started: {self.path.name}")

    def _write(self, message: str) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(f"{message}\n")

    def section(self, title: str) -> None:
        line = "=" * 72
        self.info("")
        self.info(line)
        self.info(title)
        self.info(line)

    def info(self, message: str) -> None:
        self._write(str(message))

    def check(self, description: str, predicate: Callable[[], bool] | bool) -> None:
        try:
            result = predicate() if callable(predicate) else bool(predicate)
        except Exception as exc:
            self.failed += 1
            self.info(f"[ERROR] {description}: {exc}")
            raise

        if result:
            self.passed += 1
            self.info(f"[PASS] {description}")
            return

        self.failed += 1
        self.info(f"[FAIL] {description}")
        raise AssertionError(description)

    def summary(self) -> None:
        total = self.passed + self.failed
        self.info("")
        self.info(f"Summary: total={total}, passed={self.passed}, failed={self.failed}")
