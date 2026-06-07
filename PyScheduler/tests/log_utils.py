from pathlib import Path


class Logger:
    """Минимальный logger для старых FD scheduler tests.

    Тесты используют его только для человекочитаемого отчёта; на assertions
    он не влияет, кроме check(), который вызывает переданную lambda и делает assert.
    """

    def __init__(self, filename="test.log"):
        self.filename = filename
        self.passed = 0
        self.failed = 0
        self._lines = []
        Path("logs").mkdir(exist_ok=True)
        self.path = Path("logs") / filename

    def _write(self, message: str) -> None:
        self._lines.append(str(message))

    def section(self, title: str) -> None:
        self._write(f"\n=== {title} ===")

    def info(self, message: str) -> None:
        self._write(message)

    def check(self, message: str, predicate) -> None:
        ok = bool(predicate())
        if ok:
            self.passed += 1
            self._write(f"[OK] {message}")
        else:
            self.failed += 1
            self._write(f"[FAIL] {message}")
        assert ok, message

    def summary(self) -> None:
        self._write(f"\nSummary: passed={self.passed}, failed={self.failed}")
        self.path.write_text("\n".join(self._lines), encoding="utf-8")
