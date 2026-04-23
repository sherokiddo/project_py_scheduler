"""
Progress callback для отправки данных симуляции во внешние системы (WebSocket, API).
Используется sim-sched-api для стриминга прогресса.
"""
from typing import Callable, Optional, Dict, Any


class ProgressCallback:
    """
    Обёртка для отправки прогресса симуляции.
    
    Usage:
        callback = ProgressCallback()
        callback.set_handler(lambda data: print(data))
        callback.emit({"tti": 100, "progress": 10.0})
    """
    
    def __init__(self):
        self._handler: Optional[Callable[[Dict[str, Any]], None]] = None
    
    def set_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Установить обработчик для отправки данных."""
        self._handler = handler
    
    def emit(self, data: Dict[str, Any]):
        """Отправить данные (если handler установлен)."""
        if self._handler:
            try:
                self._handler(data)
            except Exception as e:
                print(f"[ProgressCallback] Error in handler: {e}")
    
    def is_active(self) -> bool:
        """Проверить, установлен ли обработчик."""
        return self._handler is not None
