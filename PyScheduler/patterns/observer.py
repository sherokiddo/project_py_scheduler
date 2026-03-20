"""
#------------------------------------------------------------------------------
# Паттерн: Observer — событийная шина симуляции
#------------------------------------------------------------------------------
# Описание:
#   EventBus реализует классический паттерн Observer (издатель/подписчик).
#   Модули публикуют типизированные события, остальные модули реагируют —
#   без прямых зависимостей друг от друга.
#
#   Типичное использование:
#
#       bus = EventBus()
#
#       # Подписка
#       def on_tti(event: SimulationEvent):
#           print(f"TTI {event.data['tti']} завершён")
#
#       bus.subscribe(EventType.TTI_COMPLETED, on_tti)
#
#       # Публикация (из основного цикла симуляции)
#       bus.publish(SimulationEvent(EventType.TTI_COMPLETED, data={'tti': 42}))
#
# Версия: 1.0.0
#------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional


# ==============================================================================
#                              ТИПЫ СОБЫТИЙ
# ==============================================================================

class EventType(Enum):
    """
    Каталог всех событий симуляции.

    Каждый модуль публикует события своего домена, остальные — подписываются.
    Добавлять новые типы можно без изменения существующего кода.
    """

    # --- Жизненный цикл симуляции ---
    SIMULATION_STARTED   = auto()   # Симуляция запущена
    SIMULATION_STOPPED   = auto()   # Симуляция остановлена (штатно)
    SIMULATION_ERROR     = auto()   # Аварийное завершение

    # --- TTI цикл ---
    TTI_STARTED          = auto()   # Начало нового TTI
    TTI_COMPLETED        = auto()   # TTI завершён, результат готов

    # --- UE события ---
    UE_REGISTERED        = auto()   # UE зарегистрировано на BS
    UE_DEREGISTERED      = auto()   # UE отключено от BS
    UE_POSITION_UPDATED  = auto()   # UE сменило координаты
    UE_BUFFER_OVERFLOW   = auto()   # Буфер UE переполнен

    # --- Планировщик ---
    SCHEDULING_COMPLETED = auto()   # Планировщик завершил распределение RB
    PDCCH_BLOCKED        = auto()   # UE не получило PDCCH ресурсы

    # --- Трафик ---
    PACKET_GENERATED     = auto()   # Пакет добавлен в буфер UE
    PACKET_TRANSMITTED   = auto()   # Пакет передан (извлечён из буфера)

    # --- Метрики ---
    METRICS_COLLECTED    = auto()   # StatsManager собрал snapshot
    METRICS_EXPORTED     = auto()   # Метрики записаны в файл


# ==============================================================================
#                              СОБЫТИЕ
# ==============================================================================

@dataclass
class SimulationEvent:
    """
    Контейнер события.

    Attributes:
        event_type (EventType): Тип события.
        data (Dict[str, Any]): Произвольные данные, специфичные для события.
        source (Optional[str]): Имя модуля-издателя (для отладки).
        tti (Optional[int]): Номер TTI, в котором произошло событие.
    """

    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    tti: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"SimulationEvent("
            f"type={self.event_type.name}, "
            f"tti={self.tti}, "
            f"source={self.source!r})"
        )


# ==============================================================================
#                              EVENT BUS
# ==============================================================================

class EventBus:
    """
    Централизованная шина событий симуляции (паттерн Observer).

    Позволяет модулям общаться без прямых ссылок друг на друга:
    - Издатель вызывает publish() — не знает о подписчиках.
    - Подписчик вызывает subscribe() — не знает об издателях.

    Шина намеренно синхронная: обработчики вызываются сразу в момент publish(),
    в порядке регистрации. Для асинхронного режима — вынести в очередь.

    Пример использования::

        bus = EventBus()

        def log_tti(event: SimulationEvent):
            print(f"[LOG] TTI {event.tti} done")

        bus.subscribe(EventType.TTI_COMPLETED, log_tti)
        bus.publish(SimulationEvent(EventType.TTI_COMPLETED, tti=1))
        # → [LOG] TTI 1 done
    """

    def __init__(self) -> None:
        # { EventType -> [callable, ...] }
        self._handlers: Dict[EventType, List[Callable[[SimulationEvent], None]]] = (
            defaultdict(list)
        )
        self._history: List[SimulationEvent] = []
        self._history_enabled: bool = False

    # ------------------------------------------------------------------
    #  Подписка / отписка
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[SimulationEvent], None],
    ) -> None:
        """
        Зарегистрировать обработчик на тип события.

        Args:
            event_type: Тип события для подписки.
            handler: Функция / метод, принимающий SimulationEvent.

        Example::

            bus.subscribe(EventType.UE_BUFFER_OVERFLOW, my_handler)
        """
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable[[SimulationEvent], None],
    ) -> None:
        """
        Отменить подписку обработчика.

        Args:
            event_type: Тип события.
            handler: Ранее зарегистрированный обработчик.
        """
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def subscribe_all(self, handler: Callable[[SimulationEvent], None]) -> None:
        """
        Подписаться на ВСЕ типы событий.
        Удобно для логгеров и отладочных трассировщиков.

        Args:
            handler: Обработчик для всех событий.
        """
        for event_type in EventType:
            self.subscribe(event_type, handler)

    # ------------------------------------------------------------------
    #  Публикация
    # ------------------------------------------------------------------

    def publish(self, event: SimulationEvent) -> None:
        """
        Опубликовать событие — вызвать всех подписчиков синхронно.

        Args:
            event: Экземпляр SimulationEvent.

        Note:
            Ошибки в обработчиках не подавляются намеренно —
            сломанный обработчик должен быть виден сразу.
        """
        if self._history_enabled:
            self._history.append(event)

        for handler in list(self._handlers.get(event.event_type, [])):
            handler(event)

    def publish_simple(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        tti: Optional[int] = None,
    ) -> None:
        """
        Shortcut: создать и опубликовать событие в одну строку.

        Args:
            event_type: Тип события.
            data: Словарь с данными события.
            source: Имя модуля-издателя.
            tti: Номер TTI.

        Example::

            bus.publish_simple(EventType.TTI_COMPLETED, {'rb_allocated': 50}, tti=100)
        """
        self.publish(SimulationEvent(
            event_type=event_type,
            data=data or {},
            source=source,
            tti=tti,
        ))

    # ------------------------------------------------------------------
    #  Утилиты
    # ------------------------------------------------------------------

    def enable_history(self, enabled: bool = True) -> None:
        """
        Включить/выключить хранение истории событий.
        По умолчанию выключено — для экономии памяти в длинных симуляциях.

        Args:
            enabled: True — хранить, False — не хранить.
        """
        self._history_enabled = enabled

    def get_history(
        self, event_type: Optional[EventType] = None
    ) -> List[SimulationEvent]:
        """
        Вернуть историю событий (если включена).

        Args:
            event_type: Фильтр по типу (None = все).

        Returns:
            Список событий в порядке публикации.
        """
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e.event_type == event_type]

    def clear_history(self) -> None:
        """Очистить историю событий."""
        self._history.clear()

    def clear_all_subscriptions(self) -> None:
        """Снять все подписки. Используется при сбросе симуляции."""
        self._handlers.clear()

    def subscriber_count(self, event_type: EventType) -> int:
        """
        Количество подписчиков на тип события.

        Args:
            event_type: Тип события.

        Returns:
            Количество зарегистрированных обработчиков.
        """
        return len(self._handlers.get(event_type, []))

    def __repr__(self) -> str:
        total = sum(len(h) for h in self._handlers.values())
        return f"EventBus(subscriptions={total}, history_enabled={self._history_enabled})"
