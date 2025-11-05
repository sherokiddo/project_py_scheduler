"""
#------------------------------------------------------------------------------
# Модуль: EventManager - Система событий для межмодульного взаимодействия
#------------------------------------------------------------------------------
# Описание:
#   Реализует паттерн Observer для слабосвязанного взаимодействия между модулями
#   симулятора LTE-сети. Позволяет модулям подписываться на события и уведомлять
#   друг друга о изменениях состояния без прямых зависимостей.
#
# Версия: 1.0.0
# Дата создания: 2025-01-27
# Автор: Ляпин Никита
#------------------------------------------------------------------------------
"""
from typing import Dict, List, Callable, Any, Optional
from enum import Enum
import threading
import time
from dataclasses import dataclass
from collections import defaultdict

class EventType(Enum):
    """Типы событий в системе симуляции"""
    # События жизненного цикла симуляции
    SIMULATION_START = "simulation_start"
    SIMULATION_END = "simulation_end"
    TTI_START = "tti_start"
    TTI_END = "tti_end"

    # События пользовательского оборудования
    UE_REGISTERED = "ue_registered"
    UE_DEREGISTERED = "ue_deregistered"
    UE_POSITION_UPDATED = "ue_position_updated"
    UE_CHANNEL_UPDATED = "ue_channel_updated"
    UE_TRAFFIC_GENERATED = "ue_traffic_generated"

    # События базовой станции
    BS_BUFFER_UPDATED = "bs_buffer_updated"
    BS_SCHEDULING_COMPLETED = "bs_scheduling_completed"

    # События планировщика
    SCHEDULER_RESOURCE_ALLOCATED = "scheduler_resource_allocated"
    SCHEDULER_RESOURCE_RELEASED = "scheduler_resource_released"

    # События ошибок и предупреждений
    ERROR_OCCURRED = "error_occurred"
    WARNING_ISSUED = "warning_issued"

    # События метрик и статистики
    METRICS_UPDATED = "metrics_updated"
    STATISTICS_CALCULATED = "statistics_calculated"

@dataclass
class Event:
    """Класс для представления события в системе"""
    event_type: EventType
    timestamp: float
    source: str
    data: Dict[str, Any]
    priority: int = 0  # 0 = normal, 1 = high, 2 = critical

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

class EventManager:
    """
    Менеджер событий для координации взаимодействия между модулями.
    Реализует паттерн Observer с поддержкой приоритетов и асинхронной обработки.
    """

    def __init__(self, enable_async: bool = True, max_queue_size: int = 10000):
        """
        Инициализация менеджера событий.

        Args:
            enable_async: Включить асинхронную обработку событий
            max_queue_size: Максимальный размер очереди событий
        """
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_queue: List[Event] = []
        self._enable_async = enable_async
        self._max_queue_size = max_queue_size
        self._lock = threading.RLock()
        self._processing = False
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'subscribers_notified': 0,
            'errors_occurred': 0
        }

        if enable_async:
            self._start_async_processor()

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None],
                 priority: int = 0) -> bool:
        """
        Подписка на событие определенного типа.

        Args:
            event_type: Тип события для подписки
            callback: Функция-обработчик события
            priority: Приоритет обработчика (0 = normal, 1 = high, 2 = critical)

        Returns:
            bool: True если подписка успешна, False в противном случае
        """
        try:
            with self._lock:
                # Добавляем приоритет к callback для сортировки
                wrapped_callback = (priority, callback)
                self._subscribers[event_type].append(wrapped_callback)
                # Сортируем по приоритету (высший приоритет = меньший номер)
                self._subscribers[event_type].sort(key=lambda x: x[0])
                return True
        except Exception as e:
            self._stats['errors_occurred'] += 1
            print(f"Ошибка подписки на событие {event_type}: {e}")
            return False

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> bool:
        """
        Отписка от события.

        Args:
            event_type: Тип события
            callback: Функция-обработчик для удаления

        Returns:
            bool: True если отписка успешна, False в противном случае
        """
        try:
            with self._lock:
                if event_type in self._subscribers:
                    # Удаляем все вхождения callback (с любым приоритетом)
                    self._subscribers[event_type] = [
                        (p, cb) for p, cb in self._subscribers[event_type]
                        if cb != callback
                    ]
                    return True
                return False
        except Exception as e:
            self._stats['errors_occurred'] += 1
            print(f"Ошибка отписки от события {event_type}: {e}")
            return False

    def publish(self, event_type: EventType, source: str, data: Dict[str, Any] = None,
                priority: int = 0) -> bool:
        """
        Публикация события.

        Args:
            event_type: Тип события
            source: Источник события (имя модуля)
            data: Данные события
            priority: Приоритет события

        Returns:
            bool: True если событие успешно опубликовано
        """
        try:
            if data is None:
                data = {}

            event = Event(
                event_type=event_type,
                timestamp=time.time(),
                source=source,
                data=data,
                priority=priority
            )

            with self._lock:
                if len(self._event_queue) >= self._max_queue_size:
                    # Удаляем старые события с низким приоритетом
                    self._event_queue = [
                        e for e in self._event_queue
                        if e.priority >= priority
                    ]

                self._event_queue.append(event)
                self._stats['events_published'] += 1

            if not self._enable_async:
                self._process_event(event)

            return True

        except Exception as e:
            self._stats['errors_occurred'] += 1
            print(f"Ошибка публикации события {event_type}: {e}")
            return False

    def _process_event(self, event: Event) -> None:
        """
        Обработка одного события.

        Args:
            event: Событие для обработки
        """
        try:
            if event.event_type in self._subscribers:
                for priority, callback in self._subscribers[event.event_type]:
                    try:
                        callback(event)
                        self._stats['subscribers_notified'] += 1
                    except Exception as e:
                        self._stats['errors_occurred'] += 1
                        print(f"Ошибка в обработчике события {event.event_type}: {e}")

            self._stats['events_processed'] += 1

        except Exception as e:
            self._stats['errors_occurred'] += 1
            print(f"Ошибка обработки события {event.event_type}: {e}")

    def _start_async_processor(self) -> None:
        """Запуск асинхронного процессора событий"""
        def processor():
            self._processing = True
            while self._processing:
                try:
                    with self._lock:
                        if self._event_queue:
                            # Сортируем по приоритету и времени
                            self._event_queue.sort(key=lambda e: (-e.priority, e.timestamp))
                            event = self._event_queue.pop(0)
                        else:
                            event = None

                    if event:
                        self._process_event(event)
                    else:
                        time.sleep(0.001)  # Небольшая пауза если нет событий

                except Exception as e:
                    self._stats['errors_occurred'] += 1
                    print(f"Ошибка в асинхронном процессоре: {e}")
                    time.sleep(0.01)

        thread = threading.Thread(target=processor, daemon=True)
        thread.start()

    def stop_async_processing(self) -> None:
        """Остановка асинхронной обработки событий"""
        self._processing = False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики работы менеджера событий.

        Returns:
            Dict: Статистика работы системы событий
        """
        with self._lock:
            return {
                'events_published': self._stats['events_published'],
                'events_processed': self._stats['events_processed'],
                'events_pending': len(self._event_queue),
                'subscribers_notified': self._stats['subscribers_notified'],
                'errors_occurred': self._stats['errors_occurred'],
                'subscribers_count': {
                    event_type.value: len(callbacks)
                    for event_type, callbacks in self._subscribers.items()
                }
            }

    def clear_queue(self) -> None:
        """Очистка очереди событий"""
        with self._lock:
            self._event_queue.clear()

    def get_subscribers_count(self, event_type: EventType) -> int:
        """
        Получение количества подписчиков на событие.

        Args:
            event_type: Тип события

        Returns:
            int: Количество подписчиков
        """
        with self._lock:
            return len(self._subscribers.get(event_type, []))

# Глобальный экземпляр менеджера событий
event_manager = EventManager()

def get_event_manager() -> EventManager:
    """Получение глобального экземпляра менеджера событий"""
    return event_manager

# Удобные функции для быстрой работы с событиями
def publish_event(event_type: EventType, source: str, data: Dict[str, Any] = None,
                  priority: int = 0) -> bool:
    """Быстрая публикация события через глобальный менеджер"""
    return event_manager.publish(event_type, source, data, priority)

def subscribe_to_event(event_type: EventType, callback: Callable[[Event], None],
                       priority: int = 0) -> bool:
    """Быстрая подписка на событие через глобальный менеджер"""
    return event_manager.subscribe(event_type, callback, priority)

def unsubscribe_from_event(event_type: EventType, callback: Callable[[Event], None]) -> bool:
    """Быстрая отписка от события через глобальный менеджер"""
    return event_manager.unsubscribe(event_type, callback)



