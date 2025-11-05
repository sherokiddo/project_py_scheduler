"""
#------------------------------------------------------------------------------
# Модуль: BaseModule - Базовый класс для всех модулей симулятора
#------------------------------------------------------------------------------
# Описание:
#   Предоставляет базовую реализацию интерфейса IModule с поддержкой событий,
#   конфигурации и логирования. Упрощает создание новых модулей и обеспечивает
#   единообразие в архитектуре системы.
#
# Версия: 1.0.0
# Дата создания: 2025-10-11
# Автор: Ляпин Никита
#------------------------------------------------------------------------------
"""
from typing import Dict, List, Any, Optional, Callable
from abc import abstractmethod
import time
import logging
from dataclasses import dataclass

from ModuleInterfaces import (
    IModule, ModuleStatus, ModuleInfo, IEventPublisher, IEventSubscriber, IConfigurable
)
from EventManager import EventManager, EventType, Event, get_event_manager

@dataclass
class ModuleConfig:
    """Базовая конфигурация модуля"""
    name: str
    version: str = "1.0.0"
    enabled: bool = True
    log_level: str = "INFO"
    custom_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}

class BaseModule(IModule, IEventPublisher, IEventSubscriber, IConfigurable):
    """
    Базовый класс для всех модулей симулятора.
    Предоставляет общую функциональность для работы с событиями, конфигурацией и логированием.
    """

    def __init__(self, config: ModuleConfig):
        """
        Инициализация базового модуля.

        Args:
            config: Конфигурация модуля
        """
        self.config = config
        self.status = ModuleStatus.INITIALIZED
        self.event_manager = get_event_manager()
        self.logger = self._setup_logger()

        # Статистика модуля
        self.stats = {
            'initialization_time': 0.0,
            'start_time': 0.0,
            'stop_time': 0.0,
            'update_count': 0,
            'error_count': 0,
            'events_published': 0,
            'events_received': 0
        }

        # Подписанные события
        self._subscribed_events: List[str] = []

        # Обработчики событий
        self._event_handlers: Dict[str, Callable] = {}

        # Инициализация времени
        self._last_update_time = 0.0

    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера для модуля"""
        logger = logging.getLogger(f"Module.{self.config.name}")
        logger.setLevel(getattr(logging, self.config.log_level))
        return logger

    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Инициализация модуля с заданной конфигурацией.

        Args:
            config: Словарь с параметрами конфигурации

        Returns:
            bool: True если инициализация успешна
        """
        try:
            start_time = time.time()

            # Обновляем конфигурацию
            self.config.custom_params.update(config)

            # Выполняем специфичную инициализацию
            if not self._initialize_impl():
                self.logger.error(f"Ошибка инициализации модуля {self.config.name}")
                return False

            # Подписываемся на события
            self._setup_event_subscriptions()

            self.stats['initialization_time'] = time.time() - start_time
            self.status = ModuleStatus.INITIALIZED

            self.logger.info(f"Модуль {self.config.name} инициализирован")
            self.publish_event("module_initialized", {
                "module_name": self.config.name,
                "initialization_time": self.stats['initialization_time']
            })

            return True

        except Exception as e:
            self.logger.error(f"Ошибка инициализации модуля {self.config.name}: {e}")
            self.status = ModuleStatus.ERROR
            self.stats['error_count'] += 1
            return False

    def start(self) -> bool:
        """
        Запуск модуля.

        Returns:
            bool: True если запуск успешен
        """
        try:
            if self.status != ModuleStatus.INITIALIZED:
                self.logger.warning(f"Модуль {self.config.name} не инициализирован")
                return False

            if not self.config.enabled:
                self.logger.info(f"Модуль {self.config.name} отключен")
                return True

            # Выполняем специфичный запуск
            if not self._start_impl():
                self.logger.error(f"Ошибка запуска модуля {self.config.name}")
                return False

            self.stats['start_time'] = time.time()
            self.status = ModuleStatus.RUNNING

            self.logger.info(f"Модуль {self.config.name} запущен")
            self.publish_event("module_started", {
                "module_name": self.config.name,
                "start_time": self.stats['start_time']
            })

            return True

        except Exception as e:
            self.logger.error(f"Ошибка запуска модуля {self.config.name}: {e}")
            self.status = ModuleStatus.ERROR
            self.stats['error_count'] += 1
            return False

    def stop(self) -> bool:
        """
        Остановка модуля.

        Returns:
            bool: True если остановка успешна
        """
        try:
            if self.status not in [ModuleStatus.RUNNING, ModuleStatus.PAUSED]:
                self.logger.warning(f"Модуль {self.config.name} не запущен")
                return True

            # Выполняем специфичную остановку
            if not self._stop_impl():
                self.logger.error(f"Ошибка остановки модуля {self.config.name}")
                return False

            # Отписываемся от событий
            self._cleanup_event_subscriptions()

            self.stats['stop_time'] = time.time()
            self.status = ModuleStatus.STOPPED

            self.logger.info(f"Модуль {self.config.name} остановлен")
            self.publish_event("module_stopped", {
                "module_name": self.config.name,
                "stop_time": self.stats['stop_time']
            })

            return True

        except Exception as e:
            self.logger.error(f"Ошибка остановки модуля {self.config.name}: {e}")
            self.status = ModuleStatus.ERROR
            self.stats['error_count'] += 1
            return False

    def pause(self) -> bool:
        """
        Приостановка работы модуля.

        Returns:
            bool: True если приостановка успешна
        """
        try:
            if self.status != ModuleStatus.RUNNING:
                self.logger.warning(f"Модуль {self.config.name} не запущен")
                return False

            # Выполняем специфичную приостановку
            if not self._pause_impl():
                self.logger.error(f"Ошибка приостановки модуля {self.config.name}")
                return False

            self.status = ModuleStatus.PAUSED

            self.logger.info(f"Модуль {self.config.name} приостановлен")
            self.publish_event("module_paused", {
                "module_name": self.config.name,
                "pause_time": time.time()
            })

            return True

        except Exception as e:
            self.logger.error(f"Ошибка приостановки модуля {self.config.name}: {e}")
            self.status = ModuleStatus.ERROR
            self.stats['error_count'] += 1
            return False

    def resume(self) -> bool:
        """
        Возобновление работы модуля.

        Returns:
            bool: True если возобновление успешно
        """
        try:
            if self.status != ModuleStatus.PAUSED:
                self.logger.warning(f"Модуль {self.config.name} не приостановлен")
                return False

            # Выполняем специфичное возобновление
            if not self._resume_impl():
                self.logger.error(f"Ошибка возобновления модуля {self.config.name}")
                return False

            self.status = ModuleStatus.RUNNING

            self.logger.info(f"Модуль {self.config.name} возобновлен")
            self.publish_event("module_resumed", {
                "module_name": self.config.name,
                "resume_time": time.time()
            })

            return True

        except Exception as e:
            self.logger.error(f"Ошибка возобновления модуля {self.config.name}: {e}")
            self.status = ModuleStatus.ERROR
            self.stats['error_count'] += 1
            return False

    def get_status(self) -> ModuleStatus:
        """
        Получение текущего статуса модуля.

        Returns:
            ModuleStatus: Текущий статус модуля
        """
        return self.status

    def get_info(self) -> ModuleInfo:
        """
        Получение информации о модуле.

        Returns:
            ModuleInfo: Информация о модуле
        """
        return ModuleInfo(
            name=self.config.name,
            version=self.config.version,
            status=self.status,
            dependencies=self._get_dependencies(),
            capabilities=self._get_capabilities()
        )

    def update(self, time_delta: float) -> bool:
        """
        Обновление состояния модуля.

        Args:
            time_delta: Время, прошедшее с последнего обновления (мс)

        Returns:
            bool: True если обновление успешно
        """
        try:
            if self.status != ModuleStatus.RUNNING:
                return True

            if not self.config.enabled:
                return True

            # Выполняем специфичное обновление
            if not self._update_impl(time_delta):
                self.logger.error(f"Ошибка обновления модуля {self.config.name}")
                self.stats['error_count'] += 1
                return False

            self.stats['update_count'] += 1
            self._last_update_time = time.time()

            return True

        except Exception as e:
            self.logger.error(f"Ошибка обновления модуля {self.config.name}: {e}")
            self.status = ModuleStatus.ERROR
            self.stats['error_count'] += 1
            return False

    # Реализация IEventPublisher
    def publish_event(self, event_type: str, data: Dict[str, Any], priority: int = 0) -> bool:
        """
        Публикация события.

        Args:
            event_type: Тип события
            data: Данные события
            priority: Приоритет события

        Returns:
            bool: True если публикация успешна
        """
        try:
            # Добавляем информацию о модуле-источнике
            data['source_module'] = self.config.name
            data['timestamp'] = time.time()

            success = self.event_manager.publish(
                EventType(event_type) if hasattr(EventType, event_type) else EventType.METRICS_UPDATED,
                self.config.name,
                data,
                priority
            )

            if success:
                self.stats['events_published'] += 1

            return success

        except Exception as e:
            self.logger.error(f"Ошибка публикации события {event_type}: {e}")
            return False

    # Реализация IEventSubscriber
    def handle_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Обработка события.

        Args:
            event_type: Тип события
            data: Данные события
        """
        try:
            if event_type in self._event_handlers:
                self._event_handlers[event_type](data)
                self.stats['events_received'] += 1

        except Exception as e:
            self.logger.error(f"Ошибка обработки события {event_type}: {e}")
            self.stats['error_count'] += 1

    def get_subscribed_events(self) -> List[str]:
        """
        Получение списка подписанных событий.

        Returns:
            List[str]: Список типов событий
        """
        return self._subscribed_events.copy()

    # Реализация IConfigurable
    def get_configuration(self) -> Dict[str, Any]:
        """
        Получение текущей конфигурации модуля.

        Returns:
            Dict: Текущая конфигурация
        """
        return {
            "name": self.config.name,
            "version": self.config.version,
            "enabled": self.config.enabled,
            "log_level": self.config.log_level,
            "custom_params": self.config.custom_params.copy()
        }

    def set_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Установка конфигурации модуля.

        Args:
            config: Новая конфигурация

        Returns:
            bool: True если конфигурация установлена успешно
        """
        try:
            # Валидируем конфигурацию
            is_valid, errors = self.validate_configuration(config)
            if not is_valid:
                self.logger.error(f"Неверная конфигурация: {errors}")
                return False

            # Обновляем конфигурацию
            if "enabled" in config:
                self.config.enabled = config["enabled"]
            if "log_level" in config:
                self.config.log_level = config["log_level"]
                self.logger.setLevel(getattr(logging, config["log_level"]))
            if "custom_params" in config:
                self.config.custom_params.update(config["custom_params"])

            # Выполняем специфичное обновление конфигурации
            self._update_configuration_impl(config)

            self.logger.info(f"Конфигурация модуля {self.config.name} обновлена")
            return True

        except Exception as e:
            self.logger.error(f"Ошибка установки конфигурации: {e}")
            return False

    def validate_configuration(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Валидация конфигурации.

        Args:
            config: Конфигурация для проверки

        Returns:
            Tuple: (валидна_ли, список_ошибок)
        """
        errors = []

        # Базовая валидация
        if "enabled" in config and not isinstance(config["enabled"], bool):
            errors.append("enabled должен быть boolean")

        if "log_level" in config and config["log_level"] not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            errors.append("log_level должен быть одним из: DEBUG, INFO, WARNING, ERROR")

        # Специфичная валидация
        specific_errors = self._validate_configuration_impl(config)
        errors.extend(specific_errors)

        return len(errors) == 0, errors

    def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики модуля.

        Returns:
            Dict: Статистика модуля
        """
        return {
            "module_name": self.config.name,
            "status": self.status.value,
            "stats": self.stats.copy(),
            "subscribed_events": self._subscribed_events.copy(),
            "last_update_time": self._last_update_time
        }

    # Абстрактные методы для переопределения в дочерних классах
    @abstractmethod
    def _initialize_impl(self) -> bool:
        """Специфичная инициализация модуля"""
        pass

    @abstractmethod
    def _start_impl(self) -> bool:
        """Специфичный запуск модуля"""
        pass

    @abstractmethod
    def _stop_impl(self) -> bool:
        """Специфичная остановка модуля"""
        pass

    def _pause_impl(self) -> bool:
        """Специфичная приостановка модуля (по умолчанию не требуется)"""
        return True

    def _resume_impl(self) -> bool:
        """Специфичное возобновление модуля (по умолчанию не требуется)"""
        return True

    @abstractmethod
    def _update_impl(self, time_delta: float) -> bool:
        """Специфичное обновление модуля"""
        pass

    def _get_dependencies(self) -> List[str]:
        """Получение списка зависимостей модуля"""
        return []

    def _get_capabilities(self) -> List[str]:
        """Получение списка возможностей модуля"""
        return []

    def _setup_event_subscriptions(self) -> None:
        """Настройка подписок на события"""
        pass

    def _cleanup_event_subscriptions(self) -> None:
        """Очистка подписок на события"""
        for event_type in self._subscribed_events:
            self.event_manager.unsubscribe(EventType(event_type), self.handle_event)
        self._subscribed_events.clear()

    def _update_configuration_impl(self, config: Dict[str, Any]) -> None:
        """Специфичное обновление конфигурации модуля"""
        pass

    def _validate_configuration_impl(self, config: Dict[str, Any]) -> List[str]:
        """Специфичная валидация конфигурации модуля"""
        return []

    def _subscribe_to_event(self, event_type: str, handler: Callable = None) -> None:
        """
        Подписка на событие.

        Args:
            event_type: Тип события
            handler: Обработчик события (по умолчанию используется handle_event)
        """
        if handler is None:
            handler = self.handle_event

        if hasattr(EventType, event_type):
            self.event_manager.subscribe(EventType(event_type), handler)
            self._subscribed_events.append(event_type)
            self._event_handlers[event_type] = handler
        else:
            self.logger.warning(f"Неизвестный тип события: {event_type}")



