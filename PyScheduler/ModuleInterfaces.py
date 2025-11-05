"""
#------------------------------------------------------------------------------
# Модуль: ModuleInterfaces - Абстрактные интерфейсы для модулей симулятора
#------------------------------------------------------------------------------
# Описание:
#   Определяет единые интерфейсы для всех модулей симулятора LTE-сети.
#   Обеспечивает слабую связанность между компонентами и упрощает тестирование.
#   Реализует принципы SOLID и паттерн Strategy.
#
# Версия: 1.0.0
# Дата создания: 2025-01-27
# Автор: Ляпин Никита
#------------------------------------------------------------------------------
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

class ModuleStatus(Enum):
    """Статусы модуля"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class ModuleInfo:
    """Информация о модуле"""
    name: str
    version: str
    status: ModuleStatus
    dependencies: List[str]
    capabilities: List[str]

class IModule(ABC):
    """
    Базовый интерфейс для всех модулей симулятора.
    Определяет общий контракт для инициализации, запуска и остановки модулей.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Инициализация модуля с заданной конфигурацией.

        Args:
            config: Словарь с параметрами конфигурации

        Returns:
            bool: True если инициализация успешна
        """
        pass

    @abstractmethod
    def start(self) -> bool:
        """
        Запуск модуля.

        Returns:
            bool: True если запуск успешен
        """
        pass

    @abstractmethod
    def stop(self) -> bool:
        """
        Остановка модуля.

        Returns:
            bool: True если остановка успешна
        """
        pass

    @abstractmethod
    def pause(self) -> bool:
        """
        Приостановка работы модуля.

        Returns:
            bool: True если приостановка успешна
        """
        pass

    @abstractmethod
    def resume(self) -> bool:
        """
        Возобновление работы модуля.

        Returns:
            bool: True если возобновление успешно
        """
        pass

    @abstractmethod
    def get_status(self) -> ModuleStatus:
        """
        Получение текущего статуса модуля.

        Returns:
            ModuleStatus: Текущий статус модуля
        """
        pass

    @abstractmethod
    def get_info(self) -> ModuleInfo:
        """
        Получение информации о модуле.

        Returns:
            ModuleInfo: Информация о модуле
        """
        pass

    @abstractmethod
    def update(self, time_delta: float) -> bool:
        """
        Обновление состояния модуля.

        Args:
            time_delta: Время, прошедшее с последнего обновления (мс)

        Returns:
            bool: True если обновление успешно
        """
        pass

class IChannelModel(IModule):
    """
    Интерфейс для моделей радиоканалов.
    Определяет методы расчета качества канала и потерь сигнала.
    """

    @abstractmethod
    def calculate_path_loss(self, distance_2d: float, distance_3d: float,
                           ue_height: float, ue_class: str) -> float:
        """
        Расчет затухания сигнала.

        Args:
            distance_2d: 2D расстояние до БС (м)
            distance_3d: 3D расстояние до БС (м)
            ue_height: Высота пользовательского оборудования (м)
            ue_class: Класс пользовательского оборудования

        Returns:
            float: Затухание сигнала в дБ
        """
        pass

    @abstractmethod
    def calculate_sinr(self, distance_2d: float, distance_3d: float,
                      ue_height: float, ue_class: str) -> float:
        """
        Расчет отношения сигнал-интерференция-шум.

        Args:
            distance_2d: 2D расстояние до БС (м)
            distance_3d: 3D расстояние до БС (м)
            ue_height: Высота пользовательского оборудования (м)
            ue_class: Класс пользовательского оборудования

        Returns:
            float: SINR в дБ
        """
        pass

    @abstractmethod
    def calculate_los_probability(self, distance_2d: float, ue_height: float) -> float:
        """
        Расчет вероятности прямой видимости.

        Args:
            distance_2d: 2D расстояние до БС (м)
            ue_height: Высота пользовательского оборудования (м)

        Returns:
            float: Вероятность LOS (0-1)
        """
        pass

class IMobilityModel(IModule):
    """
    Интерфейс для моделей мобильности пользователей.
    Определяет методы обновления позиции и состояния движения.
    """

    @abstractmethod
    def update_position(self, current_position: Tuple[float, float],
                       current_velocity: float, current_direction: float,
                       time_delta: float) -> Tuple[Tuple[float, float], float, float]:
        """
        Обновление позиции пользователя.

        Args:
            current_position: Текущая позиция (x, y)
            current_velocity: Текущая скорость (м/с)
            current_direction: Текущее направление (радианы)
            time_delta: Время с последнего обновления (мс)

        Returns:
            Tuple: (новая_позиция, новая_скорость, новое_направление)
        """
        pass

    @abstractmethod
    def get_velocity_limits(self) -> Tuple[float, float]:
        """
        Получение ограничений скорости для модели.

        Returns:
            Tuple: (минимальная_скорость, максимальная_скорость)
        """
        pass

    @abstractmethod
    def is_boundary_respected(self, position: Tuple[float, float]) -> bool:
        """
        Проверка соблюдения граничных условий.

        Args:
            position: Позиция для проверки (x, y)

        Returns:
            bool: True если позиция в допустимых границах
        """
        pass

class ITrafficModel(IModule):
    """
    Интерфейс для моделей генерации трафика.
    Определяет методы создания пакетов данных.
    """

    @abstractmethod
    def generate_traffic(self, ue_id: int, current_time: float,
                        time_interval: float) -> List[Dict[str, Any]]:
        """
        Генерация трафика для пользователя.

        Args:
            ue_id: Идентификатор пользователя
            current_time: Текущее время симуляции (мс)
            time_interval: Интервал генерации (мс)

        Returns:
            List[Dict]: Список сгенерированных пакетов
        """
        pass

    @abstractmethod
    def get_traffic_statistics(self, ue_id: int) -> Dict[str, Any]:
        """
        Получение статистики генерации трафика.

        Args:
            ue_id: Идентификатор пользователя

        Returns:
            Dict: Статистика трафика
        """
        pass

    @abstractmethod
    def reset_statistics(self, ue_id: int) -> None:
        """
        Сброс статистики для пользователя.

        Args:
            ue_id: Идентификатор пользователя
        """
        pass

class IScheduler(IModule):
    """
    Интерфейс для планировщиков ресурсов.
    Определяет методы распределения ресурсных блоков между пользователями.
    """

    @abstractmethod
    def schedule(self, tti: int, users: List[Dict[str, Any]],
                resources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Планирование ресурсов для заданного TTI.

        Args:
            tti: Индекс TTI для планирования
            users: Список пользователей с их параметрами
            resources: Доступные ресурсы для распределения

        Returns:
            Dict: Результаты планирования
        """
        pass

    @abstractmethod
    def get_scheduling_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик планировщика.

        Returns:
            Dict: Метрики производительности планировщика
        """
        pass

    @abstractmethod
    def reset_metrics(self) -> None:
        """Сброс метрик планировщика"""
        pass

class IResourceGrid(IModule):
    """
    Интерфейс для управления ресурсной сеткой LTE.
    Определяет методы выделения и освобождения ресурсных блоков.
    """

    @abstractmethod
    def allocate_resource_block(self, tti: int, frequency_idx: int,
                              ue_id: int) -> bool:
        """
        Выделение ресурсного блока пользователю.

        Args:
            tti: Индекс TTI
            frequency_idx: Частотный индекс
            ue_id: Идентификатор пользователя

        Returns:
            bool: True если выделение успешно
        """
        pass

    @abstractmethod
    def release_resource_block(self, tti: int, frequency_idx: int) -> bool:
        """
        Освобождение ресурсного блока.

        Args:
            tti: Индекс TTI
            frequency_idx: Частотный индекс

        Returns:
            bool: True если освобождение успешно
        """
        pass

    @abstractmethod
    def get_available_resources(self, tti: int) -> List[int]:
        """
        Получение списка доступных ресурсов для TTI.

        Args:
            tti: Индекс TTI

        Returns:
            List[int]: Список доступных частотных индексов
        """
        pass

    @abstractmethod
    def get_resource_utilization(self, tti: int) -> float:
        """
        Получение коэффициента использования ресурсов.

        Args:
            tti: Индекс TTI

        Returns:
            float: Коэффициент использования (0-1)
        """
        pass

class IUserEquipment(IModule):
    """
    Интерфейс для пользовательского оборудования.
    Определяет методы управления состоянием и взаимодействия с сетью.
    """

    @abstractmethod
    def update_position(self, time_delta: float, bs_position: Tuple[float, float],
                       bs_height: float) -> None:
        """
        Обновление позиции пользователя.

        Args:
            time_delta: Время с последнего обновления (мс)
            bs_position: Позиция базовой станции (x, y)
            bs_height: Высота базовой станции (м)
        """
        pass

    @abstractmethod
    def update_channel_quality(self) -> None:
        """Обновление качества канала"""
        pass

    @abstractmethod
    def generate_traffic(self, time_delta: float) -> None:
        """
        Генерация трафика.

        Args:
            time_delta: Время с последней генерации (мс)
        """
        pass

    @abstractmethod
    def get_channel_quality(self) -> Dict[str, Any]:
        """
        Получение текущего качества канала.

        Returns:
            Dict: Параметры качества канала (CQI, SINR, etc.)
        """
        pass

    @abstractmethod
    def get_buffer_status(self) -> Dict[str, Any]:
        """
        Получение статуса буфера.

        Returns:
            Dict: Статус буфера (размер, количество пакетов, etc.)
        """
        pass

class IBaseStation(IModule):
    """
    Интерфейс для базовой станции.
    Определяет методы управления пользователями и ресурсами.
    """

    @abstractmethod
    def register_user(self, ue: IUserEquipment) -> bool:
        """
        Регистрация пользователя.

        Args:
            ue: Объект пользовательского оборудования

        Returns:
            bool: True если регистрация успешна
        """
        pass

    @abstractmethod
    def deregister_user(self, ue_id: int) -> bool:
        """
        Отмена регистрации пользователя.

        Args:
            ue_id: Идентификатор пользователя

        Returns:
            bool: True если отмена регистрации успешна
        """
        pass

    @abstractmethod
    def get_registered_users(self) -> List[IUserEquipment]:
        """
        Получение списка зарегистрированных пользователей.

        Returns:
            List[IUserEquipment]: Список пользователей
        """
        pass

    @abstractmethod
    def update_user_buffers(self, time_delta: float) -> None:
        """
        Обновление буферов пользователей.

        Args:
            time_delta: Время с последнего обновления (мс)
        """
        pass

    @abstractmethod
    def get_buffer_status(self) -> Dict[str, Any]:
        """
        Получение статуса буферов базовой станции.

        Returns:
            Dict: Статус всех буферов
        """
        pass

class IEventPublisher(ABC):
    """
    Интерфейс для модулей, которые могут публиковать события.
    """

    @abstractmethod
    def publish_event(self, event_type: str, data: Dict[str, Any],
                     priority: int = 0) -> bool:
        """
        Публикация события.

        Args:
            event_type: Тип события
            data: Данные события
            priority: Приоритет события

        Returns:
            bool: True если публикация успешна
        """
        pass

class IEventSubscriber(ABC):
    """
    Интерфейс для модулей, которые могут подписываться на события.
    """

    @abstractmethod
    def handle_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Обработка события.

        Args:
            event_type: Тип события
            data: Данные события
        """
        pass

    @abstractmethod
    def get_subscribed_events(self) -> List[str]:
        """
        Получение списка подписанных событий.

        Returns:
            List[str]: Список типов событий
        """
        pass

class IConfigurable(ABC):
    """
    Интерфейс для модулей, которые могут быть сконфигурированы.
    """

    @abstractmethod
    def get_configuration(self) -> Dict[str, Any]:
        """
        Получение текущей конфигурации модуля.

        Returns:
            Dict: Текущая конфигурация
        """
        pass

    @abstractmethod
    def set_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Установка конфигурации модуля.

        Args:
            config: Новая конфигурация

        Returns:
            bool: True если конфигурация установлена успешно
        """
        pass

    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Валидация конфигурации.

        Args:
            config: Конфигурация для проверки

        Returns:
            Tuple: (валидна_ли, список_ошибок)
        """
        pass

