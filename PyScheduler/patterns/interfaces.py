"""
#------------------------------------------------------------------------------
# Паттерн: Strategy — абстрактные интерфейсы модулей
#------------------------------------------------------------------------------
# Описание:
#   Формальные ABC (Abstract Base Class) для всех ключевых модулей симуляции.
#   Паттерн Strategy — поведение инкапсулируется в сменных объектах,
#   реализующих общий интерфейс.
#
#   Зачем нужно:
#   - Гарантирует, что любой новый планировщик / модель канала / модель
#     мобильности реализует нужный контракт.
#   - Позволяет SimulationFacade и другим оркестраторам работать через
#     абстракцию, не зная о конкретных реализациях.
#   - Документирует ожидаемый API каждого модуля в одном месте.
#
#   Существующие классы (SchedulerInterface, ChannelInterface и др.) уже
#   частично реализуют эти контракты. Адаптеры (adapters.py) подшивают их
#   под данные интерфейсы без изменения оригинального кода.
#
# Версия: 1.0.0
#------------------------------------------------------------------------------
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
#                          ИНТЕРФЕЙС ПЛАНИРОВЩИКА
# ==============================================================================

class IScheduler(ABC):
    """
    Паттерн Strategy: контракт для всех алгоритмов планирования ресурсов.

    Конкретные алгоритмы (RoundRobin, BestCQI, ProportionalFair, FD-*)
    реализуют этот интерфейс. Оркестратор симуляции работает только
    через IScheduler — и не зависит от конкретного алгоритма.

    Пример подключения нового алгоритма::

        class MyScheduler(IScheduler):
            def schedule(self, tti, users): ...
            def get_stats(self): ...
            def reset(self): ...
    """

    @abstractmethod
    def schedule(self, tti: int, users: List[Dict]) -> Dict:
        """
        Распределить ресурсные блоки между UE для текущего TTI.

        Args:
            tti: Номер текущего Transmission Time Interval.
            users: Список словарей с данными UE
                   (ue_id, buffer_size, cqi и т.д.)

        Returns:
            Словарь с результатами планирования:
            - 'allocation': {ue_id: [rb_indices]}
            - 'grants':     [SchedulingGrant, ...]
            - 'stats':      метрики текущего TTI
        """
        ...

    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Вернуть статистику последнего вызова schedule().

        Returns:
            Словарь метрик: tti, rb_allocated, active_ue_count и др.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Сбросить внутреннее состояние планировщика.
        Вызывается при перезапуске симуляции.
        """
        ...

    @property
    def algorithm_name(self) -> str:
        """Название алгоритма для логов и экспорта статистики."""
        return self.__class__.__name__


# ==============================================================================
#                         ИНТЕРФЕЙС МОДЕЛИ МОБИЛЬНОСТИ
# ==============================================================================

class IMobilityModel(ABC):
    """
    Паттерн Strategy: контракт для моделей перемещения UE.

    Конкретные модели (RandomWalk, RandomWaypoint, GaussMarkov и др.)
    реализуют этот интерфейс. UE_MODULE получает модель через конструктор
    и вызывает update() на каждом шаге — не зная, какая именно модель.

    Архитектурный смысл «стратегии» здесь:
    - Алгоритм перемещения сменяем без изменения UE.
    - Параметры скорости, границ, направления — инкапсулированы в модели.
    """

    @abstractmethod
    def update(self, ue: Any) -> Tuple[float, float, float, float]:
        """
        Рассчитать новое положение UE.

        Args:
            ue: Объект UserEquipment с текущими координатами и скоростью.

        Returns:
            Кортеж (new_x, new_y, new_velocity, new_direction).
        """
        ...

    @abstractmethod
    def initialize(self, ue: Any) -> None:
        """
        Инициализировать начальное состояние модели для конкретного UE.
        Вызывается один раз при регистрации UE.

        Args:
            ue: Объект UserEquipment.
        """
        ...

    @property
    def model_name(self) -> str:
        """Название модели для логов."""
        return self.__class__.__name__


# ==============================================================================
#                         ИНТЕРФЕЙС МОДЕЛИ КАНАЛА
# ==============================================================================

class IChannelModel(ABC):
    """
    Паттерн Strategy: контракт для моделей радиоканала.

    Конкретные модели (UMa, UMi, RMa) реализуют этот интерфейс.
    Позволяет подключать новые сценарии развёртывания без изменения
    логики расчёта CQI и планировщика.
    """

    @abstractmethod
    def calculate_path_loss(self, ue: Any) -> float:
        """
        Рассчитать потери на трассе (path loss) для UE.

        Args:
            ue: Объект UserEquipment с координатами и indoor/outdoor флагом.

        Returns:
            Path loss в дБ.
        """
        ...

    @abstractmethod
    def calculate_sinr(self, ue: Any) -> float:
        """
        Рассчитать SINR для UE.

        Args:
            ue: Объект UserEquipment.

        Returns:
            SINR в дБ.
        """
        ...

    @abstractmethod
    def get_cqi(self, ue: Any) -> int:
        """
        Получить CQI (Channel Quality Indicator) для UE.

        Args:
            ue: Объект UserEquipment.

        Returns:
            Значение CQI в диапазоне [1, 15].
        """
        ...

    @abstractmethod
    def update(self, ue: Any, tti: int) -> None:
        """
        Обновить состояние канала для UE (вызывается на каждом TTI).

        Args:
            ue: Объект UserEquipment.
            tti: Номер текущего TTI.
        """
        ...

    @property
    def model_name(self) -> str:
        """Название модели для логов."""
        return self.__class__.__name__


# ==============================================================================
#                         ИНТЕРФЕЙС МОДЕЛИ ТРАФИКА
# ==============================================================================

class ITrafficModel(ABC):
    """
    Паттерн Strategy: контракт для моделей генерации трафика.

    Уже частично реализован в TRAFFIC_MODEL.py под тем же именем.
    Данная версия добавляет методы reset() и get_stats(),
    необходимые для интеграции с жизненным циклом модуля (lifecycle.py).

    Конкретные реализации: PoissonModel, OnOffModel, MMPPModel.
    """

    @abstractmethod
    def generate(self, tti: int) -> int:
        """
        Сгенерировать трафик для текущего TTI.

        Args:
            tti: Номер текущего TTI.

        Returns:
            Количество байт для добавления в буфер UE.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Сбросить внутреннее состояние генератора.
        Вызывается при перезапуске симуляции или смене сценария.
        """
        ...

    def get_stats(self) -> Dict:
        """
        Вернуть статистику генератора трафика.
        Переопределение опционально.

        Returns:
            Словарь с метриками (total_bytes, avg_rate и др.)
        """
        return {}

    @property
    def model_name(self) -> str:
        """Название модели для логов."""
        return self.__class__.__name__


# ==============================================================================
#                       ИНТЕРФЕЙС СБОРЩИКА МЕТРИК
# ==============================================================================

class IMetricsCollector(ABC):
    """
    Паттерн Strategy: контракт для сборщиков статистики.

    Позволяет подключать разные бэкенды (CSV, JSON, in-memory, Prometheus)
    без изменения SimulationManager. Текущий StatsManager реализует
    этот контракт через адаптер (adapters.py).
    """

    @abstractmethod
    def collect(self, tti: int) -> None:
        """
        Собрать метрики на текущем TTI.

        Args:
            tti: Номер текущего TTI.
        """
        ...

    @abstractmethod
    def export(self, filepath: str) -> None:
        """
        Экспортировать накопленные метрики в файл.

        Args:
            filepath: Путь к файлу экспорта.
        """
        ...

    @abstractmethod
    def get_snapshot(self, tti: int) -> Optional[Dict]:
        """
        Получить snapshot метрик для конкретного TTI из истории.

        Args:
            tti: Номер TTI.

        Returns:
            Словарь с метриками или None, если TTI не найден.
        """
        ...

    def reset(self) -> None:
        """Сбросить накопленные данные. Переопределение опционально."""
        ...

    @property
    def collector_name(self) -> str:
        """Название сборщика для логов."""
        return self.__class__.__name__
