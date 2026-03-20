"""
#------------------------------------------------------------------------------
# Паттерн: Adapter — интеграция существующего кода под новые интерфейсы
#------------------------------------------------------------------------------
# Описание:
#   Адаптеры оборачивают существующие классы (SchedulerInterface,
#   ChannelInterface, StatsManager) и приводят их к формальным ABCs из
#   interfaces.py — без изменения оригинального кода.
#
#   Зачем это нужно:
#   - Старый код продолжает работать как прежде.
#   - Новый код (SimulationFacade, lifecycle-модули) использует
#     унифицированный интерфейс, не зная о деталях реализации.
#   - Упрощает подключение внешних / legacy компонентов.
#
#   Структура адаптеров:
#
#     SchedulerAdapter      — SchedulerInterface  → IScheduler
#     ChannelModelAdapter   — ChannelInterface    → IChannelModel
#     StatsCollectorAdapter — StatsManager        → IMetricsCollector
#
# Версия: 1.0.0
#------------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from patterns.interfaces import IChannelModel, IMetricsCollector, IScheduler


# ==============================================================================
#                         АДАПТЕР ПЛАНИРОВЩИКА
# ==============================================================================

class SchedulerAdapter(IScheduler):
    """
    Паттерн Adapter: приводит существующий SchedulerInterface к IScheduler.

    Позволяет использовать любой планировщик (RoundRobin, BestCQI, FD_PF…),
    созданный через ``SchedulerInterface.create()``, в коде, который
    ожидает IScheduler — без изменений в SCHEDULER.py.

    Пример::

        from SCHEDULER import SchedulerInterface
        from patterns.adapters import SchedulerAdapter

        raw_scheduler = SchedulerInterface.create('ProportionalFair', grid, bs)
        scheduler: IScheduler = SchedulerAdapter(raw_scheduler)
        result = scheduler.schedule(tti=1, users=users_list)
    """

    def __init__(self, scheduler_instance: Any) -> None:
        """
        Args:
            scheduler_instance: Экземпляр любого класса, унаследованного
                от SchedulerInterface (RoundRobinScheduler, BestCQIScheduler и т.д.)
        """
        self._scheduler = scheduler_instance

    # ------------------------------------------------------------------
    #  IScheduler contract
    # ------------------------------------------------------------------

    def schedule(self, tti: int, users: List[Dict]) -> Dict:
        """
        Делегирует вызов оригинальному schedule().

        Args:
            tti: Номер TTI.
            users: Список словарей UE.

        Returns:
            Результат планирования от оригинального планировщика.
        """
        return self._scheduler.schedule(tti, users)

    def get_stats(self) -> Dict:
        """
        Возвращает статистику через оригинальный get_stats().

        Returns:
            Словарь метрик планировщика.
        """
        return self._scheduler.get_stats()

    def reset(self) -> None:
        """
        Сбрасывает внутреннее состояние планировщика.
        Вызывает reset() если метод существует, иначе предупреждает.
        """
        if hasattr(self._scheduler, 'reset'):
            self._scheduler.reset()
        else:
            warnings.warn(
                f"[SchedulerAdapter] {self._scheduler.__class__.__name__} "
                f"не реализует reset(). Состояние не сброшено.",
                UserWarning,
            )

    @property
    def algorithm_name(self) -> str:
        return self._scheduler.__class__.__name__

    # ------------------------------------------------------------------
    #  Проброс оригинальных атрибутов (для совместимости)
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """
        Прозрачный проброс атрибутов к оригинальному объекту.
        Позволяет обращаться к amc, pdcch_manager и другим компонентам
        планировщика через адаптер.
        """
        return getattr(self._scheduler, name)

    def __repr__(self) -> str:
        return f"SchedulerAdapter(wraps={self._scheduler.__class__.__name__})"


# ==============================================================================
#                        АДАПТЕР МОДЕЛИ КАНАЛА
# ==============================================================================

class ChannelModelAdapter(IChannelModel):
    """
    Паттерн Adapter: приводит существующий ChannelInterface к IChannelModel.

    Исторически ChannelInterface не имел формального ABC с методами
    calculate_sinr() / get_cqi(). Адаптер добавляет этот контракт поверх
    существующего кода.

    Пример::

        from CHANNEL_MODEL import ChannelInterface
        from patterns.adapters import ChannelModelAdapter

        raw_channel = ChannelInterface(bs).create('UMa', ...)
        channel: IChannelModel = ChannelModelAdapter(raw_channel)
        sinr = channel.calculate_sinr(ue)
    """

    def __init__(self, channel_instance: Any) -> None:
        """
        Args:
            channel_instance: Экземпляр конкретной модели канала
                (RMaModel, UMaModel, UMiModel и др.)
        """
        self._channel = channel_instance

    # ------------------------------------------------------------------
    #  IChannelModel contract
    # ------------------------------------------------------------------

    def calculate_path_loss(self, ue: Any) -> float:
        """
        Рассчитать path loss через оригинальный метод.

        Пробует вызвать get_path_loss(ue) → path_loss(ue) → path_loss_los(ue).
        Если ни один не найден — возвращает 0.0 с предупреждением.

        Args:
            ue: Объект UserEquipment.

        Returns:
            Path loss в дБ.
        """
        for method_name in ('get_path_loss', 'path_loss', 'path_loss_los'):
            method = getattr(self._channel, method_name, None)
            if callable(method):
                return method(ue)

        warnings.warn(
            f"[ChannelModelAdapter] {self._channel.__class__.__name__} "
            f"не имеет метода расчёта path loss. Возвращено 0.0.",
            UserWarning,
        )
        return 0.0

    def calculate_sinr(self, ue: Any) -> float:
        """
        Рассчитать SINR через оригинальный метод.

        Пробует get_sinr(ue) → calculate_sinr(ue).

        Args:
            ue: Объект UserEquipment.

        Returns:
            SINR в дБ.
        """
        for method_name in ('get_sinr', 'calculate_sinr'):
            method = getattr(self._channel, method_name, None)
            if callable(method):
                return method(ue)

        warnings.warn(
            f"[ChannelModelAdapter] {self._channel.__class__.__name__} "
            f"не имеет метода расчёта SINR. Возвращено 0.0.",
            UserWarning,
        )
        return 0.0

    def get_cqi(self, ue: Any) -> int:
        """
        Получить CQI через оригинальный метод.

        Пробует get_cqi(ue) → calculate_cqi(ue).

        Args:
            ue: Объект UserEquipment.

        Returns:
            CQI в диапазоне [1, 15].
        """
        for method_name in ('get_cqi', 'calculate_cqi'):
            method = getattr(self._channel, method_name, None)
            if callable(method):
                return method(ue)

        warnings.warn(
            f"[ChannelModelAdapter] {self._channel.__class__.__name__} "
            f"не имеет метода расчёта CQI. Возвращено 7 (середина диапазона).",
            UserWarning,
        )
        return 7

    def update(self, ue: Any, tti: int) -> None:
        """
        Обновить состояние канала на текущем TTI.

        Пробует update(ue, tti) → update(ue).

        Args:
            ue: Объект UserEquipment.
            tti: Номер TTI.
        """
        update_fn = getattr(self._channel, 'update', None)
        if callable(update_fn):
            try:
                update_fn(ue, tti)
            except TypeError:
                update_fn(ue)

    @property
    def model_name(self) -> str:
        return self._channel.__class__.__name__

    def __getattr__(self, name: str) -> Any:
        return getattr(self._channel, name)

    def __repr__(self) -> str:
        return f"ChannelModelAdapter(wraps={self._channel.__class__.__name__})"


# ==============================================================================
#                       АДАПТЕР СБОРЩИКА МЕТРИК
# ==============================================================================

class StatsCollectorAdapter(IMetricsCollector):
    """
    Паттерн Adapter: приводит существующий StatsManager к IMetricsCollector.

    StatsManager в SIMULATION_MANAGER.py имеет методы collect() и export_*(),
    но не реализует формальный интерфейс IMetricsCollector. Адаптер
    подшивает его под контракт — и позволяет SimulationFacade работать
    с любым сборщиком метрик единообразно.

    Пример::

        from SIMULATION_MANAGER import StatsManager, StatisticsConfig
        from patterns.adapters import StatsCollectorAdapter

        raw_stats = StatsManager(scheduler, StatisticsConfig())
        collector: IMetricsCollector = StatsCollectorAdapter(raw_stats)
        collector.collect(tti=100)
        collector.export('results/run_01.csv')
    """

    def __init__(self, stats_manager: Any) -> None:
        """
        Args:
            stats_manager: Экземпляр StatsManager из SIMULATION_MANAGER.py.
        """
        self._stats = stats_manager

    # ------------------------------------------------------------------
    #  IMetricsCollector contract
    # ------------------------------------------------------------------

    def collect(self, tti: int) -> None:
        """
        Делегирует collect(tti) к StatsManager.

        Args:
            tti: Номер TTI.
        """
        self._stats.collect(tti)

    def export(self, filepath: str) -> None:
        """
        Экспортирует метрики через доступный метод экспорта.

        Пробует export_csv(filepath) → export_json(filepath) → export(filepath).

        Args:
            filepath: Путь к файлу.
        """
        for method_name in ('export_csv', 'export_json', 'export'):
            method = getattr(self._stats, method_name, None)
            if callable(method):
                method(filepath)
                return

        warnings.warn(
            f"[StatsCollectorAdapter] StatsManager не имеет метода экспорта.",
            UserWarning,
        )

    def get_snapshot(self, tti: int) -> Optional[Dict]:
        """
        Найти snapshot метрик для TTI в истории.

        Args:
            tti: Номер TTI.

        Returns:
            Словарь snapshot или None.
        """
        history = getattr(self._stats, 'history', [])
        for snapshot in history:
            if isinstance(snapshot, dict) and snapshot.get('tti') == tti:
                return snapshot
        return None

    def reset(self) -> None:
        """
        Сбросить историю сборщика (если метод существует).
        """
        if hasattr(self._stats, 'history'):
            self._stats.history.clear()
        if hasattr(self._stats, 'detailed_history'):
            self._stats.detailed_history.clear()

    @property
    def collector_name(self) -> str:
        return self._stats.__class__.__name__

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stats, name)

    def __repr__(self) -> str:
        return f"StatsCollectorAdapter(wraps={self._stats.__class__.__name__})"
