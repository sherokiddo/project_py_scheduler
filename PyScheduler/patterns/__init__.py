"""
#------------------------------------------------------------------------------
# Пакет: patterns — Паттерны проектирования для LTE Scheduler
#------------------------------------------------------------------------------
# Описание:
#   Набор паттернов GoF, реализованных поверх существующей кодовой базы
#   PyScheduler. Паттерны не ломают текущий API — только добавляют новые
#   возможности и формальные интерфейсы.
#
#   Observer  — events.py     — событийная шина симуляции
#   Strategy  — interfaces.py — абстрактные интерфейсы модулей
#   Adapter   — adapters.py   — оборачивает существующие классы под интерфейсы
#   Facade    — facade.py     — упрощённое управление симуляцией
#   Template  — lifecycle.py  — жизненный цикл модуля (initialize → step → finalize)
#
# Версия: 1.0.0
# Автор: feature/design-patterns
#------------------------------------------------------------------------------
"""

from patterns.observer import EventBus, SimulationEvent, EventType
from patterns.interfaces import IScheduler, IMobilityModel, IChannelModel, ITrafficModel, IMetricsCollector
from patterns.adapters import SchedulerAdapter, ChannelModelAdapter, StatsCollectorAdapter
from patterns.facade import SimulationFacade
from patterns.lifecycle import SimulationModule

__all__ = [
    # Observer
    "EventBus",
    "SimulationEvent",
    "EventType",
    # Strategy interfaces
    "IScheduler",
    "IMobilityModel",
    "IChannelModel",
    "ITrafficModel",
    "IMetricsCollector",
    # Adapters
    "SchedulerAdapter",
    "ChannelModelAdapter",
    "StatsCollectorAdapter",
    # Facade
    "SimulationFacade",
    # Template Method
    "SimulationModule",
]
