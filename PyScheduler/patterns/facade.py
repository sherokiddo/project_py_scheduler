"""
#------------------------------------------------------------------------------
# Паттерн: Facade — упрощённое управление симуляцией
#------------------------------------------------------------------------------
# Описание:
#   SimulationFacade предоставляет лаконичный высокоуровневый API поверх
#   сложной подсистемы (SimulationManager, SchedulerInterface, StatsManager,
#   EventBus и т.д.).
#
#   Без Facade для запуска симуляции нужно:
#     - создать BaseStation, UECollection вручную
#     - настроить 3 конфига (SimulationConfig, SchedulerConfig, StatsManagerConfig)
#     - вручную вызвать set_*() методы в правильном порядке
#     - передать всё в SimulationManager.run()
#
#   С Facade:
#       sim = SimulationFacade()
#       sim.configure(num_ue=10, bandwidth=10, scheduler='ProportionalFair')
#       sim.run(duration_ms=1000)
#       results = sim.get_results()
#
#   Facade не ограничивает возможности — при необходимости даёт доступ
#   к underlying объектам через sim.manager / sim.scheduler / sim.event_bus.
#
# Версия: 1.0.0
#------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from patterns.observer import EventBus, EventType, SimulationEvent


# ==============================================================================
#                           КОНФИГУРАЦИЯ ФАСАДА
# ==============================================================================

@dataclass
class FacadeConfig:
    """
    Единый конфиг для SimulationFacade.

    Скрывает деление на SimulationConfig / SchedulerConfig / StatsManagerConfig.
    Поля намеренно плоские — чтобы конфигурация читалась как одна строка.

    Attributes:
        duration_ms (int): Длительность симуляции в TTI (мс).
        bandwidth (float): Полоса пропускания БС в МГц (1.4/3/5/10/15/20).
        scheduler (str): Алгоритм планировщика ('RoundRobin', 'BestCQI',
            'ProportionalFair', 'FD_PF', 'FD_BCQI', 'FD_FGS').
        num_ue (int): Количество UE (если создаём автоматически).
        map_size (float): Размер карты в метрах (квадрат от 0 до map_size).
        collect_stats (bool): Включить сбор статистики.
        stats_interval (int): Интервал сбора метрик в TTI.
        verbose (bool): Подробный вывод в консоль.
        seed (Optional[int]): Сид генератора случайных чисел.
    """

    duration_ms: int = 1000
    bandwidth: float = 10.0
    scheduler: str = 'RoundRobin'
    num_ue: int = 10
    map_size: float = 500.0
    collect_stats: bool = True
    stats_interval: int = 10
    verbose: bool = False
    seed: Optional[int] = None


# ==============================================================================
#                             РЕЗУЛЬТАТ СИМУЛЯЦИИ
# ==============================================================================

@dataclass
class SimulationResult:
    """
    Контейнер с результатами завершённой симуляции.

    Attributes:
        total_tti (int): Суммарное количество выполненных TTI.
        history (List[Dict]): История snapshot-ов StatsManager.
        summary (Dict): Сводные метрики (средний throughput, fairness и т.д.)
        scheduler_name (str): Название использованного алгоритма.
        events (List[SimulationEvent]): История событий EventBus (если включена).
    """

    total_tti: int = 0
    history: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    scheduler_name: str = ''
    events: List[SimulationEvent] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SimulationResult("
            f"tti={self.total_tti}, "
            f"snapshots={len(self.history)}, "
            f"scheduler={self.scheduler_name!r})"
        )


# ==============================================================================
#                            SIMULATION FACADE
# ==============================================================================

class SimulationFacade:
    """
    Паттерн Facade: единая точка входа для управления симуляцией LTE.

    Оркеструет подсистемы симулятора (SimulationManager, EventBus,
    адаптеры паттернов) и скрывает их сложность за простым API.

    Быстрый старт::

        from patterns.facade import SimulationFacade

        sim = SimulationFacade()
        sim.configure(
            duration_ms=2000,
            bandwidth=10,
            scheduler='ProportionalFair',
            num_ue=20,
            verbose=False,
        )
        result = sim.run()
        print(result.summary)

    Продвинутое использование (подписка на события)::

        sim = SimulationFacade()
        sim.configure(...)

        @sim.on(EventType.TTI_COMPLETED)
        def on_tti(event):
            if event.data.get('rb_allocated', 0) < 10:
                print(f"[WARN] TTI {event.tti}: мало RB выделено!")

        result = sim.run()

    Доступ к underlying объектам::

        sim.configure(...)
        sim.build()                      # только создать объекты, не запускать
        sim.manager.set_upd_interval(5)  # дотянуться до SimulationManager
        result = sim.run()
    """

    def __init__(self) -> None:
        self._config: Optional[FacadeConfig] = None
        self._event_bus: EventBus = EventBus()
        self._result: Optional[SimulationResult] = None

        # Underlying объекты — доступны после build()
        self._manager: Any = None       # SimulationManager
        self._scheduler: Any = None     # SchedulerInterface instance

        self._built: bool = False
        self._ran: bool = False

    # ------------------------------------------------------------------
    #  Публичный API — конфигурация
    # ------------------------------------------------------------------

    def configure(self, **kwargs) -> "SimulationFacade":
        """
        Задать параметры симуляции.

        Принимает те же поля, что и FacadeConfig (через **kwargs).
        Возвращает self для цепочки вызовов.

        Args:
            **kwargs: Поля FacadeConfig (duration_ms, bandwidth, scheduler,
                num_ue, map_size, collect_stats, stats_interval, verbose, seed).

        Returns:
            self — для fluent interface: ``sim.configure(...).run()``.

        Raises:
            ValueError: Если передан неизвестный параметр.

        Example::

            sim.configure(duration_ms=5000, scheduler='BestCQI', num_ue=30)
        """
        known = {f.name for f in FacadeConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(kwargs) - known
        if unknown:
            raise ValueError(
                f"[SimulationFacade] Неизвестные параметры: {unknown}. "
                f"Допустимые: {known}"
            )

        self._config = FacadeConfig(**{
            **FacadeConfig().__dict__,
            **kwargs,
        })
        self._built = False
        self._ran = False
        return self

    def on(self, event_type: EventType):
        """
        Декоратор-подписчик на события EventBus.

        Использование::

            @sim.on(EventType.UE_BUFFER_OVERFLOW)
            def handler(event):
                print(f"Переполнение буфера UE {event.data['ue_id']}")

        Args:
            event_type: Тип события из EventType.

        Returns:
            Декоратор, регистрирующий функцию как обработчик.
        """
        def decorator(func):
            self._event_bus.subscribe(event_type, func)
            return func
        return decorator

    # ------------------------------------------------------------------
    #  Публичный API — запуск
    # ------------------------------------------------------------------

    def build(self) -> "SimulationFacade":
        """
        Создать все объекты симуляции (без запуска цикла).

        Полезно когда нужно настроить underlying объекты вручную
        перед стартом (см. ``sim.manager``).

        Returns:
            self — для fluent interface.

        Raises:
            RuntimeError: Если configure() не был вызван.
        """
        if self._config is None:
            raise RuntimeError(
                "[SimulationFacade] Вызовите configure() перед build()."
            )

        self._manager = self._create_manager()
        self._built = True
        return self

    def run(self) -> SimulationResult:
        """
        Запустить симуляцию и вернуть результат.

        Если build() не был вызван — вызывает его автоматически.
        Если configure() не был вызван — использует параметры по умолчанию.

        Returns:
            SimulationResult с историей, сводкой и событиями.

        Example::

            result = SimulationFacade().configure(duration_ms=500).run()
        """
        if self._config is None:
            self._config = FacadeConfig()

        if not self._built:
            self.build()

        self._event_bus.enable_history(True)
        self._event_bus.publish_simple(
            EventType.SIMULATION_STARTED,
            data={'scheduler': self._config.scheduler,
                  'duration_ms': self._config.duration_ms},
            source='SimulationFacade',
        )

        try:
            self._manager.run()
            self._ran = True

            result = self._collect_result()

            self._event_bus.publish_simple(
                EventType.SIMULATION_STOPPED,
                data={'total_tti': result.total_tti},
                source='SimulationFacade',
            )

        except Exception as exc:
            self._event_bus.publish_simple(
                EventType.SIMULATION_ERROR,
                data={'error': str(exc)},
                source='SimulationFacade',
            )
            raise

        result.events = self._event_bus.get_history()
        self._result = result
        return result

    # ------------------------------------------------------------------
    #  Публичный API — доступ к результатам
    # ------------------------------------------------------------------

    def get_results(self) -> Optional[SimulationResult]:
        """
        Вернуть результаты последнего run() или None.

        Returns:
            SimulationResult или None если run() ещё не вызывался.
        """
        return self._result

    def get_summary(self) -> Dict:
        """
        Вернуть сводные метрики последней симуляции.

        Returns:
            Словарь summary или пустой словарь.
        """
        if self._result is None:
            return {}
        return self._result.summary

    # ------------------------------------------------------------------
    #  Доступ к underlying объектам
    # ------------------------------------------------------------------

    @property
    def manager(self) -> Any:
        """
        Прямой доступ к SimulationManager.
        Доступен после build() или run().
        """
        if not self._built:
            raise RuntimeError(
                "[SimulationFacade] Вызовите build() или run() перед обращением к manager."
            )
        return self._manager

    @property
    def scheduler(self) -> Any:
        """
        Прямой доступ к планировщику.
        Доступен после build() или run().
        """
        if not self._built:
            raise RuntimeError(
                "[SimulationFacade] Вызовите build() или run() перед обращением к scheduler."
            )
        return self._manager.scheduler

    @property
    def event_bus(self) -> EventBus:
        """Доступ к EventBus для расширенной подписки."""
        return self._event_bus

    # ------------------------------------------------------------------
    #  Внутренние методы
    # ------------------------------------------------------------------

    def _create_manager(self) -> Any:
        """
        Создать и настроить SimulationManager из FacadeConfig.

        Returns:
            Готовый к запуску SimulationManager.
        """
        # Импорт здесь — чтобы не создавать циклических зависимостей
        # на уровне модуля patterns/
        try:
            from SIMULATION_MANAGER import SimulationManager
            from BS_MODULE import BaseStation
            from UE_MODULE import UECollection
            import GLOBALS
        except ImportError as exc:
            raise ImportError(
                f"[SimulationFacade] Не удалось импортировать модули PyScheduler: {exc}. "
                f"Убедитесь, что facade запускается из директории PyScheduler."
            ) from exc

        cfg = self._config

        # Установить seed
        if cfg.seed is not None:
            GLOBALS.SEED = cfg.seed

        manager = SimulationManager()
        manager.set_sim_duration(cfg.duration_ms)
        manager.set_map_borders(0, cfg.map_size, 0, cfg.map_size)

        # BaseStation — минимальная конфигурация
        bs = BaseStation(bandwidth=cfg.bandwidth)
        manager.set_base_station(bs)

        # UECollection
        ue_collection = UECollection()
        manager.set_ue_collection(ue_collection)

        # Планировщик
        manager.set_scheduler(cfg.scheduler)

        # Статистика
        if cfg.collect_stats:
            manager.set_stats_enabled(True)
            manager.set_stats_collect_interval(cfg.stats_interval)

        manager.set_verbose(cfg.verbose)

        # Подписать EventBus на события менеджера (если менеджер поддерживает)
        self._bind_event_bus(manager)

        return manager

    def _bind_event_bus(self, manager: Any) -> None:
        """
        Связать EventBus с менеджером симуляции, если он поддерживает hooks.
        Не выбрасывает ошибку если менеджер не имеет нужных атрибутов.

        Args:
            manager: SimulationManager.
        """
        if hasattr(manager, 'event_bus'):
            manager.event_bus = self._event_bus
        elif hasattr(manager, 'on_tti_completed'):
            # Альтернативный hook если EventBus не встроен
            original = manager.on_tti_completed

            def tti_hook(tti: int, result: Dict) -> None:
                original(tti, result)
                self._event_bus.publish_simple(
                    EventType.TTI_COMPLETED,
                    data=result,
                    source='SimulationManager',
                    tti=tti,
                )

            manager.on_tti_completed = tti_hook

    def _collect_result(self) -> SimulationResult:
        """
        Собрать результаты из SimulationManager после завершения run().

        Returns:
            Заполненный SimulationResult.
        """
        result = SimulationResult()

        if self._manager is None:
            return result

        result.scheduler_name = self._config.scheduler if self._config else ''

        # История из StatsManager
        stats_manager = getattr(self._manager, 'stats_manager', None)
        if stats_manager is not None:
            history = list(getattr(stats_manager, 'history', []))
            result.history = history
            result.total_tti = history[-1].get('tti', 0) if history else 0
            result.summary = self._compute_summary(history)
        else:
            result.total_tti = getattr(self._config, 'duration_ms', 0)

        return result

    def _compute_summary(self, history: List[Dict]) -> Dict:
        """
        Рассчитать сводные метрики из истории snapshot-ов.

        Args:
            history: Список snapshot словарей.

        Returns:
            Словарь с avg_throughput_kbps, avg_rb_utilization_pct,
            avg_active_ue, total_snapshots.
        """
        if not history:
            return {}

        def avg(key: str) -> float:
            vals = [s.get(key, 0) for s in history if s.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        return {
            'total_snapshots':          len(history),
            'avg_throughput_kbps':      avg('dl_throughput_sum_kbps'),
            'avg_rb_allocated':         avg('dl_rb_allocated_count'),
            'avg_active_ue':            avg('sch_active_ue_count'),
            'avg_cqi':                  avg('dl_cqi_wb_avg_idx'),
            'avg_sinr_db':              avg('dl_sinr_avg'),
            'avg_buffer_bytes':         avg('buffer_size_sum_bytes'),
            'avg_sch_time_us':          avg('sch_total_time_us'),
        }

    def __repr__(self) -> str:
        state = 'ready'
        if self._ran:
            state = 'done'
        elif self._built:
            state = 'built'
        elif self._config is not None:
            state = 'configured'
        scheduler = self._config.scheduler if self._config else 'not configured'
        return f"SimulationFacade(state={state!r}, scheduler={scheduler!r})"
