"""
#------------------------------------------------------------------------------
# Паттерн: Template Method — жизненный цикл модуля симуляции
#------------------------------------------------------------------------------
# Описание:
#   SimulationModule — абстрактный базовый класс, определяющий скелет
#   жизненного цикла любого модуля симуляции.
#
#   Паттерн Template Method фиксирует порядок шагов (initialize → run_step
#   → finalize), оставляя конкретным классам реализацию каждого шага.
#   Хуки (pre_step, post_step, on_error) опциональны и имеют пустую
#   реализацию по умолчанию — переопределять нужно только то, что нужно.
#
#   Жизненный цикл модуля:
#
#     initialize()          ← вызывается один раз перед симуляцией
#         │
#     for each TTI:
#         ├── pre_step(tti)     ← хук: подготовка (логирование, валидация)
#         ├── step(tti)         ← ОБЯЗАТЕЛЕН: основная логика TTI
#         └── post_step(tti)    ← хук: постобработка (метрики, события)
#         │
#     finalize()            ← вызывается один раз после симуляции
#
#   Пример нового модуля::
#
#       class MyTrafficModule(SimulationModule):
#           def initialize(self):
#               self.total_bytes = 0
#
#           def step(self, tti: int):
#               bytes_gen = self.traffic_model.generate(tti)
#               self.total_bytes += bytes_gen
#
#           def finalize(self):
#               print(f"Итого сгенерировано: {self.total_bytes} байт")
#
#       module = MyTrafficModule(name='traffic')
#       module.initialize()
#       for tti in range(1000):
#           module.run_step(tti)
#       module.finalize()
#
# Версия: 1.0.0
#------------------------------------------------------------------------------
"""

from __future__ import annotations

import traceback
import warnings
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from patterns.observer import EventBus, EventType, SimulationEvent


# ==============================================================================
#                          СОСТОЯНИЯ МОДУЛЯ
# ==============================================================================

class ModuleState(Enum):
    """
    Состояния жизненного цикла модуля.

    Переходы:
        CREATED → INITIALIZED → RUNNING → FINALIZED
                      ↑                      ↓
                   RESET ←────────────── FINALIZED
                                   (при ошибке → ERROR)
    """
    CREATED     = auto()   # Объект создан, initialize() не вызывался
    INITIALIZED = auto()   # initialize() выполнен успешно
    RUNNING     = auto()   # Идёт выполнение step()
    FINALIZED   = auto()   # finalize() выполнен
    ERROR       = auto()   # Произошла ошибка в step() или initialize()


# ==============================================================================
#                         БАЗОВЫЙ КЛАСС МОДУЛЯ
# ==============================================================================

class SimulationModule(ABC):
    """
    Паттерн Template Method: скелет жизненного цикла модуля симуляции.

    Любой компонент симуляции (планировщик, модель трафика, сборщик метрик,
    модель канала) может наследоваться от этого класса и получить:
    - стандартный жизненный цикл initialize → step → finalize
    - автоматическое управление состоянием (ModuleState)
    - хуки pre_step / post_step для расширения поведения
    - обработку ошибок через on_error
    - интеграцию с EventBus (опционально)

    Подклассы ДОЛЖНЫ реализовать:
        - initialize() — настройка при старте
        - step(tti)    — логика одного TTI
        - finalize()   — завершение работы

    Подклассы МОГУТ переопределить (хуки):
        - pre_step(tti)  — до step()
        - post_step(tti) — после step()
        - on_error(tti, exc) — при ошибке в step()
    """

    def __init__(
        self,
        name: Optional[str] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """
        Args:
            name: Имя модуля для логов и событий.
                По умолчанию — имя класса.
            event_bus: Шина событий для публикации TTI-событий.
                Если None — события не публикуются.
        """
        self.name: str = name or self.__class__.__name__
        self._event_bus: Optional[EventBus] = event_bus
        self._state: ModuleState = ModuleState.CREATED
        self._current_tti: int = -1
        self._step_count: int = 0
        self._error_count: int = 0

    # ------------------------------------------------------------------
    #  Template Method — публичный API (не переопределять!)
    # ------------------------------------------------------------------

    def run_step(self, tti: int) -> bool:
        """
        Шаблонный метод: выполнить один шаг симуляции.

        Последовательность:
            1. pre_step(tti)
            2. step(tti)         ← обязательная реализация в подклассе
            3. post_step(tti)

        При ошибке в step() вызывает on_error(tti, exc).

        Args:
            tti: Номер текущего TTI.

        Returns:
            True — шаг выполнен успешно, False — произошла ошибка.

        Raises:
            RuntimeError: Если initialize() не был вызван перед run_step().
        """
        if self._state not in (ModuleState.INITIALIZED, ModuleState.RUNNING):
            raise RuntimeError(
                f"[{self.name}] run_step() вызван до initialize(). "
                f"Текущее состояние: {self._state.name}"
            )

        self._state = ModuleState.RUNNING
        self._current_tti = tti

        try:
            self.pre_step(tti)
            self.step(tti)
            self.post_step(tti)
            self._step_count += 1
            return True

        except Exception as exc:
            self._error_count += 1
            self._state = ModuleState.ERROR
            self.on_error(tti, exc)
            return False

    def run_all(self, ttis: List[int]) -> Dict[str, Any]:
        """
        Выполнить initialize → все TTI → finalize в одном вызове.

        Удобен для изолированного запуска модуля в тестах или benchmarks.

        Args:
            ttis: Список номеров TTI для выполнения.

        Returns:
            Словарь с результатами: {'success': bool, 'steps': int, 'errors': int}
        """
        self.initialize()
        for tti in ttis:
            self.run_step(tti)
            if self._state == ModuleState.ERROR:
                break
        self.finalize()

        return {
            'success': self._error_count == 0,
            'steps': self._step_count,
            'errors': self._error_count,
            'module': self.name,
        }

    # ------------------------------------------------------------------
    #  Абстрактные методы — ОБЯЗАТЕЛЬНЫ к реализации
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """
        Инициализация модуля перед запуском симуляции.

        Создаёт внутренние структуры данных, устанавливает начальное
        состояние, загружает конфигурацию.

        Вызывается ровно один раз перед первым run_step().
        """
        ...

    @abstractmethod
    def step(self, tti: int) -> None:
        """
        Основная логика одного TTI.

        Args:
            tti: Номер текущего TTI (начинается с 1).
        """
        ...

    @abstractmethod
    def finalize(self) -> None:
        """
        Завершение работы модуля после окончания симуляции.

        Экспорт данных, освобождение ресурсов, итоговый лог.
        Вызывается ровно один раз после последнего run_step().
        """
        ...

    # ------------------------------------------------------------------
    #  Хуки — опциональны, имеют пустую реализацию по умолчанию
    # ------------------------------------------------------------------

    def pre_step(self, tti: int) -> None:
        """
        Хук: выполняется ДО step(tti).

        Типичное использование: логирование, валидация входных данных,
        инкремент счётчиков. По умолчанию — пустая операция.

        Args:
            tti: Номер текущего TTI.
        """

    def post_step(self, tti: int) -> None:
        """
        Хук: выполняется ПОСЛЕ step(tti).

        Типичное использование: публикация событий, сбор метрик,
        обновление sliding window. По умолчанию — пустая операция.

        Args:
            tti: Номер текущего TTI.
        """
        if self._event_bus is not None:
            self._event_bus.publish_simple(
                EventType.TTI_COMPLETED,
                data={'module': self.name, 'tti': tti},
                source=self.name,
                tti=tti,
            )

    def on_error(self, tti: int, exc: Exception) -> None:
        """
        Хук: вызывается при исключении в step(tti).

        По умолчанию выводит предупреждение с трейсбеком.
        Переопределить чтобы: логировать в файл, откатить состояние,
        восстановить модуль, или пробросить исключение выше.

        Args:
            tti: TTI, на котором произошла ошибка.
            exc: Пойманное исключение.
        """
        tb = traceback.format_exc()
        warnings.warn(
            f"[{self.name}] Ошибка на TTI {tti}: {type(exc).__name__}: {exc}\n{tb}",
            RuntimeWarning,
            stacklevel=3,
        )

        if self._event_bus is not None:
            self._event_bus.publish_simple(
                EventType.SIMULATION_ERROR,
                data={'module': self.name, 'tti': tti, 'error': str(exc)},
                source=self.name,
                tti=tti,
            )

    # ------------------------------------------------------------------
    #  Управление состоянием
    # ------------------------------------------------------------------

    def _mark_initialized(self) -> None:
        """
        Пометить модуль как инициализированный.
        Должен вызываться в конце реализации initialize().

        Пример::

            def initialize(self):
                self.buffer = []
                self._mark_initialized()   # ← обязательно
        """
        self._state = ModuleState.INITIALIZED
        self._step_count = 0
        self._error_count = 0

    def _mark_finalized(self) -> None:
        """
        Пометить модуль как завершённый.
        Должен вызываться в конце реализации finalize().
        """
        self._state = ModuleState.FINALIZED

    def reset(self) -> None:
        """
        Сбросить модуль в состояние CREATED.

        Позволяет переиспользовать объект для повторных симуляций
        без пересоздания.
        """
        self._state = ModuleState.CREATED
        self._current_tti = -1
        self._step_count = 0
        self._error_count = 0

    # ------------------------------------------------------------------
    #  Свойства
    # ------------------------------------------------------------------

    @property
    def state(self) -> ModuleState:
        """Текущее состояние жизненного цикла."""
        return self._state

    @property
    def is_running(self) -> bool:
        """True если модуль в состоянии RUNNING."""
        return self._state == ModuleState.RUNNING

    @property
    def is_error(self) -> bool:
        """True если модуль в состоянии ERROR."""
        return self._state == ModuleState.ERROR

    @property
    def step_count(self) -> int:
        """Количество успешно выполненных шагов."""
        return self._step_count

    @property
    def error_count(self) -> int:
        """Количество ошибок за время работы модуля."""
        return self._error_count

    @property
    def current_tti(self) -> int:
        """Номер последнего выполненного TTI (-1 если step не вызывался)."""
        return self._current_tti

    def get_stats(self) -> Dict[str, Any]:
        """
        Базовая статистика модуля.
        Переопределяется в подклассах для добавления специфичных метрик.

        Returns:
            Словарь с именем, состоянием, шагами, ошибками.
        """
        return {
            'module':        self.name,
            'state':         self._state.name,
            'step_count':    self._step_count,
            'error_count':   self._error_count,
            'current_tti':   self._current_tti,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"state={self._state.name}, "
            f"steps={self._step_count})"
        )


# ==============================================================================
#                    СОСТАВНОЙ МОДУЛЬ (Composite)
# ==============================================================================

class CompositeModule(SimulationModule):
    """
    Составной модуль — управляет жизненным циклом дочерних модулей.

    Позволяет группировать несколько SimulationModule под одним менеджером.
    Дочерние модули выполняются в порядке добавления.

    Пример::

        pipeline = CompositeModule(name='main_pipeline')
        pipeline.add(MobilityModule(...))
        pipeline.add(ChannelModule(...))
        pipeline.add(TrafficModule(...))
        pipeline.add(SchedulerModule(...))

        for tti in range(1000):
            pipeline.run_step(tti)
    """

    def __init__(self, name: Optional[str] = None,
                 event_bus: Optional[EventBus] = None) -> None:
        super().__init__(name=name or 'CompositeModule', event_bus=event_bus)
        self._children: List[SimulationModule] = []

    def add(self, module: SimulationModule) -> "CompositeModule":
        """
        Добавить дочерний модуль.

        Args:
            module: Экземпляр SimulationModule.

        Returns:
            self — для fluent interface: ``pipeline.add(m1).add(m2)``.
        """
        self._children.append(module)
        return self

    def remove(self, module: SimulationModule) -> None:
        """
        Удалить дочерний модуль.

        Args:
            module: Ранее добавленный модуль.
        """
        self._children.remove(module)

    # ------------------------------------------------------------------
    #  Template Method реализация
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Инициализировать все дочерние модули."""
        for child in self._children:
            child.initialize()
        self._mark_initialized()

    def step(self, tti: int) -> None:
        """Выполнить step() для всех дочерних модулей."""
        for child in self._children:
            child.run_step(tti)

    def finalize(self) -> None:
        """Финализировать все дочерние модули."""
        for child in self._children:
            child.finalize()
        self._mark_finalized()

    def get_stats(self) -> Dict[str, Any]:
        """Агрегированная статистика всех дочерних модулей."""
        base = super().get_stats()
        base['children'] = [child.get_stats() for child in self._children]
        return base
