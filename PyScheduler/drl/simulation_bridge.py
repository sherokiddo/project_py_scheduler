"""
Базовые контракты моста между runtime PyScheduler и DRL-компонентами.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class DRLSimulationRuntime:
    """
    Набор runtime-объектов, которые PyScheduler предоставляет DRL-слою.

    Структура намеренно остается легковесной. Здесь пока нет описания
    observation/action/reward. На этом этапе она только фиксирует те объекты,
    которые понадобятся будущему DRL-стеку для построения этих сущностей.
    """

    simulation_manager: Any
    base_station: Any
    ue_collection: Any
    lte_grid: Any
    scheduler: Any


@dataclass(slots=True)
class DRLSchedulerStepContext:
    """
    Снимок одного вызова scheduler внутри одного TTI симуляции.
    """

    current_time: int
    users: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DRLSimulationBridge:
    """
    Базовый lifecycle-мост между PyScheduler и будущими DRL-модулями.

    Цели дизайна:
    - сохранить `SimulationManager` главным оркестратором
    - оставить scheduler-логику расширяемой
    - дать DRL-слою стабильную точку доступа к runtime-состоянию
    - подготовить основу для observation builder, action adapter,
      parity-метрик и анимации
    """

    def __init__(self) -> None:
        self.runtime: Optional[DRLSimulationRuntime] = None

    def bind_runtime(self, runtime: DRLSimulationRuntime) -> None:
        """
        Привязать мост к текущему runtime симуляции.
        """

        self.runtime = runtime

    def require_runtime(self) -> DRLSimulationRuntime:
        """
        Вернуть привязанный runtime или выбросить понятную ошибку,
        если привязка еще не выполнена.
        """

        if self.runtime is None:
            raise RuntimeError(
                "DRL bridge еще не привязан к runtime симуляции."
            )

        return self.runtime

    def on_simulation_start(self, runtime: DRLSimulationRuntime) -> None:
        """
        Lifecycle-hook, вызываемый один раз после подготовки runtime симуляции.
        """

    def before_scheduler_step(self, step_context: DRLSchedulerStepContext) -> None:
        """
        Lifecycle-hook, вызываемый непосредственно перед
        `scheduler.schedule(...)`.
        """

    def after_scheduler_step(
        self,
        step_context: DRLSchedulerStepContext,
        scheduler_result: Dict[str, Any],
    ) -> None:
        """
        Lifecycle-hook, вызываемый непосредственно после
        `scheduler.schedule(...)`.
        """

    def on_simulation_end(self) -> None:
        """
        Lifecycle-hook, вызываемый один раз при завершении симуляции.
        """
