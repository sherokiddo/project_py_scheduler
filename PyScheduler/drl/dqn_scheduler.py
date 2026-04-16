"""
Inference-only DQN scheduler для PyScheduler.
"""

from typing import Any, Dict, List, Optional

from drl.drl_scheduler import DrlScheduler
from drl.model_runners import DQNModelRunner


class DqnScheduler(DrlScheduler):
    """
    DQN-планировщик поверх общего runtime-layer `DrlScheduler`.

    Общий DRL pipeline живет в базовом классе. Эта реализация отвечает только
    за DQN-специфичные параметры, создание runner'а и совместимые trace-алиасы.
    """

    def __init__(self, lte_grid, bs, **kwargs):
        dqn_model_path = kwargs.pop("dqn_model_path", None)
        dqn_max_n_ue = kwargs.pop("dqn_max_n_ue", None)
        dqn_wb_cqi_report_period_tti = kwargs.pop(
            "dqn_wb_cqi_report_period_tti", 5
        )
        dqn_episode_len_tti = kwargs.pop("dqn_episode_len_tti", None)
        dqn_strict_observation = kwargs.pop("dqn_strict_observation", False)
        dqn_deterministic = kwargs.pop("dqn_deterministic", True)
        dqn_inference_device = kwargs.pop("dqn_inference_device", "cpu")
        dqn_policy_runner = kwargs.pop("dqn_policy_runner", None)

        super().__init__(
            lte_grid,
            bs,
            model_path=dqn_model_path,
            max_n_ue=dqn_max_n_ue,
            wb_cqi_report_period_tti=dqn_wb_cqi_report_period_tti,
            episode_len_tti=dqn_episode_len_tti,
            strict_observation=dqn_strict_observation,
            deterministic=dqn_deterministic,
            inference_device=dqn_inference_device,
            policy_runner=dqn_policy_runner,
            stats_prefix="dqn",
            **kwargs,
        )

        self.dqn_model_path = self.model_path
        self.dqn_max_n_ue = self.max_n_ue
        self.dqn_wb_cqi_report_period_tti = self.wb_cqi_report_period_tti
        self.dqn_episode_len_tti = self.episode_len_tti
        self.dqn_strict_observation = self.strict_observation
        self.dqn_deterministic = self.deterministic
        self.dqn_inference_device = self.inference_device
        self._sync_dqn_trace_aliases()

    def _allocate_pdsch(
        self,
        tti: int,
        ues_with_pdcch: List[Dict],
        eligible_ues: List[Dict],
    ) -> Dict[int, List[int]]:
        """
        Выполнить per-RBG распределение через DQN policy и обновить trace-алиасы.
        """

        allocation = super()._allocate_pdsch(
            tti=tti,
            ues_with_pdcch=ues_with_pdcch,
            eligible_ues=eligible_ues,
        )
        self._sync_dqn_trace_aliases()
        return allocation

    def get_stats(self) -> Dict:
        """
        Вернуть базовую DRL-статистику и синхронизировать DQN-алиасы.
        """

        self._sync_dqn_trace_aliases()
        return super().get_stats()

    def _reset_dqn_step_trace(self) -> None:
        """
        Сбросить trace последнего inference TTI в DQN-совместимых полях.
        """

        self._reset_step_trace()
        self._sync_dqn_trace_aliases()

    def _sync_dqn_trace_aliases(self) -> None:
        """
        Поддержать DQN-специфичные trace-поля для совместимости.
        """

        self._last_dqn_invalid_action_count = self._last_invalid_action_count
        self._last_dqn_selected_ue_ids = list(self._last_selected_ue_ids)
        self._last_dqn_raw_actions = list(self._last_raw_actions)
        self._last_dqn_step_count = self._last_step_count

    @staticmethod
    def _build_policy_runner(
        *,
        model_path: Optional[str],
        provided_runner: Optional[Any],
        deterministic: bool,
        device: Optional[Any],
    ) -> Any:
        """
        Построить runner для DQN policy.
        """

        if provided_runner is not None:
            return provided_runner

        if not model_path:
            raise ValueError(
                "Для DqnScheduler необходимо указать dqn_model_path или dqn_policy_runner."
            )

        return DQNModelRunner(
            model_path=str(model_path),
            deterministic=deterministic,
            device=device,
        )
