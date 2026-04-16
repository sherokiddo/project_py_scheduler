"""
Inference-only PPO scheduler для PyScheduler.
"""

from typing import Any, Optional

from drl.drl_scheduler import DrlScheduler
from drl.model_runners import PPOModelRunner


class PpoScheduler(DrlScheduler):
    """
    PPO-планировщик поверх общего runtime-layer `DrlScheduler`.

    Класс использует тот же общий DRL pipeline, что и DQN-вариант, но работает
    с PPO policy и собственным набором `ppo_*` параметров конфигурации.
    """

    def __init__(self, lte_grid, bs, **kwargs):
        ppo_model_path = kwargs.pop("ppo_model_path", None)
        ppo_max_n_ue = kwargs.pop("ppo_max_n_ue", None)
        ppo_wb_cqi_report_period_tti = kwargs.pop(
            "ppo_wb_cqi_report_period_tti", 5
        )
        ppo_episode_len_tti = kwargs.pop("ppo_episode_len_tti", None)
        ppo_strict_observation = kwargs.pop("ppo_strict_observation", False)
        ppo_deterministic = kwargs.pop("ppo_deterministic", True)
        ppo_inference_device = kwargs.pop("ppo_inference_device", "cpu")
        ppo_policy_runner = kwargs.pop("ppo_policy_runner", None)

        super().__init__(
            lte_grid,
            bs,
            model_path=ppo_model_path,
            max_n_ue=ppo_max_n_ue,
            wb_cqi_report_period_tti=ppo_wb_cqi_report_period_tti,
            episode_len_tti=ppo_episode_len_tti,
            strict_observation=ppo_strict_observation,
            deterministic=ppo_deterministic,
            inference_device=ppo_inference_device,
            policy_runner=ppo_policy_runner,
            stats_prefix="ppo",
            **kwargs,
        )

        self.ppo_model_path = self.model_path
        self.ppo_max_n_ue = self.max_n_ue
        self.ppo_wb_cqi_report_period_tti = self.wb_cqi_report_period_tti
        self.ppo_episode_len_tti = self.episode_len_tti
        self.ppo_strict_observation = self.strict_observation
        self.ppo_deterministic = self.deterministic
        self.ppo_inference_device = self.inference_device

    @staticmethod
    def _build_policy_runner(
        *,
        model_path: Optional[str],
        provided_runner: Optional[Any],
        deterministic: bool,
        device: Optional[Any],
    ) -> Any:
        if provided_runner is not None:
            return provided_runner

        if not model_path:
            raise ValueError(
                "Для PpoScheduler необходимо указать ppo_model_path или ppo_policy_runner."
            )

        return PPOModelRunner(
            model_path=str(model_path),
            deterministic=deterministic,
            device=device,
        )
