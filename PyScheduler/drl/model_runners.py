"""
Унифицированные runtime-runner'ы для DRL-моделей внутри PyScheduler.
"""

from typing import Any, Optional

import numpy as np


class BaseTorchAgentRunner:
    """
    Общая рантайм-обертка над torch-агентом.

    Подклассы отвечают только за ленивую загрузку конкретного типа агента.
    """

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        agent: Optional[Any] = None,
        device: Optional[Any] = None,
        deterministic: bool = True,
    ) -> None:
        self.model_path = model_path
        self._agent = agent
        self.device = device
        self.deterministic = bool(deterministic)

        if self._agent is None and not self.model_path:
            raise ValueError(
                f"Для {self.__class__.__name__} необходимо передать либо model_path, либо agent."
            )

    @property
    def agent(self) -> Any:
        if self._agent is None:
            self._agent = self._load_agent(
                path=str(self.model_path),
                device=self.device,
            )
        return self._agent

    @property
    def inference_device(self) -> str:
        agent_device = getattr(self._agent, "device", None)
        if agent_device is not None:
            return str(agent_device)

        if self.device is None:
            return "auto"

        return str(self.device)

    @property
    def max_n_ue(self) -> Optional[int]:
        value = getattr(self.agent, "max_n_ue", None)
        return int(value) if value is not None else None

    @property
    def ue_feature_dim(self) -> Optional[int]:
        value = getattr(self.agent, "ue_feature_dim", None)
        return int(value) if value is not None else None

    @property
    def context_dim(self) -> Optional[int]:
        value = getattr(self.agent, "context_dim", None)
        return int(value) if value is not None else None

    def predict(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: Optional[bool] = None,
    ) -> int:
        use_deterministic = (
            self.deterministic if deterministic is None else bool(deterministic)
        )
        return int(
            self.agent.predict(
                obs=np.asarray(obs, dtype=np.float32),
                action_mask=np.asarray(action_mask, dtype=bool),
                deterministic=use_deterministic,
            )
        )

    @staticmethod
    def _load_agent(*, path: str, device: Optional[Any]) -> Any:
        raise NotImplementedError(
            "BaseTorchAgentRunner._load_agent() должен быть реализован в подклассе."
        )


class DQNModelRunner(BaseTorchAgentRunner):
    """
    Runtime-runner для DQN-модели.
    """

    @staticmethod
    def _load_agent(*, path: str, device: Optional[Any]) -> Any:
        from drl.agents.lte_dqn_agent import LTEDQNAgent

        return LTEDQNAgent.load(path=path, device=device)


class PPOModelRunner(BaseTorchAgentRunner):
    """
    Runtime-runner для PPO-модели.
    """

    @staticmethod
    def _load_agent(*, path: str, device: Optional[Any]) -> Any:
        from drl.agents.lte_ppo_agent import LTEPPOAgent

        return LTEPPOAgent.load(path=path, device=device)


class PPORankerModelRunner(BaseTorchAgentRunner):
    """
    Runtime-runner для PPO ranker-модели.

    В отличие от per-RBG runner'ов, ranker возвращает не индекс действия, а
    полный score-вектор по всем UE текущего TTI.
    """

    @staticmethod
    def _load_agent(*, path: str, device: Optional[Any]) -> Any:
        from drl.agents.lte_ppo_ranker_agent import LTEPPORankerAgent

        return LTEPPORankerAgent.load(path=path, device=device)

    def predict(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: Optional[bool] = None,
    ) -> np.ndarray:
        use_deterministic = (
            self.deterministic if deterministic is None else bool(deterministic)
        )
        return np.asarray(
            self.agent.predict(
                obs=np.asarray(obs, dtype=np.float32),
                action_mask=np.asarray(action_mask, dtype=bool),
                deterministic=use_deterministic,
            ),
            dtype=np.float32,
        )
