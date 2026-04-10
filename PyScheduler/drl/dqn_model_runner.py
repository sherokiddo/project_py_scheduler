"""
Рантайм-обертка над DQN-моделью для использования внутри PyScheduler.
"""

from typing import Any, Optional

import numpy as np


class DQNModelRunner:
    """
    Унифицированный runner для инференса DQN-модели.

    Может работать либо с уже готовым объектом агента, либо загружать его
    из файла весов.
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
                "Для DQNModelRunner необходимо передать либо model_path, либо agent."
            )

    @property
    def agent(self) -> Any:
        """
        Вернуть загруженный объект агента, при необходимости загрузив его лениво.
        """

        if self._agent is None:
            from drl.dqn_agent import LTEDQNAgent

            self._agent = LTEDQNAgent.load(
                path=str(self.model_path),
                device=self.device,
            )
        return self._agent

    @property
    def max_n_ue(self) -> Optional[int]:
        """
        Число UE, на которое рассчитана модель.
        """

        value = getattr(self.agent, "max_n_ue", None)
        return int(value) if value is not None else None

    @property
    def ue_feature_dim(self) -> Optional[int]:
        """
        Число признаков на один UE.
        """

        value = getattr(self.agent, "ue_feature_dim", None)
        return int(value) if value is not None else None

    @property
    def context_dim(self) -> Optional[int]:
        """
        Число глобальных признаков observation.
        """

        value = getattr(self.agent, "context_dim", None)
        return int(value) if value is not None else None

    def predict(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: Optional[bool] = None,
    ) -> int:
        """
        Выполнить предсказание действия.
        """

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
