"""
Unified runtime runners for DRL models inside PyScheduler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


class BaseTorchAgentRunner:
    """
    Common lazy-loading runtime wrapper for torch agents.
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
                f"{self.__class__.__name__} requires either model_path or agent."
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

    def initialize(self) -> None:
        _ = self.agent

    @staticmethod
    def _load_agent(*, path: str, device: Optional[Any]) -> Any:
        raise NotImplementedError


class DQNModelRunner(BaseTorchAgentRunner):
    @staticmethod
    def _load_agent(*, path: str, device: Optional[Any]) -> Any:
        from drl.agents.lte_dqn_agent import LTEDQNAgent

        return LTEDQNAgent.load(path=path, device=device)


class PPOModelRunner(BaseTorchAgentRunner):
    @staticmethod
    def _load_agent(*, path: str, device: Optional[Any]) -> Any:
        from drl.agents.lte_ppo_agent import LTEPPOAgent

        return LTEPPOAgent.load(path=path, device=device)


class PPORankerModelRunner(BaseTorchAgentRunner):
    """
    Runtime runner for PPO ranker torch model.
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


class CppPPORankerModelRunner:
    """
    Runtime runner for PPO ranker through C++/LibTorch backend.
    """

    def __init__(
        self,
        *,
        model_path: str,
        runtime_library_path: str,
        dll_search_paths: Optional[Sequence[str | Path]] = None,
        deterministic: bool = True,
        max_n_ue: Optional[int] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.runtime_library_path = str(runtime_library_path)
        self.dll_search_paths = (
            None
            if dll_search_paths is None
            else [str(Path(path)) for path in dll_search_paths]
        )
        self.deterministic = bool(deterministic)
        self._max_n_ue = None if max_n_ue is None else int(max_n_ue)
        self._bridge = None

    @property
    def bridge(self) -> Any:
        if self._bridge is None:
            from drl.cpp_libtorch_ranker_bridge import CppPPORankerBridge

            self._bridge = CppPPORankerBridge(
                runtime_library_path=self.runtime_library_path,
                model_path=self.model_path,
                dll_search_paths=self.dll_search_paths,
            )
        return self._bridge

    @property
    def inference_device(self) -> str:
        return "cpu_cpp_libtorch"

    @property
    def max_n_ue(self) -> Optional[int]:
        return self._max_n_ue

    @property
    def ue_feature_dim(self) -> Optional[int]:
        return None

    @property
    def context_dim(self) -> Optional[int]:
        return None

    def predict(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: Optional[bool] = None,
    ) -> np.ndarray:
        use_deterministic = (
            self.deterministic if deterministic is None else bool(deterministic)
        )
        if not use_deterministic:
            raise ValueError(
                "CppPPORankerModelRunner supports only deterministic inference."
            )

        return np.asarray(
            self.bridge.predict_scores(
                obs=np.asarray(obs, dtype=np.float32),
                action_mask=np.asarray(action_mask, dtype=bool),
            ),
            dtype=np.float32,
        )

    def initialize(self) -> None:
        _ = self.bridge

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None


class CppONNXPPORankerModelRunner:
    """
    Runtime runner for PPO ranker through C++/ONNX Runtime backend.
    """

    def __init__(
        self,
        *,
        model_path: str,
        runtime_library_path: str,
        dll_search_paths: Optional[Sequence[str | Path]] = None,
        deterministic: bool = True,
        max_n_ue: Optional[int] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.runtime_library_path = str(runtime_library_path)
        self.dll_search_paths = (
            None
            if dll_search_paths is None
            else [str(Path(path)) for path in dll_search_paths]
        )
        self.deterministic = bool(deterministic)
        self._max_n_ue = None if max_n_ue is None else int(max_n_ue)
        self._bridge = None

    @property
    def bridge(self) -> Any:
        if self._bridge is None:
            from drl.cpp_onnx_ranker_bridge import CppONNXPPORankerBridge

            self._bridge = CppONNXPPORankerBridge(
                runtime_library_path=self.runtime_library_path,
                model_path=self.model_path,
                dll_search_paths=self.dll_search_paths,
            )
        return self._bridge

    @property
    def inference_device(self) -> str:
        return "cpu_cpp_onnxruntime"

    @property
    def max_n_ue(self) -> Optional[int]:
        return self._max_n_ue

    @property
    def ue_feature_dim(self) -> Optional[int]:
        return None

    @property
    def context_dim(self) -> Optional[int]:
        return None

    def predict(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        deterministic: Optional[bool] = None,
    ) -> np.ndarray:
        use_deterministic = (
            self.deterministic if deterministic is None else bool(deterministic)
        )
        if not use_deterministic:
            raise ValueError(
                "CppONNXPPORankerModelRunner supports only deterministic inference."
            )

        return np.asarray(
            self.bridge.predict_scores(
                obs=np.asarray(obs, dtype=np.float32),
                action_mask=np.asarray(action_mask, dtype=bool),
            ),
            dtype=np.float32,
        )

    def initialize(self) -> None:
        _ = self.bridge

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
