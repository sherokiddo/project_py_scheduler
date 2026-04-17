"""
Подготовка PPO ranker к TorchScript/LibTorch inference.

Модуль выделяет inference-only wrapper над `LTEPPORankerAgent.policy`, чтобы:
- зафиксировать production-контракт входов/выходов;
- экспортировать модель в TorchScript без training-логики PPO;
- упростить последующую интеграцию через LibTorch в C++.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from drl.agents.lte_ppo_ranker_agent import LTEPPORankerAgent


MASKED_INVALID_SCORE = -1e9


class TorchScriptPPORankerInference(nn.Module):
    """
    Inference-only wrapper для TorchScript/LibTorch.

    Контракт:
    - input:
        obs         : [batch, obs_dim]      float32
        action_mask : [batch, max_n_ue]     bool
    - output:
        score_vector: [batch, max_n_ue]     float32

    Семантика совпадает с `LTEPPORankerAgent.predict_scores(..., deterministic=True)`:
    - если action-mask в строке пустой, внутренняя policy вычисляется на all-valid mask;
    - на выходе все invalid позиции принудительно маскируются значением `-1e9`.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        max_n_ue: int,
        ue_feature_dim: int,
        context_dim: int,
        invalid_score: float = MASKED_INVALID_SCORE,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.max_n_ue = int(max_n_ue)
        self.ue_feature_dim = int(ue_feature_dim)
        self.context_dim = int(context_dim)
        self.obs_dim = int(self.max_n_ue * self.ue_feature_dim + self.context_dim)
        self.invalid_score = float(invalid_score)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = action_mask.to(dtype=torch.bool)
        valid_counts = mask.sum(dim=1, keepdim=True)
        all_valid_mask = torch.ones_like(mask, dtype=torch.bool)
        sanitized_mask = torch.where(valid_counts > 0, mask, all_valid_mask)

        score_mean, _, _ = self.policy(obs, sanitized_mask)
        invalid_fill = torch.full_like(score_mean, self.invalid_score)
        return torch.where(mask, score_mean, invalid_fill)


def build_torchscript_ranker_inference(
    agent: LTEPPORankerAgent,
    *,
    device: Optional[torch.device] = None,
) -> TorchScriptPPORankerInference:
    """
    Построить inference-wrapper над уже загруженным PPO ranker-агентом.
    """
    target_device = device or agent.device
    policy_copy = copy.deepcopy(agent.policy)
    wrapper = TorchScriptPPORankerInference(
        policy=policy_copy,
        max_n_ue=agent.max_n_ue,
        ue_feature_dim=agent.ue_feature_dim,
        context_dim=agent.context_dim,
    )
    wrapper = wrapper.to(target_device)
    wrapper.eval()
    return wrapper


def build_ranker_export_metadata(
    agent: LTEPPORankerAgent,
    *,
    torchscript_path: str | Path,
    invalid_score: float = MASKED_INVALID_SCORE,
) -> dict[str, Any]:
    """
    Сформировать metadata для TorchScript ranker artifact.
    """
    return {
        "format": "torchscript",
        "model_type": "LTEPPORankerAgent",
        "artifact_path": str(torchscript_path),
        "max_n_ue": int(agent.max_n_ue),
        "ue_feature_dim": int(agent.ue_feature_dim),
        "context_dim": int(agent.context_dim),
        "obs_dim": int(agent.max_n_ue * agent.ue_feature_dim + agent.context_dim),
        "input_names": ["obs", "action_mask"],
        "input_dtypes": {
            "obs": "float32",
            "action_mask": "bool",
        },
        "output_names": ["score_vector"],
        "output_dtypes": {
            "score_vector": "float32",
        },
        "deterministic_only": True,
        "masked_invalid_score": float(invalid_score),
        "notes": [
            "Artifact contains only inference path, without PPO training/update logic.",
            "If action-mask row is empty, internal policy is evaluated on all-valid mask, then output is fully masked to invalid_score.",
        ],
    }


def export_ranker_to_torchscript(
    *,
    checkpoint_path: str | Path,
    torchscript_path: str | Path,
    metadata_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Загрузить PPO ranker checkpoint и экспортировать inference-wrapper в TorchScript.
    """
    resolved_checkpoint_path = Path(checkpoint_path)
    resolved_torchscript_path = Path(torchscript_path)
    resolved_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else resolved_torchscript_path.with_suffix(".meta.json")
    )

    resolved_device = torch.device(device)
    agent = LTEPPORankerAgent.load(
        path=str(resolved_checkpoint_path),
        device=resolved_device,
    )
    wrapper = build_torchscript_ranker_inference(
        agent,
        device=resolved_device,
    )

    scripted_module = torch.jit.script(wrapper)
    resolved_torchscript_path.parent.mkdir(parents=True, exist_ok=True)
    scripted_module.save(str(resolved_torchscript_path))

    metadata = build_ranker_export_metadata(
        agent,
        torchscript_path=resolved_torchscript_path,
    )
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    return {
        "torchscript_path": str(resolved_torchscript_path),
        "metadata_path": str(resolved_metadata_path),
        "metadata": metadata,
    }
