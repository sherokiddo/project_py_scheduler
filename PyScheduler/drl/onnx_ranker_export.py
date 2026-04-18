"""
Экспорт PPO ranker в ONNX для CPU runtime-инференса.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn

from drl.agents.lte_ppo_ranker_agent import LTEPPORankerAgent
from drl.torchscript_ranker_export import MASKED_INVALID_SCORE


DEFAULT_ONNX_OPSET_VERSION = 15


class ONNXPPORankerInference(nn.Module):
    """
    ONNX-friendly actor-only inference wrapper без `torch.where`.
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
        valid_rows = mask.any(dim=1, keepdim=True)
        fallback_mask = torch.logical_not(valid_rows).expand_as(mask)
        sanitized_mask = torch.logical_or(mask, fallback_mask)

        score_mean, _ = self.policy.forward_actor(obs, sanitized_mask)

        mask_f = mask.to(dtype=score_mean.dtype)
        invalid_fill = torch.full_like(score_mean, self.invalid_score)
        return score_mean * mask_f + invalid_fill * (1.0 - mask_f)


def build_ranker_onnx_inference(
    agent: LTEPPORankerAgent,
    *,
    device: torch.device | None = None,
) -> ONNXPPORankerInference:
    """
    Построить ONNX-friendly inference-wrapper над PPO ranker агентом.
    """
    target_device = device or agent.device
    policy_copy = agent.policy.__class__(
        max_n_ue=agent.max_n_ue,
        ue_feature_dim=agent.ue_feature_dim,
        context_dim=agent.context_dim,
        hidden_dim=agent.hidden_dim,
    )
    policy_copy.load_state_dict(agent.policy.state_dict())
    wrapper = ONNXPPORankerInference(
        policy=policy_copy,
        max_n_ue=agent.max_n_ue,
        ue_feature_dim=agent.ue_feature_dim,
        context_dim=agent.context_dim,
    )
    wrapper = wrapper.to(target_device)
    wrapper.eval()
    return wrapper


def build_ranker_onnx_export_metadata(
    agent: LTEPPORankerAgent,
    *,
    onnx_path: str | Path,
    opset_version: int = DEFAULT_ONNX_OPSET_VERSION,
    invalid_score: float = MASKED_INVALID_SCORE,
) -> dict[str, Any]:
    """
    Сформировать metadata для ONNX-artifact PPO ranker.
    """
    return {
        "format": "onnx",
        "model_type": "LTEPPORankerAgent",
        "artifact_path": str(onnx_path),
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
        "runtime_actor_only": True,
        "masked_invalid_score": float(invalid_score),
        "onnx_opset_version": int(opset_version),
        "notes": [
            "Artifact contains only inference path, without PPO training/update logic.",
            "Runtime forward uses actor-only branch and does not evaluate critic/value head.",
            "If action-mask row is empty, internal policy is evaluated on all-valid mask, then output is fully masked to invalid_score.",
        ],
    }


def export_ranker_to_onnx(
    *,
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    metadata_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    opset_version: int = DEFAULT_ONNX_OPSET_VERSION,
) -> dict[str, Any]:
    """
    Загрузить PPO ranker checkpoint и экспортировать actor-only inference-path в ONNX.
    """
    resolved_checkpoint_path = Path(checkpoint_path)
    resolved_onnx_path = Path(onnx_path)
    resolved_metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else resolved_onnx_path.with_suffix(".meta.json")
    )

    resolved_device = torch.device(device)
    agent = LTEPPORankerAgent.load(
        path=str(resolved_checkpoint_path),
        device=resolved_device,
    )
    wrapper = build_ranker_onnx_inference(
        agent,
        device=resolved_device,
    )

    obs_dim = int(agent.max_n_ue * agent.ue_feature_dim + agent.context_dim)
    dummy_obs = torch.zeros(
        (1, obs_dim),
        dtype=torch.float32,
        device=resolved_device,
    )
    dummy_action_mask = torch.ones(
        (1, agent.max_n_ue),
        dtype=torch.bool,
        device=resolved_device,
    )

    wrapper.eval()
    resolved_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_action_mask),
        str(resolved_onnx_path),
        export_params=True,
        dynamo=False,
        opset_version=int(opset_version),
        do_constant_folding=True,
        input_names=["obs", "action_mask"],
        output_names=["score_vector"],
        dynamic_axes={
            "obs": {0: "batch"},
            "action_mask": {0: "batch"},
            "score_vector": {0: "batch"},
        },
    )

    onnx_model = onnx.load(str(resolved_onnx_path))
    onnx.checker.check_model(onnx_model)

    metadata = build_ranker_onnx_export_metadata(
        agent,
        onnx_path=resolved_onnx_path,
        opset_version=opset_version,
    )
    resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_metadata_path, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    return {
        "onnx_path": str(resolved_onnx_path),
        "metadata_path": str(resolved_metadata_path),
        "metadata": metadata,
    }
