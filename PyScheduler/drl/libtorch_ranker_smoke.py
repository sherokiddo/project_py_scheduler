"""
Подготовка smoke-test артефактов для LibTorch inference ranker-модели.

Модуль формирует минимальный набор файлов:
- `obs.txt`
- `action_mask.txt`
- `expected_scores.txt`
- `smoke_manifest.json`

Эти файлы можно читать из C++ smoke-бинарника, чтобы проверить, что LibTorch
выдает те же score, что и Python-реализация `LTEPPORankerAgent`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drl.agents.lte_ppo_ranker_agent import LTEPPORankerAgent
from drl.runtime_scenario_factory import (
    SCENARIO_CONFIGS,
    make_pyscheduler_lte_ranker_env,
)


def _write_float_vector_txt(path: Path, values: np.ndarray) -> None:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"{len(flat)}\n")
        file_obj.write(" ".join(f"{float(value):.9f}" for value in flat))
        file_obj.write("\n")


def _write_int_vector_txt(path: Path, values: np.ndarray) -> None:
    flat = np.asarray(values).reshape(-1)
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"{len(flat)}\n")
        file_obj.write(" ".join(str(int(value)) for value in flat))
        file_obj.write("\n")


def prepare_ranker_libtorch_smoke_pack(
    *,
    checkpoint_path: str | Path,
    torchscript_path: str | Path,
    output_dir: str | Path,
    scenario_key: str = "anchor_5ue_10mhz_wb5_umi_fb",
    seed: int = 123,
    rank_weight_beta: float = 0.3,
    pf_epsilon_bps: float = 1e-6,
) -> dict[str, Any]:
    """
    Подготовить sample-pack для LibTorch smoke-test.

    Возвращает manifest и сохраняет текстовые входы/эталонные выходы на диск.
    """
    resolved_checkpoint_path = Path(checkpoint_path)
    resolved_torchscript_path = Path(torchscript_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    agent = LTEPPORankerAgent.load(
        path=str(resolved_checkpoint_path),
        device=torch.device("cpu"),
    )
    scenario = SCENARIO_CONFIGS[str(scenario_key)]
    env = make_pyscheduler_lte_ranker_env(
        scenario,
        max_n_ue=agent.max_n_ue,
        seed=int(seed),
        rank_weight_beta=float(rank_weight_beta),
        pf_epsilon_bps=float(pf_epsilon_bps),
    )

    try:
        observation, info = env.reset(seed=int(seed))
        action_mask = np.asarray(info["action_mask"], dtype=bool)
        expected_scores = agent.predict_scores(
            obs=observation,
            action_mask=action_mask,
            deterministic=True,
        )
    finally:
        env.close()

    obs_path = resolved_output_dir / "obs.txt"
    mask_path = resolved_output_dir / "action_mask.txt"
    expected_path = resolved_output_dir / "expected_scores.txt"
    manifest_path = resolved_output_dir / "smoke_manifest.json"

    _write_float_vector_txt(obs_path, observation)
    _write_int_vector_txt(mask_path, action_mask.astype(np.int32))
    _write_float_vector_txt(expected_path, expected_scores)

    manifest = {
        "checkpoint_path": str(resolved_checkpoint_path),
        "torchscript_path": str(resolved_torchscript_path),
        "scenario_key": str(scenario_key),
        "seed": int(seed),
        "obs_path": str(obs_path),
        "action_mask_path": str(mask_path),
        "expected_scores_path": str(expected_path),
        "obs_dim": int(observation.shape[0]),
        "max_n_ue": int(agent.max_n_ue),
        "ue_feature_dim": int(agent.ue_feature_dim),
        "context_dim": int(agent.context_dim),
        "masked_invalid_score": -1e9,
    }
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=False, indent=2)

    manifest["manifest_path"] = str(manifest_path)
    return manifest
