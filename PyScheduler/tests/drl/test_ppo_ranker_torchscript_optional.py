import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


pytest.importorskip("gymnasium")
torch = pytest.importorskip("torch")


from drl.agents.lte_ppo_ranker_agent import LTEPPORankerAgent, UEPPOScoreRolloutBuffer
from drl.scripts.train_lte_ppo_ranker import SCENARIO_CONFIGS, make_env
from drl.torchscript_ranker_export import (
    MASKED_INVALID_SCORE,
    build_ranker_export_metadata,
    build_torchscript_ranker_inference,
    export_ranker_to_torchscript,
)


def _workspace_tmp_dir() -> Path:
    base_dir = Path("d:/project_py_scheduler/.tmp_test_artifacts")
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / f"ppo_ranker_torchscript_{uuid4().hex}"
    run_dir.mkdir()
    return run_dir


def _build_smoke_agent() -> tuple[object, LTEPPORankerAgent]:
    scenario = replace(
        SCENARIO_CONFIGS["anchor_5ue_10mhz_wb5_umi_fb"],
        sim_duration_tti=6,
    )
    env = make_env(scenario, max_n_ue=8, seed=123)
    agent = LTEPPORankerAgent(
        max_n_ue=env.max_n_ue,
        ue_feature_dim=env.ue_feature_dim,
        context_dim=env.context_dim,
        hidden_dim=32,
        batch_size=8,
        n_epochs=2,
    )
    rollout_buffer = UEPPOScoreRolloutBuffer()

    obs, info = env.reset(seed=123)
    last_done = False

    for step_idx in range(24):
        action, log_prob, value = agent.select_action(
            obs,
            info["action_mask"],
            deterministic=False,
        )
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        last_done = bool(terminated or truncated)
        rollout_buffer.push(
            state=obs,
            action_mask=info["action_mask"],
            action=action,
            reward=float(reward),
            done=last_done,
            log_prob=log_prob,
            value=value,
        )

        obs, info = next_obs, next_info
        if last_done:
            obs, info = env.reset(seed=124 + step_idx)

    if last_done:
        last_value = 0.0
    else:
        last_value = agent.estimate_value(obs, info["action_mask"])

    agent.update(rollout_buffer, last_value=last_value)
    return env, agent


def test_ppo_ranker_torchscript_wrapper_matches_python_agent_on_real_observations():
    env, agent = _build_smoke_agent()
    wrapper = build_torchscript_ranker_inference(agent, device=torch.device("cpu"))
    wrapper.eval()

    obs, info = env.reset(seed=321)
    observations = [obs.copy()]
    masks = [np.asarray(info["action_mask"], dtype=bool).copy()]

    for step_idx in range(3):
        action, _, _ = agent.select_action(
            obs,
            info["action_mask"],
            deterministic=False,
        )
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        observations.append(obs.copy())
        masks.append(np.asarray(info["action_mask"], dtype=bool).copy())

    observations_np = np.asarray(observations, dtype=np.float32)
    masks_np = np.asarray(masks, dtype=bool)

    python_scores = np.stack(
        [
            agent.predict_scores(
                obs=row_obs,
                action_mask=row_mask,
                deterministic=True,
            )
            for row_obs, row_mask in zip(observations_np, masks_np, strict=True)
        ],
        axis=0,
    )

    with torch.no_grad():
        scripted_scores = (
            wrapper(
                torch.as_tensor(observations_np, dtype=torch.float32),
                torch.as_tensor(masks_np, dtype=torch.bool),
            )
            .cpu()
            .numpy()
        )

    assert python_scores.shape == scripted_scores.shape
    assert np.allclose(python_scores, scripted_scores, atol=1e-5, rtol=1e-5)

    all_invalid_mask = np.zeros_like(masks_np[:1], dtype=bool)
    with torch.no_grad():
        all_invalid_scores = (
            wrapper(
                torch.as_tensor(observations_np[:1], dtype=torch.float32),
                torch.as_tensor(all_invalid_mask, dtype=torch.bool),
            )
            .cpu()
            .numpy()
        )
    assert np.all(all_invalid_scores == pytest.approx(MASKED_INVALID_SCORE))

    env.close()


def test_ppo_ranker_torchscript_export_saves_artifact_and_metadata():
    env, agent = _build_smoke_agent()
    workdir = _workspace_tmp_dir()
    checkpoint_path = workdir / "lte_ppo_ranker_policy.pt"
    torchscript_path = workdir / "lte_ppo_ranker_policy.ts"
    metadata_path = workdir / "lte_ppo_ranker_policy.meta.json"

    try:
        agent.save(str(checkpoint_path))
        result = export_ranker_to_torchscript(
            checkpoint_path=checkpoint_path,
            torchscript_path=torchscript_path,
            metadata_path=metadata_path,
            device="cpu",
        )

        assert torchscript_path.exists()
        assert metadata_path.exists()
        assert result["torchscript_path"] == str(torchscript_path)
        assert result["metadata_path"] == str(metadata_path)

        loaded_module = torch.jit.load(str(torchscript_path), map_location="cpu")

        obs, info = env.reset(seed=456)
        obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
        mask_tensor = torch.as_tensor(
            np.asarray(info["action_mask"], dtype=bool)[None, :],
            dtype=torch.bool,
        )

        python_scores = agent.predict_scores(
            obs=obs,
            action_mask=np.asarray(info["action_mask"], dtype=bool),
            deterministic=True,
        )
        with torch.no_grad():
            ts_scores = loaded_module(obs_tensor, mask_tensor).cpu().numpy()[0]

        assert np.allclose(python_scores, ts_scores, atol=1e-5, rtol=1e-5)

        with open(metadata_path, "r", encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)

        expected_metadata = build_ranker_export_metadata(
            agent,
            torchscript_path=torchscript_path,
        )
        assert metadata["format"] == "torchscript"
        assert metadata["model_type"] == "LTEPPORankerAgent"
        assert metadata["max_n_ue"] == expected_metadata["max_n_ue"]
        assert metadata["ue_feature_dim"] == expected_metadata["ue_feature_dim"]
        assert metadata["context_dim"] == expected_metadata["context_dim"]
        assert metadata["obs_dim"] == expected_metadata["obs_dim"]
        assert metadata["runtime_actor_only"] is True
        assert metadata["masked_invalid_score"] == expected_metadata["masked_invalid_score"]
        assert metadata["input_names"] == ["obs", "action_mask"]
        assert metadata["output_names"] == ["score_vector"]
    finally:
        env.close()
        shutil.rmtree(workdir, ignore_errors=True)
