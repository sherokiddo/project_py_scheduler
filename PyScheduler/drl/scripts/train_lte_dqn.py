"""
Сценарий обучения LTE DQN на перенесенной playground-среде.
"""

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import csv
import os
from pathlib import Path

import numpy as np
import torch

from drl.agents.lte_dqn_agent   import LTEDQNAgent, MaskedReplayBuffer
from drl.envs.lte_padded_env    import PaddedLTESchedulerEnv
from drl.envs.lte_scheduler_env import LTESchedulerEnv
from drl.paths                  import PLAYGROUND_DQN_RUN_DIR


MAX_N_UE = 40

SCENARIO_CONFIGS = {
    "train_3ue_10mhz_wb5":      {"n_ue": 3,  "n_rb_dl": 50, "wb_cqi_report_period_tti": 5},
    "mid_8ue_10mhz_wb5":        {"n_ue": 8,  "n_rb_dl": 50, "wb_cqi_report_period_tti": 5},
    "mid_16ue_10mhz_wb5":       {"n_ue": 16, "n_rb_dl": 50, "wb_cqi_report_period_tti": 5},
    "target_40ue_10mhz_wb5":    {"n_ue": 40, "n_rb_dl": 50, "wb_cqi_report_period_tti": 5},
    "bw_16ue_5mhz_wb5":         {"n_ue": 16, "n_rb_dl": 25, "wb_cqi_report_period_tti": 5},
    "bw_16ue_20mhz_wb5":        {"n_ue": 16, "n_rb_dl": 100, "wb_cqi_report_period_tti": 5},
    "cqi_16ue_10mhz_wb1":       {"n_ue": 16, "n_rb_dl": 50, "wb_cqi_report_period_tti": 1},
    "cqi_16ue_10mhz_wb10":      {"n_ue": 16, "n_rb_dl": 50, "wb_cqi_report_period_tti": 10},
}

CURRICULUM_STAGES = (
    (0.00, ("train_3ue_10mhz_wb5",)),
    (0.20, ("train_3ue_10mhz_wb5", "mid_8ue_10mhz_wb5")),
    (0.45, ("train_3ue_10mhz_wb5", "mid_8ue_10mhz_wb5", "mid_16ue_10mhz_wb5")),
    (
        0.70,
        (
            "train_3ue_10mhz_wb5",
            "mid_8ue_10mhz_wb5",
            "mid_16ue_10mhz_wb5",
            "target_40ue_10mhz_wb5",
            "bw_16ue_5mhz_wb5",
            "bw_16ue_20mhz_wb5",
            "cqi_16ue_10mhz_wb1",
            "cqi_16ue_10mhz_wb10",
        ),
    ),
)

EVAL_SCENARIO_KEYS = (
    "train_3ue_10mhz_wb5",
    "mid_8ue_10mhz_wb5",
    "mid_16ue_10mhz_wb5",
    "target_40ue_10mhz_wb5",
    "bw_16ue_5mhz_wb5",
    "bw_16ue_20mhz_wb5",
    "cqi_16ue_10mhz_wb1",
    "cqi_16ue_10mhz_wb10",
)


def get_default_traffic_profile(n_ue: int) -> str | tuple[str, ...]:
    if n_ue == 3:
        return ("full_buffer", "on_off", "bursty")
    return "on_off"


def format_traffic_profile(profile: str | tuple[str, ...]) -> str:
    if isinstance(profile, tuple):
        return "/".join(profile)
    return str(profile)


def make_base_env(
    scenario: dict[str, int],
    seed: int | None = None,
) -> LTESchedulerEnv:
    n_ue = int(scenario["n_ue"])
    traffic_profile = get_default_traffic_profile(n_ue)
    return LTESchedulerEnv(
        n_ue=n_ue,
        n_rb_dl=int(scenario["n_rb_dl"]),
        episode_len=300,
        reward_mode="per_tti",
        reward_window=1,
        alpha=1.0,
        beta=2.0,
        wb_cqi_report_period_tti=int(scenario["wb_cqi_report_period_tti"]),
        traffic_lambda=5000.0,
        traffic_profile=traffic_profile,
        rate_scale_bps=1e6,
        jfi_target=0.70,
        lambda_jfi=2.0,
        seed=seed,
    )


def make_env(
    scenario: dict[str, int],
    max_n_ue: int,
    seed: int | None = None,
) -> PaddedLTESchedulerEnv:
    return PaddedLTESchedulerEnv(
        make_base_env(scenario=scenario, seed=seed),
        max_n_ue=max_n_ue,
    )


def build_env_pool(
    scenario_keys: tuple[str, ...],
    max_n_ue: int,
    seed_base: int = 0,
) -> dict[str, PaddedLTESchedulerEnv]:
    env_pool: dict[str, PaddedLTESchedulerEnv] = {}
    for idx, scenario_key in enumerate(scenario_keys):
        env_pool[scenario_key] = make_env(
            scenario=SCENARIO_CONFIGS[scenario_key],
            max_n_ue=max_n_ue,
            seed=seed_base + idx,
        )
    return env_pool


def get_curriculum_candidates(progress: float) -> tuple[str, ...]:
    candidates = CURRICULUM_STAGES[0][1]
    for min_progress, stage_candidates in CURRICULUM_STAGES:
        if progress >= min_progress:
            candidates = stage_candidates
    return candidates


def sample_curriculum_scenario(
    rng: np.random.Generator,
    candidates: tuple[str, ...],
) -> str:
    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


def evaluate_agent(agent: LTEDQNAgent, env: PaddedLTESchedulerEnv) -> dict[str, float]:
    obs, info = env.reset()
    done = False
    total_steps = 0
    invalid_action_count = 0

    while not done:
        action = agent.predict(obs, info["action_mask"], deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        invalid_action_count += int(
            info.get("invalid_action", False) or info.get("padded_invalid_action", False)
        )
        total_steps += 1
        done = terminated or truncated

    summary = env.unwrapped.get_episode_summary()
    summary["invalid_action_rate"] = invalid_action_count / max(total_steps, 1)
    return summary


def evaluate_scenarios(
    agent: LTEDQNAgent,
    env_pool: dict[str, PaddedLTESchedulerEnv],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for scenario_key in EVAL_SCENARIO_KEYS:
        env = env_pool[scenario_key]
        scenario = SCENARIO_CONFIGS[scenario_key]
        summary = evaluate_agent(agent, env)
        traffic_profile = get_default_traffic_profile(int(scenario["n_ue"]))
        summary["scenario"] = scenario_key
        summary["n_ue"] = int(scenario["n_ue"])
        summary["n_rb_dl"] = int(scenario["n_rb_dl"])
        summary["n_rbg"] = int(env.unwrapped.n_rbg)
        summary["wb_cqi_report_period_tti"] = int(scenario["wb_cqi_report_period_tti"])
        summary["traffic_profile"] = format_traffic_profile(traffic_profile)
        rows.append(summary)
    return rows


def save_metrics_csv(path: str | Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    run_dir = PLAYGROUND_DQN_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_scenario_keys = tuple(
        dict.fromkeys(
            scenario_key
            for _, stage_candidates in CURRICULUM_STAGES
            for scenario_key in stage_candidates
        )
    )
    all_scenario_keys = tuple(dict.fromkeys(train_scenario_keys + EVAL_SCENARIO_KEYS))

    train_env_pool  = build_env_pool(all_scenario_keys, max_n_ue=MAX_N_UE, seed_base=0)
    eval_env_pool   = build_env_pool(EVAL_SCENARIO_KEYS, max_n_ue=MAX_N_UE, seed_base=10_000)

    bootstrap_key = CURRICULUM_STAGES[0][1][0]
    bootstrap_env = train_env_pool[bootstrap_key]

    agent = LTEDQNAgent(
        max_n_ue=bootstrap_env.max_n_ue,
        ue_feature_dim=bootstrap_env.ue_feature_dim,
        context_dim=bootstrap_env.context_dim,
        hidden_dim=64,
        device=device,
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay_steps=200_000,
        gamma=0.995,
        lr=3e-4,
    )
    replay_buffer = MaskedReplayBuffer(capacity=100_000)

    total_env_steps     = 200_000
    learning_starts     = 5_000
    batch_size          = 128
    gradient_steps      = 1
    target_update_freq  = 2_000

    rng = np.random.default_rng(12345)
    global_step = 0
    episode_idx = 0
    losses: list[float] = []
    episode_metrics: list[dict[str, float]] = []

    current_scenario_key = bootstrap_key
    env = train_env_pool[current_scenario_key]
    obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))

    while global_step < total_env_steps:
        action = agent.select_action(obs, info["action_mask"], deterministic=False)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        tti_completed = next_info["rbg_step"] == 0

        replay_buffer.push(
            obs,
            info["action_mask"],
            action,
            float(reward),
            next_obs,
            next_info["action_mask"],
            done,
        )

        obs = next_obs
        info = next_info
        global_step += 1

        if len(replay_buffer) >= learning_starts and tti_completed:
            for _ in range(gradient_steps):
                batch = replay_buffer.sample(batch_size)
                losses.append(agent.train_step(batch))

        if global_step % target_update_freq == 0:
            agent.update_target()

        if done:
            episode_idx += 1
            progress = global_step / max(total_env_steps, 1)
            stage_candidates = get_curriculum_candidates(progress)
            scenario = SCENARIO_CONFIGS[current_scenario_key]
            traffic_profile = get_default_traffic_profile(int(scenario["n_ue"]))

            summary = env.unwrapped.get_episode_summary()
            summary["episode"] = episode_idx
            summary["epsilon"] = agent._get_epsilon()
            summary["global_step"] = global_step
            summary["curriculum_candidates"] = "|".join(stage_candidates)
            summary["train_scenario"] = current_scenario_key
            summary["train_n_ue"] = int(scenario["n_ue"])
            summary["train_n_rb_dl"] = int(scenario["n_rb_dl"])
            summary["train_n_rbg"] = int(env.unwrapped.n_rbg)
            summary["train_wb_cqi_report_period_tti"] = int(
                scenario["wb_cqi_report_period_tti"]
            )
            summary["train_traffic_profile"] = format_traffic_profile(traffic_profile)
            summary["mean_loss_last_100"] = float(np.mean(losses[-100:])) if losses else 0.0
            episode_metrics.append(summary)

            if episode_idx % 5 == 0:
                print(
                    f"Episode {episode_idx:>3} | "
                    f"step {global_step:>7} | "
                    f"scenario: {current_scenario_key} | "
                    f"n_ue: {int(scenario['n_ue']):>2} | "
                    f"n_rb_dl: {int(scenario['n_rb_dl']):>3} | "
                    f"wb: {int(scenario['wb_cqi_report_period_tti']):>2} | "
                    f"mean_reward: {summary.get('mean_reward', 0.0):.4f} | "
                    f"mean_se: {summary.get('mean_se_bps_hz', 0.0):.4f} | "
                    f"mean_jfi_act: {summary.get('mean_jfi_active', 0.0):.4f} | "
                    f"mean_jfi_all: {summary.get('mean_jfi_all', 0.0):.4f} | "
                    f"eps: {summary['epsilon']:.3f}"
                )

            current_scenario_key = sample_curriculum_scenario(rng, stage_candidates)
            env = train_env_pool[current_scenario_key]
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))

    weights_path = run_dir / "lte_dqn_shared_q.pt"
    agent.save(weights_path)
    print(f"Weights saved: {weights_path}")

    metrics_path = run_dir / "train_metrics.csv"
    save_metrics_csv(metrics_path, episode_metrics)
    print(f"Metrics saved: {metrics_path}")

    eval_rows = evaluate_scenarios(agent, eval_env_pool)
    eval_metrics_path = run_dir / "eval_metrics.csv"
    save_metrics_csv(eval_metrics_path, eval_rows)
    print(f"Eval metrics saved: {eval_metrics_path}")

    for row in eval_rows:
        print(
            f"\n=== Eval: {row['scenario']} "
            f"({row['n_ue']} UE, n_rb_dl={row['n_rb_dl']}, "
            f"n_rbg={row['n_rbg']}, wb_period={row['wb_cqi_report_period_tti']}) ==="
        )
        print(f"traffic_profile: {row['traffic_profile']}")
        print(f"mean_throughput_mbps: {row['mean_throughput_mbps']:.4f}")
        print(f"mean_se_bps_hz: {row['mean_se_bps_hz']:.4f}")
        print(f"mean_jfi_active: {row['mean_jfi_active']:.4f}")
        print(f"mean_jfi_all: {row['mean_jfi_all']:.4f}")
        print(f"mean_reward: {row['mean_reward']:.4f}")
        print(f"min_jfi_active: {row['min_jfi_active']:.4f}")
        print(f"min_jfi_all: {row['min_jfi_all']:.4f}")
        print(f"max_se_bps_hz: {row['max_se_bps_hz']:.4f}")
        print(f"invalid_action_rate: {row['invalid_action_rate']:.4f}")

    for env_obj in set(train_env_pool.values()) | set(eval_env_pool.values()):
        env_obj.close()


if __name__ == "__main__":
    main()
