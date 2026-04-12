"""
Сценарий обучения LTE DQN на реальном runtime PyScheduler.

Скрипт использует `PySchedulerLteEnv`, то есть обучает агента на настоящей
динамике симулятора: UE collection, traffic generation, CQI refresh, PDCCH,
buffer updates и resource-grid механике.
"""

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from drl.paths import PYSCHEDULER_DQN_RUN_DIR
from drl.runtime_scenario_factory import (
    CURRICULUM_STAGES,
    EVAL_SCENARIO_KEYS,
    SCENARIO_CONFIGS,
    build_pyscheduler_env_pool,
    get_curriculum_candidates,
    make_pyscheduler_lte_env,
    sample_curriculum_scenario,
)
try:
    import torch
except ModuleNotFoundError:
    torch = None


DEFAULT_RUN_DIR = PYSCHEDULER_DQN_RUN_DIR
DEFAULT_MAX_N_UE = 40


def _format_traffic_profile(scenario: Any) -> str:
    traffic_params = dict(getattr(scenario, "traffic_params", {}) or {})

    if str(scenario.traffic_model_type) == "OnOff":
        duration_on = traffic_params.get("duration_on")
        duration_off = traffic_params.get("duration_off")
        return (
            f"OnOff(packet_rate={int(scenario.traffic_packet_rate)}, "
            f"duration_on={duration_on}, duration_off={duration_off})"
        )

    return f"{scenario.traffic_model_type}(packet_rate={int(scenario.traffic_packet_rate)})"


def evaluate_agent(
    agent: Any,
    env: Any,
    *,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    original_total_steps = int(agent.total_steps)

    obs, info = env.reset(seed=seed)
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

    agent.total_steps = original_total_steps

    summary = env.get_episode_summary()
    summary["invalid_action_rate"] = invalid_action_count / max(total_steps, 1)
    return summary


def evaluate_scenarios(
    agent: Any,
    env_pool: Dict[str, Any],
    *,
    seed_base: int = 100_000,
) -> list[Dict[str, float]]:
    rows: list[Dict[str, float]] = []

    for idx, scenario_key in enumerate(EVAL_SCENARIO_KEYS):
        env = env_pool[scenario_key]
        scenario = SCENARIO_CONFIGS[scenario_key]
        summary = evaluate_agent(agent, env, seed=seed_base + idx)
        summary["scenario"] = scenario_key
        summary["n_ue"] = int(scenario.n_ue)
        summary["bandwidth_mhz"] = float(scenario.bandwidth_mhz)
        summary["sim_duration_tti"] = int(scenario.sim_duration_tti)
        summary["n_rbg"] = int(env.n_rbg)
        summary["wb_cqi_report_period_tti"] = int(scenario.wb_cqi_report_period_tti)
        summary["traffic_model_type"] = str(scenario.traffic_model_type)
        summary["traffic_packet_rate"] = int(scenario.traffic_packet_rate)
        summary["traffic_profile"] = _format_traffic_profile(scenario)
        summary["channel_model_type"] = str(scenario.ch_model_type)
        summary["enable_tdl"] = bool(scenario.enable_tdl)
        summary["mobility_model"] = str(scenario.mobility_model)
        summary["mobility_update_interval_tti"] = int(scenario.mobility_update_interval_tti)
        summary["channel_update_interval_tti"] = int(scenario.channel_update_interval_tti)
        rows.append(summary)

    return rows


def save_metrics_csv(path: str | Path, rows: list[Dict[str, float]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение LTE DQN на simulation-backed env PyScheduler.",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max-n-ue", type=int, default=DEFAULT_MAX_N_UE)
    parser.add_argument("--total-env-steps", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=25_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-freq", type=int, default=10_000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--bootstrap-scenario", default=CURRICULUM_STAGES[0][1][0])
    return parser.parse_args()


def make_env(
    scenario: Any,
    *,
    max_n_ue: int,
    seed: Optional[int] = None,
) -> Any:
    return make_pyscheduler_lte_env(
        scenario=scenario,
        max_n_ue=max_n_ue,
        seed=seed,
    )


def main() -> None:
    args = parse_args()

    if torch is None:
        raise ModuleNotFoundError(
            "Для train_lte_dqn_pyscheduler.py требуется установленный torch."
        )

    from drl.agents.lte_dqn_agent import LTEDQNAgent, MaskedReplayBuffer

    run_dir = Path(args.run_dir)
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

    train_env_pool = build_pyscheduler_env_pool(
        all_scenario_keys,
        max_n_ue=int(args.max_n_ue),
        seed_base=int(args.seed),
    )
    eval_env_pool = build_pyscheduler_env_pool(
        EVAL_SCENARIO_KEYS,
        max_n_ue=int(args.max_n_ue),
        seed_base=int(args.seed) + 10_000,
    )

    bootstrap_key = str(args.bootstrap_scenario)
    if bootstrap_key not in train_env_pool:
        raise ValueError(f"Unknown bootstrap scenario: {bootstrap_key}")

    bootstrap_env = train_env_pool[bootstrap_key]
    agent = LTEDQNAgent(
        max_n_ue=bootstrap_env.max_n_ue,
        ue_feature_dim=bootstrap_env.ue_feature_dim,
        context_dim=bootstrap_env.context_dim,
        hidden_dim=int(args.hidden_dim),
        device=device,
        epsilon_start=1.0,
        epsilon_end=0.10,
        epsilon_decay_steps=max(int(args.total_env_steps), 1),
        gamma=float(args.gamma),
        lr=float(args.lr),
    )
    replay_buffer = MaskedReplayBuffer(capacity=250_000)

    rng = np.random.default_rng(int(args.seed))
    global_step = 0
    episode_idx = 0
    losses: list[float] = []
    episode_metrics: list[Dict[str, float]] = []

    current_scenario_key = bootstrap_key
    env = train_env_pool[current_scenario_key]
    obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))

    while global_step < int(args.total_env_steps):
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

        if len(replay_buffer) >= int(args.learning_starts) and tti_completed:
            for _ in range(int(args.gradient_steps)):
                batch = replay_buffer.sample(int(args.batch_size))
                losses.append(agent.train_step(batch))

        if global_step % int(args.target_update_freq) == 0:
            agent.update_target()

        if done:
            episode_idx += 1
            progress = global_step / max(int(args.total_env_steps), 1)
            stage_candidates = get_curriculum_candidates(progress)
            scenario = SCENARIO_CONFIGS[current_scenario_key]

            summary = env.get_episode_summary()
            summary["episode"] = episode_idx
            summary["epsilon"] = agent._get_epsilon()
            summary["global_step"] = global_step
            summary["curriculum_candidates"] = "|".join(stage_candidates)
            summary["train_scenario"] = current_scenario_key
            summary["train_n_ue"] = int(scenario.n_ue)
            summary["train_bandwidth_mhz"] = float(scenario.bandwidth_mhz)
            summary["train_sim_duration_tti"] = int(scenario.sim_duration_tti)
            summary["train_n_rbg"] = int(env.n_rbg)
            summary["train_wb_cqi_report_period_tti"] = int(
                scenario.wb_cqi_report_period_tti
            )
            summary["train_traffic_model_type"] = str(scenario.traffic_model_type)
            summary["train_traffic_packet_rate"] = int(scenario.traffic_packet_rate)
            summary["train_traffic_profile"] = _format_traffic_profile(scenario)
            summary["train_channel_model_type"] = str(scenario.ch_model_type)
            summary["train_enable_tdl"] = bool(scenario.enable_tdl)
            summary["train_mobility_model"] = str(scenario.mobility_model)
            summary["train_mobility_update_interval_tti"] = int(
                scenario.mobility_update_interval_tti
            )
            summary["train_channel_update_interval_tti"] = int(
                scenario.channel_update_interval_tti
            )
            summary["mean_loss_last_100"] = float(np.mean(losses[-100:])) if losses else 0.0
            episode_metrics.append(summary)

            if episode_idx % 5 == 0:
                print(
                    f"Episode {episode_idx:>3} | "
                    f"step {global_step:>7} | "
                    f"scenario: {current_scenario_key} | "
                    f"n_ue: {int(scenario.n_ue):>2} | "
                    f"bw: {float(scenario.bandwidth_mhz):>4.1f} MHz | "
                    f"ep_tti: {int(scenario.sim_duration_tti):>4} | "
                    f"wb: {int(scenario.wb_cqi_report_period_tti):>2} | "
                    f"ch: {str(scenario.ch_model_type):>3} | "
                    f"mob: {str(scenario.mobility_model):>14} | "
                    f"mob_upd: {int(scenario.mobility_update_interval_tti):>3} | "
                    f"ch_upd: {int(scenario.channel_update_interval_tti):>3} | "
                    f"traffic: {_format_traffic_profile(scenario)} | "
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

    eval_rows = evaluate_scenarios(
        agent,
        eval_env_pool,
        seed_base=int(args.seed) + 100_000,
    )
    eval_metrics_path = run_dir / "eval_metrics.csv"
    save_metrics_csv(eval_metrics_path, eval_rows)
    print(f"Eval metrics saved: {eval_metrics_path}")

    for row in eval_rows:
        print(
            f"\n=== Eval: {row['scenario']} "
            f"({row['n_ue']} UE, {row['bandwidth_mhz']} MHz, "
            f"n_rbg={row['n_rbg']}, wb_period={row['wb_cqi_report_period_tti']}) ==="
        )
        print(f"traffic_model_type: {row['traffic_model_type']}")
        print(f"traffic_packet_rate: {row['traffic_packet_rate']}")
        print(f"traffic_profile: {row['traffic_profile']}")
        print(f"channel_model_type: {row['channel_model_type']}")
        print(f"enable_tdl: {row['enable_tdl']}")
        print(f"mobility_model: {row['mobility_model']}")
        print(f"mobility_update_interval_tti: {row['mobility_update_interval_tti']}")
        print(f"channel_update_interval_tti: {row['channel_update_interval_tti']}")
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
