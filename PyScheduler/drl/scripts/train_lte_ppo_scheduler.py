"""
Сценарий обучения LTE PPO на реальном runtime PyScheduler.

Скрипт использует ту же `PySchedulerLteEnv`, что и runtime DQN-path, но вместо
Q-learning обучает masked actor-critic policy по PPO. На этом этапе PPO учится
только в env и еще не подключается как runtime scheduler внутри симуляции.
"""

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import csv
from pathlib import Path
import time
from typing import Any, Dict, Optional

import numpy as np

from drl.paths import PYSCHEDULER_PPO_RUN_DIR
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


DEFAULT_RUN_DIR = PYSCHEDULER_PPO_RUN_DIR
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
    scenario_keys: tuple[str, ...],
    seed_base: int = 100_000,
    verbose: bool = True,
) -> list[Dict[str, float]]:
    rows: list[Dict[str, float]] = []

    total = len(scenario_keys)
    for idx, scenario_key in enumerate(scenario_keys, start=1):
        started_at = time.perf_counter()
        if verbose:
            print(f"[Eval {idx}/{total}] Starting scenario: {scenario_key}", flush=True)

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

        if verbose:
            elapsed = time.perf_counter() - started_at
            print(
                f"[Eval {idx}/{total}] Done: {scenario_key} "
                f"(mean_reward={summary.get('mean_reward', 0.0):.4f}, "
                f"mean_se={summary.get('mean_se_bps_hz', 0.0):.4f}, "
                f"mean_jfi_all={summary.get('mean_jfi_all', 0.0):.4f}, "
                f"{elapsed:.1f}s)",
                flush=True,
            )

    return rows


def save_metrics_csv(path: str | Path, rows: list[Dict[str, float]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение LTE PPO на simulation-backed env PyScheduler.",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max-n-ue", type=int, default=DEFAULT_MAX_N_UE)
    parser.add_argument("--total-env-steps", type=int, default=750_000)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--bootstrap-scenario", default=CURRICULUM_STAGES[0][1][0])
    parser.add_argument(
        "--eval-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )
    parser.add_argument(
        "--eval-scenario-limit",
        type=int,
        default=None,
        help="Ограничить число eval-сценариев. 0 = пропустить eval.",
    )
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


def resolve_eval_scenario_keys(limit: Optional[int]) -> tuple[str, ...]:
    if limit is None:
        return EVAL_SCENARIO_KEYS

    resolved_limit = max(int(limit), 0)
    if resolved_limit == 0:
        return tuple()
    return EVAL_SCENARIO_KEYS[:resolved_limit]


def resolve_eval_device(
    device_name: str,
    *,
    train_device: torch.device,
) -> torch.device:
    if device_name == "auto":
        if train_device.type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            print("Requested eval device cuda, but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")

    return torch.device("cpu")


def build_eval_agent(
    agent: Any,
    *,
    eval_device: torch.device,
) -> Any:
    from drl.agents.lte_ppo_agent import LTEPPOAgent

    eval_agent = LTEPPOAgent(
        max_n_ue=agent.max_n_ue,
        ue_feature_dim=agent.ue_feature_dim,
        context_dim=agent.context_dim,
        hidden_dim=agent.hidden_dim,
        device=eval_device,
        lr=agent.lr,
        gamma=agent.gamma,
        gae_lambda=agent.gae_lambda,
        clip_epsilon=agent.clip_epsilon,
        value_coef=agent.value_coef,
        entropy_coef=agent.entropy_coef,
        n_epochs=agent.n_epochs,
        batch_size=agent.batch_size,
    )
    state_dict = {
        name: tensor.detach().to(eval_device).clone()
        for name, tensor in agent.policy.state_dict().items()
    }
    eval_agent.policy.load_state_dict(state_dict)
    eval_agent.policy.eval()
    eval_agent.total_steps = int(agent.total_steps)
    return eval_agent


def main() -> None:
    args = parse_args()

    if torch is None:
        raise ModuleNotFoundError(
            "Для train_lte_ppo_scheduler.py требуется установленный torch."
        )

    from drl.agents.lte_ppo_agent import LTEPPOAgent, MaskedPPORolloutBuffer

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    eval_scenario_keys = resolve_eval_scenario_keys(args.eval_scenario_limit)
    eval_device = resolve_eval_device(
        str(args.eval_device),
        train_device=device,
    )
    print(f"Evaluation device: {eval_device}")
    if not eval_scenario_keys:
        print("Evaluation is disabled because --eval-scenario-limit=0")

    train_scenario_keys = tuple(
        dict.fromkeys(
            scenario_key
            for _, stage_candidates in CURRICULUM_STAGES
            for scenario_key in stage_candidates
        )
    )
    all_scenario_keys = tuple(dict.fromkeys(train_scenario_keys + eval_scenario_keys))

    train_env_pool = build_pyscheduler_env_pool(
        all_scenario_keys,
        max_n_ue=int(args.max_n_ue),
        seed_base=int(args.seed),
    )
    eval_env_pool = build_pyscheduler_env_pool(
        eval_scenario_keys,
        max_n_ue=int(args.max_n_ue),
        seed_base=int(args.seed) + 10_000,
    )

    bootstrap_key = str(args.bootstrap_scenario)
    if bootstrap_key not in train_env_pool:
        raise ValueError(f"Unknown bootstrap scenario: {bootstrap_key}")

    bootstrap_env = train_env_pool[bootstrap_key]
    agent = LTEPPOAgent(
        max_n_ue=bootstrap_env.max_n_ue,
        ue_feature_dim=bootstrap_env.ue_feature_dim,
        context_dim=bootstrap_env.context_dim,
        hidden_dim=int(args.hidden_dim),
        device=device,
        lr=float(args.lr),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_epsilon=float(args.clip_epsilon),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        n_epochs=int(args.ppo_epochs),
        batch_size=int(args.batch_size),
    )
    rollout_buffer = MaskedPPORolloutBuffer()

    rng = np.random.default_rng(int(args.seed))
    global_step = 0
    episode_idx = 0
    update_idx = 0
    update_metrics_history: list[Dict[str, float]] = []
    episode_metrics: list[Dict[str, float]] = []

    current_scenario_key = bootstrap_key
    env = train_env_pool[current_scenario_key]
    obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))

    while global_step < int(args.total_env_steps):
        action, log_prob, value = agent.select_action(
            obs,
            info["action_mask"],
            deterministic=False,
        )
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = bool(terminated or truncated)

        rollout_buffer.push(
            state=obs,
            action_mask=info["action_mask"],
            action=action,
            reward=float(reward),
            done=done,
            log_prob=log_prob,
            value=value,
        )

        obs = next_obs
        info = next_info
        global_step += 1

        if len(rollout_buffer) >= int(args.rollout_steps):
            if done:
                last_value = 0.0
            else:
                last_value = agent.estimate_value(obs, info["action_mask"])

            update_metrics = agent.update(rollout_buffer, last_value=last_value)
            rollout_buffer.clear()
            update_idx += 1
            update_metrics["update_idx"] = float(update_idx)
            update_metrics["global_step"] = float(global_step)
            update_metrics_history.append(update_metrics)

        if done:
            episode_idx += 1
            progress = global_step / max(int(args.total_env_steps), 1)
            stage_candidates = get_curriculum_candidates(progress)
            scenario = SCENARIO_CONFIGS[current_scenario_key]

            summary = env.get_episode_summary()
            summary["episode"] = episode_idx
            summary["global_step"] = global_step
            summary["policy_update_idx"] = update_idx
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
            if update_metrics_history:
                summary["mean_loss_last_20"] = float(
                    np.mean([row["loss"] for row in update_metrics_history[-20:]])
                )
                summary["mean_entropy_last_20"] = float(
                    np.mean([row["entropy"] for row in update_metrics_history[-20:]])
                )
            else:
                summary["mean_loss_last_20"] = 0.0
                summary["mean_entropy_last_20"] = 0.0
            episode_metrics.append(summary)

            if episode_idx % 5 == 0:
                print(
                    f"Episode {episode_idx:>3} | "
                    f"step {global_step:>7} | "
                    f"updates: {update_idx:>4} | "
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
                    f"loss20: {summary['mean_loss_last_20']:.4f} | "
                    f"entropy20: {summary['mean_entropy_last_20']:.4f}"
                )

            current_scenario_key = sample_curriculum_scenario(rng, stage_candidates)
            env = train_env_pool[current_scenario_key]
            obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))

    if len(rollout_buffer) > 0:
        last_value = 0.0 if info.get("terminated", False) else agent.estimate_value(
            obs,
            info["action_mask"],
        )
        update_metrics = agent.update(rollout_buffer, last_value=last_value)
        rollout_buffer.clear()
        update_idx += 1
        update_metrics["update_idx"] = float(update_idx)
        update_metrics["global_step"] = float(global_step)
        update_metrics_history.append(update_metrics)

    weights_path = run_dir / "lte_ppo_policy.pt"
    agent.save(weights_path)
    print(f"Weights saved: {weights_path}")

    train_metrics_path = run_dir / "train_metrics.csv"
    save_metrics_csv(train_metrics_path, episode_metrics)
    print(f"Train metrics saved: {train_metrics_path}")

    ppo_metrics_path = run_dir / "ppo_update_metrics.csv"
    save_metrics_csv(ppo_metrics_path, update_metrics_history)
    print(f"PPO update metrics saved: {ppo_metrics_path}")

    if eval_scenario_keys:
        print(
            f"Starting evaluation on {len(eval_scenario_keys)} scenario(s) using device {eval_device}...",
            flush=True,
        )
        eval_agent = build_eval_agent(agent, eval_device=eval_device)
        eval_rows = evaluate_scenarios(
            eval_agent,
            eval_env_pool,
            scenario_keys=eval_scenario_keys,
            seed_base=int(args.seed) + 100_000,
            verbose=True,
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
    else:
        print("Evaluation skipped.")

    for env_obj in set(train_env_pool.values()) | set(eval_env_pool.values()):
        env_obj.close()


if __name__ == "__main__":
    main()
