"""
Training script for the hybrid LTE PPO ranker on the real PyScheduler runtime.

The script uses `PySchedulerLteRankerEnv`, where the agent emits one score
vector per TTI and the environment completes allocation through the hybrid
metric:

    combined_metric(u, k) = rank_weight(u) * fd_pf_metric(u, k)

This keeps the PPO ranker focused on adaptive UE prioritization, while the
per-RBG realization stays aligned with the existing FD/PF baseline.

To make training feedback closer to real inference behavior, the script can
also run a periodic deterministic probe-eval during training.
"""

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from drl.paths import PYSCHEDULER_PPO_RANKER_RUN_DIR
from drl.runtime_scenario_factory import (
    CURRICULUM_STAGES,
    EVAL_SCENARIO_KEYS,
    SCENARIO_CONFIGS,
    build_pyscheduler_ranker_env_pool,
    get_curriculum_candidates,
    make_pyscheduler_lte_ranker_env,
    sample_curriculum_scenario,
)
from drl.scripts.train_lte_ppo_scheduler import (
    _format_traffic_profile,
    resolve_eval_device,
    resolve_eval_scenario_keys,
    save_metrics_csv,
)

try:
    import torch
except ModuleNotFoundError:
    torch = None


DEFAULT_RUN_DIR = PYSCHEDULER_PPO_RANKER_RUN_DIR
DEFAULT_MAX_N_UE = 40


def evaluate_agent(
    agent: Any,
    env: Any,
    *,
    seed: Optional[int] = None,
    deterministic: bool = True,
) -> Dict[str, float]:
    original_total_steps = int(agent.total_steps)

    obs, info = env.reset(seed=seed)
    done = False
    total_steps = 0
    invalid_action_count = 0

    while not done:
        action = agent.predict(
            obs,
            info["action_mask"],
            deterministic=bool(deterministic),
        )
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
    deterministic: bool = True,
    eval_mode: str = "deterministic",
    repeats: int = 1,
    verbose: bool = True,
) -> list[Dict[str, float]]:
    rows: list[Dict[str, float]] = []
    resolved_repeats = max(int(repeats), 1)

    total = len(scenario_keys)
    for idx, scenario_key in enumerate(scenario_keys, start=1):
        if verbose:
            print(
                f"[Eval {idx}/{total}] Starting scenario: {scenario_key} "
                f"(mode={eval_mode}, repeats={resolved_repeats})",
                flush=True,
            )

        env = env_pool[scenario_key]
        scenario = SCENARIO_CONFIGS[scenario_key]
        summaries = [
            evaluate_agent(
                agent,
                env,
                seed=seed_base + idx * 100 + repeat_idx,
                deterministic=deterministic,
            )
            for repeat_idx in range(resolved_repeats)
        ]
        summary: Dict[str, float] = {}
        metric_keys = summaries[0].keys()
        for key in metric_keys:
            values = [float(row[key]) for row in summaries]
            summary[key] = float(np.mean(values))

        summary["scenario"] = scenario_key
        summary["eval_mode"] = str(eval_mode)
        summary["eval_repeats"] = int(resolved_repeats)
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
            print(
                f"[Eval {idx}/{total}] Done: {scenario_key} "
                f"(mode={eval_mode}, "
                f"mean_reward={summary.get('mean_reward', 0.0):.4f}, "
                f"mean_se={summary.get('mean_se_bps_hz', 0.0):.4f}, "
                f"mean_jfi_all={summary.get('mean_jfi_all', 0.0):.4f})",
                flush=True,
            )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the hybrid LTE PPO ranker on simulation-backed PyScheduler env.",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--max-n-ue", type=int, default=DEFAULT_MAX_N_UE)
    parser.add_argument("--total-env-steps", type=int, default=300_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--rank-weight-beta",
        type=float,
        default=0.3,
        help="Strength of PPO influence in rank_weight(u) = 1 + beta * tanh(score).",
    )
    parser.add_argument(
        "--pf-epsilon-bps",
        type=float,
        default=1e-6,
        help="Small positive floor for the average throughput denominator in FD/PF metric.",
    )
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
        help="Limit the number of eval scenarios. 0 = skip evaluation.",
    )
    parser.add_argument(
        "--eval-compare-stochastic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compare deterministic and stochastic eval in one run. "
            "Useful for ranker experiments to detect fairness driven only by policy noise."
        ),
    )
    parser.add_argument(
        "--eval-stochastic-repeats",
        type=int,
        default=3,
        help="Number of repeats for stochastic eval averaging.",
    )
    parser.add_argument(
        "--probe-every-episodes",
        type=int,
        default=10,
        help=(
            "Run deterministic probe-eval every N finished episodes. "
            "0 disables probe-eval."
        ),
    )
    parser.add_argument(
        "--probe-scenario",
        default=None,
        help=(
            "Scenario key for periodic deterministic probe-eval. "
            "Defaults to --bootstrap-scenario."
        ),
    )
    return parser.parse_args()


def make_env(
    scenario: Any,
    *,
    max_n_ue: int,
    seed: Optional[int] = None,
    rank_weight_beta: float = 0.3,
    pf_epsilon_bps: float = 1e-6,
) -> Any:
    return make_pyscheduler_lte_ranker_env(
        scenario=scenario,
        max_n_ue=max_n_ue,
        seed=seed,
        rank_weight_beta=float(rank_weight_beta),
        pf_epsilon_bps=float(pf_epsilon_bps),
    )


def build_eval_agent(
    agent: Any,
    *,
    eval_device: torch.device,
) -> Any:
    from drl.agents.lte_ppo_ranker_agent import LTEPPORankerAgent

    eval_agent = LTEPPORankerAgent(
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
        min_log_std=agent.min_log_std,
        max_log_std=agent.max_log_std,
    )
    state_dict = {
        name: tensor.detach().to(eval_device).clone()
        for name, tensor in agent.policy.state_dict().items()
    }
    eval_agent.policy.load_state_dict(state_dict)
    eval_agent.policy.eval()
    eval_agent.total_steps = int(agent.total_steps)
    return eval_agent


def run_probe_evaluation(
    agent: Any,
    scenario_key: str,
    *,
    max_n_ue: int,
    seed: int,
    rank_weight_beta: float,
    pf_epsilon_bps: float,
    eval_device: torch.device,
) -> Dict[str, float]:
    scenario = SCENARIO_CONFIGS[scenario_key]
    probe_env = make_env(
        scenario,
        max_n_ue=max_n_ue,
        seed=seed,
        rank_weight_beta=rank_weight_beta,
        pf_epsilon_bps=pf_epsilon_bps,
    )
    try:
        eval_agent = build_eval_agent(agent, eval_device=eval_device)
        summary = evaluate_agent(
            eval_agent,
            probe_env,
            seed=seed,
            deterministic=True,
        )
    finally:
        probe_env.close()

    summary["probe_scenario"] = scenario_key
    summary["probe_seed"] = int(seed)
    return summary


def main() -> None:
    args = parse_args()

    if torch is None:
        raise ModuleNotFoundError(
            "train_lte_ppo_ranker.py requires torch to be installed."
        )

    from drl.agents.lte_ppo_ranker_agent import (
        LTEPPORankerAgent,
        UEPPOScoreRolloutBuffer,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Hybrid env config: rank_weight_beta={float(args.rank_weight_beta):.3f}, "
        f"pf_epsilon_bps={float(args.pf_epsilon_bps):.3e}"
    )
    eval_scenario_keys = resolve_eval_scenario_keys(args.eval_scenario_limit)
    eval_device = resolve_eval_device(str(args.eval_device), train_device=device)
    probe_every_episodes = max(int(args.probe_every_episodes), 0)
    print(f"Evaluation device: {eval_device}")
    if not eval_scenario_keys:
        print("Evaluation is disabled because --eval-scenario-limit=0")
    elif args.eval_compare_stochastic:
        print(
            f"Stochastic evaluation is enabled with {max(int(args.eval_stochastic_repeats), 1)} repeat(s)."
        )
    if probe_every_episodes > 0:
        print(
            f"Periodic deterministic probe-eval is enabled every {probe_every_episodes} episode(s)."
        )
    else:
        print("Periodic deterministic probe-eval is disabled.")

    train_scenario_keys = tuple(
        dict.fromkeys(
            scenario_key
            for _, stage_candidates in CURRICULUM_STAGES
            for scenario_key in stage_candidates
        )
    )
    all_scenario_keys = tuple(dict.fromkeys(train_scenario_keys + eval_scenario_keys))

    train_env_pool = build_pyscheduler_ranker_env_pool(
        all_scenario_keys,
        max_n_ue=int(args.max_n_ue),
        seed_base=int(args.seed),
        rank_weight_beta=float(args.rank_weight_beta),
        pf_epsilon_bps=float(args.pf_epsilon_bps),
    )
    eval_env_pool = build_pyscheduler_ranker_env_pool(
        eval_scenario_keys,
        max_n_ue=int(args.max_n_ue),
        seed_base=int(args.seed) + 10_000,
        rank_weight_beta=float(args.rank_weight_beta),
        pf_epsilon_bps=float(args.pf_epsilon_bps),
    )

    bootstrap_key = str(args.bootstrap_scenario)
    if bootstrap_key not in train_env_pool:
        raise ValueError(f"Unknown bootstrap scenario: {bootstrap_key}")
    probe_scenario_key = (
        str(args.probe_scenario) if args.probe_scenario is not None else bootstrap_key
    )
    if probe_scenario_key not in SCENARIO_CONFIGS:
        raise ValueError(f"Unknown probe scenario: {probe_scenario_key}")

    bootstrap_env = train_env_pool[bootstrap_key]
    agent = LTEPPORankerAgent(
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
    rollout_buffer = UEPPOScoreRolloutBuffer()

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
            last_value = 0.0 if done else agent.estimate_value(obs, info["action_mask"])
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
            summary["rank_weight_beta"] = float(args.rank_weight_beta)
            summary["pf_epsilon_bps"] = float(args.pf_epsilon_bps)
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

            summary["probe_scenario"] = probe_scenario_key
            summary["probe_mean_reward"] = np.nan
            summary["probe_mean_se_bps_hz"] = np.nan
            summary["probe_mean_jfi_all"] = np.nan
            summary["probe_mean_jfi_active"] = np.nan
            if probe_every_episodes > 0 and episode_idx % probe_every_episodes == 0:
                probe_summary = run_probe_evaluation(
                    agent,
                    probe_scenario_key,
                    max_n_ue=int(args.max_n_ue),
                    seed=int(args.seed) + 300_000 + episode_idx,
                    rank_weight_beta=float(args.rank_weight_beta),
                    pf_epsilon_bps=float(args.pf_epsilon_bps),
                    eval_device=eval_device,
                )
                summary["probe_mean_reward"] = float(
                    probe_summary.get("mean_reward", np.nan)
                )
                summary["probe_mean_se_bps_hz"] = float(
                    probe_summary.get("mean_se_bps_hz", np.nan)
                )
                summary["probe_mean_jfi_all"] = float(
                    probe_summary.get("mean_jfi_all", np.nan)
                )
                summary["probe_mean_jfi_active"] = float(
                    probe_summary.get("mean_jfi_active", np.nan)
                )
            episode_metrics.append(summary)

            if episode_idx % 5 == 0:
                probe_suffix = ""
                if np.isfinite(summary["probe_mean_reward"]):
                    probe_suffix = (
                        f" | probe[{probe_scenario_key}]: "
                        f"R={summary['probe_mean_reward']:.4f}, "
                        f"SE={summary['probe_mean_se_bps_hz']:.4f}, "
                        f"JFIall={summary['probe_mean_jfi_all']:.4f}"
                    )
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
                    f"beta: {float(args.rank_weight_beta):.2f} | "
                    f"traffic: {_format_traffic_profile(scenario)} | "
                    f"mean_reward: {summary.get('mean_reward', 0.0):.4f} | "
                    f"mean_se: {summary.get('mean_se_bps_hz', 0.0):.4f} | "
                    f"mean_jfi_act: {summary.get('mean_jfi_active', 0.0):.4f} | "
                    f"mean_jfi_all: {summary.get('mean_jfi_all', 0.0):.4f} | "
                    f"loss20: {summary['mean_loss_last_20']:.4f} | "
                    f"entropy20: {summary['mean_entropy_last_20']:.4f}"
                    f"{probe_suffix}"
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

    weights_path = run_dir / "lte_ppo_ranker_policy.pt"
    agent.save(weights_path)
    print(f"Weights saved: {weights_path}")

    run_config_path = run_dir / "run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            {
                "run_dir": str(run_dir),
                "max_n_ue": int(args.max_n_ue),
                "total_env_steps": int(args.total_env_steps),
                "rollout_steps": int(args.rollout_steps),
                "batch_size": int(args.batch_size),
                "ppo_epochs": int(args.ppo_epochs),
                "hidden_dim": int(args.hidden_dim),
                "gamma": float(args.gamma),
                "gae_lambda": float(args.gae_lambda),
                "clip_epsilon": float(args.clip_epsilon),
                "value_coef": float(args.value_coef),
                "entropy_coef": float(args.entropy_coef),
                "lr": float(args.lr),
                "rank_weight_beta": float(args.rank_weight_beta),
                "pf_epsilon_bps": float(args.pf_epsilon_bps),
                "seed": int(args.seed),
                "bootstrap_scenario": str(args.bootstrap_scenario),
                "eval_device": str(args.eval_device),
                "eval_scenario_limit": args.eval_scenario_limit,
                "eval_compare_stochastic": bool(args.eval_compare_stochastic),
                "eval_stochastic_repeats": int(args.eval_stochastic_repeats),
                "probe_every_episodes": int(args.probe_every_episodes),
                "probe_scenario": args.probe_scenario,
                "train_device_resolved": str(device),
                "eval_device_resolved": str(eval_device),
            },
            file_obj,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Run config saved: {run_config_path}")

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
            deterministic=True,
            eval_mode="deterministic",
            repeats=1,
            verbose=True,
        )
        if args.eval_compare_stochastic:
            eval_rows.extend(
                evaluate_scenarios(
                    eval_agent,
                    eval_env_pool,
                    scenario_keys=eval_scenario_keys,
                    seed_base=int(args.seed) + 200_000,
                    deterministic=False,
                    eval_mode="stochastic",
                    repeats=int(args.eval_stochastic_repeats),
                    verbose=True,
                )
            )
        eval_metrics_path = run_dir / "eval_metrics.csv"
        save_metrics_csv(eval_metrics_path, eval_rows)
        print(f"Eval metrics saved: {eval_metrics_path}")

        for row in eval_rows:
            print(
                f"\n=== Eval: {row['scenario']} [{row['eval_mode']}] "
                f"({row['n_ue']} UE, {row['bandwidth_mhz']} MHz, "
                f"n_rbg={row['n_rbg']}, wb_period={row['wb_cqi_report_period_tti']}) ==="
            )
            print(f"eval_repeats: {row['eval_repeats']}")
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
