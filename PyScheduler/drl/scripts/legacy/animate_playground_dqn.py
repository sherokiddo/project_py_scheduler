"""
Визуализация работы обученного LTE DQN-агента на playground-среде.
"""

import matplotlib.pyplot as plt
import numpy as np

from drl.agents.lte_dqn_agent import LTEDQNAgent
from drl.envs.lte_padded_env import PaddedLTESchedulerEnv
from drl.envs.lte_scheduler_env import LTESchedulerEnv
from drl.paths import PLAYGROUND_DQN_MODEL_PATH


MODEL_PATH = PLAYGROUND_DQN_MODEL_PATH
MAX_N_UE = 40
EVAL_N_UE = 3


def make_env():
    return PaddedLTESchedulerEnv(
        LTESchedulerEnv(
            n_ue=EVAL_N_UE,
            n_rb_dl=50,
            episode_len=500,
            reward_mode="per_tti",
            reward_window=1,
            alpha=1.0,
            beta=2.0,
            wb_cqi_report_period_tti=5,
            traffic_lambda=5000.0,
            traffic_profile=("full_buffer", "on_off", "bursty"),
            rate_scale_bps=1e6,
            jfi_target=0.70,
            lambda_jfi=2.0,
            seed=123,
        ),
        max_n_ue=MAX_N_UE,
    )


def _init_live_metrics(n_ue: int) -> dict[str, list | np.ndarray | int]:
    return {
        "tti_index": [],
        "se": [],
        "reward": [],
        "jfi_active": [],
        "jfi_all": [],
        "active_ue_count": [],
        "served_ue_count": [],
        "dominant_share": [],
        "full_grant_single_ue_count": 0,
        "invalid_action_count": 0,
        "total_steps": 0,
        "service_gap_tti": np.zeros(n_ue, dtype=np.int32),
        "max_service_gap_history": [],
    }


def _get_last_alloc_counts(base_env: LTESchedulerEnv) -> np.ndarray:
    if base_env._last_rbg_alloc is None:
        return np.zeros(base_env.n_ue, dtype=np.int32)

    counts = np.zeros(base_env.n_ue, dtype=np.int32)
    for ue in range(base_env.n_ue):
        counts[ue] = int(np.sum(base_env._last_rbg_alloc == ue))
    return counts


def _get_last_alloc_frac(base_env: LTESchedulerEnv) -> np.ndarray:
    alloc_counts = _get_last_alloc_counts(base_env).astype(np.float64)
    return alloc_counts / max(base_env.n_rbg, 1)


def _update_live_metrics(
    live_metrics: dict[str, list | np.ndarray | int],
    base_env: LTESchedulerEnv,
    info: dict,
) -> None:
    alloc_counts = _get_last_alloc_counts(base_env)
    alloc_frac = _get_last_alloc_frac(base_env)
    served_mask = alloc_counts > 0
    active_mask = np.asarray(info["tti_active_flag"], dtype=bool)

    service_gap_tti = np.asarray(live_metrics["service_gap_tti"], dtype=np.int32)
    service_gap_tti[served_mask] = 0
    service_gap_tti[~served_mask] += 1
    live_metrics["service_gap_tti"] = service_gap_tti

    live_metrics["tti_index"].append(int(info["tti"] - 1))
    live_metrics["se"].append(float(base_env.history["se"][-1]) if base_env.history["se"] else 0.0)
    live_metrics["reward"].append(
        float(base_env.history["reward"][-1]) if base_env.history["reward"] else 0.0
    )
    live_metrics["jfi_active"].append(
        float(base_env.history["jfi"][-1]) if base_env.history["jfi"] else 0.0
    )
    live_metrics["jfi_all"].append(
        float(base_env.history["jfi_all"][-1]) if base_env.history["jfi_all"] else 0.0
    )
    live_metrics["active_ue_count"].append(int(np.sum(active_mask)))
    live_metrics["served_ue_count"].append(int(np.sum(served_mask)))
    live_metrics["dominant_share"].append(float(np.max(alloc_frac)) if len(alloc_frac) > 0 else 0.0)
    live_metrics["max_service_gap_history"].append(int(np.max(service_gap_tti)))

    if np.max(alloc_counts) == base_env.n_rbg:
        live_metrics["full_grant_single_ue_count"] = int(
            live_metrics["full_grant_single_ue_count"]
        ) + 1


def _draw_status_panel(ax_status, base_env: LTESchedulerEnv, info: dict, live_metrics: dict) -> None:
    ax_status.clear()
    ax_status.axis("off")

    total_steps = max(int(live_metrics["total_steps"]), 1)
    invalid_action_rate = int(live_metrics["invalid_action_count"]) / total_steps
    full_grant_rate = (
        int(live_metrics["full_grant_single_ue_count"]) / max(len(live_metrics["tti_index"]), 1)
    )
    dominant_share = live_metrics["dominant_share"][-1] if live_metrics["dominant_share"] else 0.0
    active_ue_count = live_metrics["active_ue_count"][-1] if live_metrics["active_ue_count"] else 0
    served_ue_count = live_metrics["served_ue_count"][-1] if live_metrics["served_ue_count"] else 0
    current_reward = base_env.history["reward"][-1] if base_env.history["reward"] else 0.0
    current_se = base_env.history["se"][-1] if base_env.history["se"] else 0.0
    current_jfi_active = base_env.history["jfi"][-1] if base_env.history["jfi"] else 0.0
    current_jfi_all = base_env.history["jfi_all"][-1] if base_env.history["jfi_all"] else 0.0

    status_lines = [
        f"TTI: {int(info['tti'] - 1)}    RBG/TTI: {base_env.n_rbg}    N_UE: {base_env.n_ue}",
        f"SE: {current_se:.3f}    Reward: {current_reward:.3f}",
        f"JFI(act): {current_jfi_active:.3f}    JFI(all): {current_jfi_all:.3f}",
        f"Active UE: {active_ue_count}    Served UE: {served_ue_count}",
        f"Invalid action rate: {invalid_action_rate:.4f}",
        f"Full-grant single UE rate: {full_grant_rate:.3f}",
        f"Dominant share this TTI: {dominant_share:.3f}",
        "",
        f"True CQI: {np.asarray(info['true_wb_cqi'], dtype=int).tolist()}",
        f"Rep CQI:  {np.asarray(info['reported_wb_cqi'], dtype=int).tolist()}",
        f"CQI age:  {np.asarray(info['wb_cqi_age'], dtype=int).tolist()}",
        f"TTI active: {np.asarray(info['tti_active_flag'], dtype=int).tolist()}",
        f"Action mask: {np.asarray(info['action_mask'][:base_env.n_ue], dtype=int).tolist()}",
    ]

    ax_status.text(
        0.01,
        0.98,
        "\n".join(status_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )
    ax_status.set_title("Live KPIs and CQI", loc="left")


def _draw_curves_panel(ax_curves, live_metrics: dict, curve_window_tti: int) -> None:
    ax_curves.clear()

    tti_index = np.asarray(live_metrics["tti_index"], dtype=np.int32)
    if len(tti_index) == 0:
        ax_curves.set_title("Rolling SE / JFI")
        return

    start = max(0, len(tti_index) - curve_window_tti)
    x = tti_index[start:]
    se = np.asarray(live_metrics["se"][start:], dtype=np.float64)
    jfi_active = np.asarray(live_metrics["jfi_active"][start:], dtype=np.float64)
    jfi_all = np.asarray(live_metrics["jfi_all"][start:], dtype=np.float64)

    ax_curves.plot(x, se, label="SE", linewidth=2.0, color="tab:blue")
    ax_curves.plot(x, jfi_active, label="JFI(act)", linewidth=2.0, color="tab:green")
    ax_curves.plot(x, jfi_all, label="JFI(all)", linewidth=2.0, color="tab:orange")
    ax_curves.set_title(f"Rolling SE / JFI (last {curve_window_tti} TTI)")
    ax_curves.set_xlabel("TTI")
    ax_curves.set_ylim(bottom=0.0)
    ax_curves.grid(alpha=0.3)
    ax_curves.legend(loc="best", fontsize=8)


def _draw_tput_panel(ax_tput, base_env: LTESchedulerEnv) -> None:
    ax_tput.clear()
    ue_idx = np.arange(base_env.n_ue)
    avg_tput_mbps = np.asarray(base_env._avg_tput, dtype=np.float64) / 1e6

    ax_tput.bar(ue_idx, avg_tput_mbps, color=plt.get_cmap("tab10").colors[: base_env.n_ue])
    ax_tput.set_title("Per-UE Avg Throughput")
    ax_tput.set_xlabel("UE")
    ax_tput.set_ylabel("Mbps")
    ax_tput.set_xticks(ue_idx)
    ax_tput.grid(axis="y", alpha=0.3)


def _draw_alloc_panel(ax_alloc, info: dict, base_env: LTESchedulerEnv) -> None:
    ax_alloc.clear()
    ue_idx = np.arange(base_env.n_ue)
    alloc_counts = _get_last_alloc_counts(base_env)
    active_mask = np.asarray(info["tti_active_flag"], dtype=bool)
    colors = ["tab:red" if active_mask[i] else "tab:gray" for i in range(base_env.n_ue)]

    ax_alloc.bar(ue_idx, alloc_counts, color=colors)
    ax_alloc.set_title("Last TTI RBG Allocation")
    ax_alloc.set_xlabel("UE")
    ax_alloc.set_ylabel("RBG")
    ax_alloc.set_xticks(ue_idx)
    ax_alloc.set_ylim(0, max(base_env.n_rbg, 1))
    ax_alloc.grid(axis="y", alpha=0.3)


def _print_final_summary(base_env: LTESchedulerEnv, live_metrics: dict) -> None:
    summary = base_env.get_episode_summary()
    invalid_action_rate = int(live_metrics["invalid_action_count"]) / max(
        int(live_metrics["total_steps"]), 1
    )
    full_grant_rate = int(live_metrics["full_grant_single_ue_count"]) / max(
        len(live_metrics["tti_index"]), 1
    )
    mean_active_ue_count = (
        float(np.mean(live_metrics["active_ue_count"])) if live_metrics["active_ue_count"] else 0.0
    )
    mean_served_ue_count = (
        float(np.mean(live_metrics["served_ue_count"])) if live_metrics["served_ue_count"] else 0.0
    )
    mean_dominant_share = (
        float(np.mean(live_metrics["dominant_share"])) if live_metrics["dominant_share"] else 0.0
    )
    p95_service_gap = (
        float(np.percentile(live_metrics["max_service_gap_history"], 95))
        if live_metrics["max_service_gap_history"]
        else 0.0
    )

    print("\n=== Animate Summary ===")
    for key in (
        "mean_throughput_mbps",
        "mean_se_bps_hz",
        "mean_jfi_active",
        "mean_jfi_all",
        "mean_reward",
        "min_jfi_active",
        "min_jfi_all",
        "max_se_bps_hz",
    ):
        if key in summary:
            print(f"{key}: {summary[key]:.4f}")
    print(f"mean_active_ue_count: {mean_active_ue_count:.4f}")
    print(f"mean_served_ue_count: {mean_served_ue_count:.4f}")
    print(f"mean_dominant_share: {mean_dominant_share:.4f}")
    print(f"full_grant_single_ue_rate: {full_grant_rate:.4f}")
    print(f"p95_service_gap_tti: {p95_service_gap:.4f}")
    print(f"invalid_action_rate: {invalid_action_rate:.4f}")


if __name__ == "__main__":
    env = make_env()
    base_env = env.unwrapped
    model = LTEDQNAgent.load(MODEL_PATH)

    obs, info = env.reset()
    done = False
    tti_to_show = 500
    shown_tti = 0

    plt.ion()
    fig_grid = plt.figure(figsize=(16, 9))
    gs = fig_grid.add_gridspec(
        3,
        2,
        width_ratios=[1.7, 1.0],
        height_ratios=[0.95, 1.0, 1.0],
        wspace=0.30,
        hspace=0.40,
    )
    ax_grid = fig_grid.add_subplot(gs[:, 0])
    ax_status = fig_grid.add_subplot(gs[0, 1])
    ax_curves = fig_grid.add_subplot(gs[1, 1])
    ax_bottom = gs[2, 1].subgridspec(1, 2, wspace=0.35)
    ax_tput = fig_grid.add_subplot(ax_bottom[0, 0])
    ax_alloc = fig_grid.add_subplot(ax_bottom[0, 1])
    curve_window_tti = 100
    live_metrics = _init_live_metrics(base_env.n_ue)
    window_tti = 20

    while not done and shown_tti < tti_to_show:
        action = model.predict(obs, info["action_mask"], deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        live_metrics["total_steps"] = int(live_metrics["total_steps"]) + 1
        live_metrics["invalid_action_count"] = int(live_metrics["invalid_action_count"]) + int(
            info.get("invalid_action", False) or info.get("padded_invalid_action", False)
        )

        if info["rbg_step"] == 0:
            tti = info["tti"] - 1

            base_env.render(mode="human")
            _update_live_metrics(live_metrics, base_env, info)

            ax_grid.clear()
            if len(base_env.grid_history) > 0:
                grid = np.array(base_env.grid_history[-window_tti:])
                rows, _ = grid.shape
                grid_vis = grid.copy()
                grid_vis[grid_vis < 0] = base_env.n_ue

                cmap = plt.get_cmap("tab10", base_env.n_ue + 1)
                ax_grid.imshow(
                    grid_vis.T,
                    aspect="auto",
                    origin="lower",
                    cmap=cmap,
                    vmin=0,
                    vmax=base_env.n_ue,
                )
                ax_grid.set_xlabel("TTI (последние)")
                ax_grid.set_ylabel("RBG index")
                ax_grid.set_title("Resource grid: time (X) vs RBG index (Y)")

                for x in range(rows):
                    for y in range(base_env.n_rbg):
                        ue = grid[x, y]
                        if ue >= 0:
                            ax_grid.text(
                                x,
                                y,
                                str(ue),
                                ha="center",
                                va="center",
                                fontsize=6,
                                color="white",
                            )

                start_tti = max(0, tti - rows + 1)
                xticks = np.arange(rows)
                xticklabels = [str(start_tti + i) for i in range(rows)]
                ax_grid.set_xticks(xticks)
                ax_grid.set_xticklabels(xticklabels, rotation=45)
                ax_grid.set_yticks(range(base_env.n_rbg))
                ax_grid.set_yticklabels(range(base_env.n_rbg))

            _draw_status_panel(ax_status, base_env, info, live_metrics)
            _draw_curves_panel(ax_curves, live_metrics, curve_window_tti)
            _draw_tput_panel(ax_tput, base_env)
            _draw_alloc_panel(ax_alloc, info, base_env)

            fig_grid.suptitle("LTE Scheduler Live Monitor", fontsize=16, y=0.98)
            plt.pause(0.2)
            shown_tti += 1

        done = terminated or truncated

    _print_final_summary(base_env, live_metrics)
    plt.ioff()
    plt.show()
    env.close()
