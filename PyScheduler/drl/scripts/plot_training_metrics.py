"""
Постобработка результатов обучения DRL-моделей по сохраненным CSV-файлам.

Скрипт читает артефакты из каталога запуска:
- `train_metrics.csv`
- `ppo_update_metrics.csv`
- `eval_metrics.csv`
- `run_config.json`

И строит:
- кривые обучения по episode-level метрикам;
- графики PPO update-метрик;
- сводку по eval-сценариям;
- `summary.json` с короткой агрегированной статистикой.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построить графики и summary по DRL training run.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Каталог запуска, где лежат train/eval CSV и run_config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Куда сохранять графики. По умолчанию <run-dir>/analysis.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help="Размер окна для сглаживания кривых по эпизодам.",
    )
    return parser.parse_args()


def _coerce_value(value: str) -> Any:
    text = value.strip()
    if text == "":
        return np.nan

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        numeric = float(text)
    except ValueError:
        return text

    if math.isfinite(numeric) and numeric.is_integer():
        return int(numeric)
    return float(numeric)


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return [
            {key: _coerce_value(value) for key, value in row.items()}
            for row in reader
        ]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values

    resolved_window = max(int(window), 1)
    result = np.empty(len(values), dtype=np.float64)
    for idx in range(len(values)):
        start = max(0, idx - resolved_window + 1)
        result[idx] = float(np.nanmean(values[start : idx + 1]))
    return result


def _series_from_rows(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    for row in rows:
        x_val = row.get(x_key)
        y_val = row.get(y_key)
        if x_val is None or y_val is None:
            continue
        try:
            x_num = float(x_val)
            y_num = float(y_val)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x_num) and math.isfinite(y_num)):
            continue
        points.append((x_num, y_num))

    if not points:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    x = np.asarray([item[0] for item in points], dtype=np.float64)
    y = np.asarray([item[1] for item in points], dtype=np.float64)
    return x, y


def _plot_series_with_smoothing(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    label: str,
    rolling_window: int,
    color: str,
) -> bool:
    x, y = _series_from_rows(rows, x_key=x_key, y_key=y_key)
    if len(x) == 0:
        return False

    ax.plot(x, y, alpha=0.25, linewidth=1.0, color=color)
    ax.plot(
        x,
        rolling_mean(y, rolling_window),
        linewidth=2.0,
        color=color,
        label=label,
    )
    return True


def plot_train_metrics(
    train_rows: list[dict[str, Any]],
    *,
    output_path: Path,
    rolling_window: int,
    run_name: str,
) -> bool:
    if plt is None or not train_rows:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_reward, ax_se, ax_jfi, ax_probe = axes.flatten()

    reward_added = _plot_series_with_smoothing(
        ax_reward,
        train_rows,
        x_key="episode",
        y_key="mean_reward",
        label="train reward",
        rolling_window=rolling_window,
        color="tab:blue",
    )
    probe_reward_added = _plot_series_with_smoothing(
        ax_reward,
        train_rows,
        x_key="episode",
        y_key="probe_mean_reward",
        label="probe reward",
        rolling_window=rolling_window,
        color="tab:orange",
    )
    ax_reward.set_title("Reward")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True, alpha=0.3)
    if reward_added or probe_reward_added:
        ax_reward.legend()

    se_added = _plot_series_with_smoothing(
        ax_se,
        train_rows,
        x_key="episode",
        y_key="mean_se_bps_hz",
        label="train SE",
        rolling_window=rolling_window,
        color="tab:green",
    )
    probe_se_added = _plot_series_with_smoothing(
        ax_se,
        train_rows,
        x_key="episode",
        y_key="probe_mean_se_bps_hz",
        label="probe SE",
        rolling_window=rolling_window,
        color="tab:red",
    )
    ax_se.set_title("Spectral Efficiency")
    ax_se.set_xlabel("Episode")
    ax_se.set_ylabel("bps/Hz")
    ax_se.grid(True, alpha=0.3)
    if se_added or probe_se_added:
        ax_se.legend()

    jfi_act_added = _plot_series_with_smoothing(
        ax_jfi,
        train_rows,
        x_key="episode",
        y_key="mean_jfi_active",
        label="JFI active",
        rolling_window=rolling_window,
        color="tab:purple",
    )
    jfi_all_added = _plot_series_with_smoothing(
        ax_jfi,
        train_rows,
        x_key="episode",
        y_key="mean_jfi_all",
        label="JFI all",
        rolling_window=rolling_window,
        color="tab:brown",
    )
    probe_jfi_added = _plot_series_with_smoothing(
        ax_jfi,
        train_rows,
        x_key="episode",
        y_key="probe_mean_jfi_all",
        label="probe JFI all",
        rolling_window=rolling_window,
        color="tab:cyan",
    )
    ax_jfi.set_title("Fairness")
    ax_jfi.set_xlabel("Episode")
    ax_jfi.set_ylabel("JFI")
    ax_jfi.set_ylim(0.0, 1.05)
    ax_jfi.grid(True, alpha=0.3)
    if jfi_act_added or jfi_all_added or probe_jfi_added:
        ax_jfi.legend()

    probe_lines = []
    probe_lines.append(
        _plot_series_with_smoothing(
            ax_probe,
            train_rows,
            x_key="episode",
            y_key="mean_loss_last_20",
            label="mean loss last 20",
            rolling_window=rolling_window,
            color="tab:red",
        )
    )
    probe_lines.append(
        _plot_series_with_smoothing(
            ax_probe,
            train_rows,
            x_key="episode",
            y_key="mean_entropy_last_20",
            label="mean entropy last 20",
            rolling_window=rolling_window,
            color="tab:olive",
        )
    )
    ax_probe.set_title("Episode-side optimizer signals")
    ax_probe.set_xlabel("Episode")
    ax_probe.grid(True, alpha=0.3)
    if any(probe_lines):
        ax_probe.legend()

    fig.suptitle(f"Training overview: {run_name}", fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_ppo_update_metrics(
    update_rows: list[dict[str, Any]],
    *,
    output_path: Path,
    rolling_window: int,
    run_name: str,
) -> bool:
    if plt is None or not update_rows:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_loss, ax_actor, ax_critic, ax_entropy = axes.flatten()

    items = [
        (ax_loss, "loss", "Total loss", "tab:red"),
        (ax_actor, "actor_loss", "Actor loss", "tab:blue"),
        (ax_critic, "critic_loss", "Critic loss", "tab:green"),
        (ax_entropy, "entropy", "Entropy", "tab:orange"),
    ]
    for ax, key, title, color in items:
        added = _plot_series_with_smoothing(
            ax,
            update_rows,
            x_key="update_idx",
            y_key=key,
            label=key,
            rolling_window=rolling_window,
            color=color,
        )
        ax.set_title(title)
        ax.set_xlabel("Update")
        ax.grid(True, alpha=0.3)
        if added:
            ax.legend()

    fig.suptitle(f"PPO updates: {run_name}", fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_eval_summary(
    eval_rows: list[dict[str, Any]],
    *,
    output_path: Path,
    run_name: str,
) -> bool:
    if plt is None or not eval_rows:
        return False

    deterministic_rows = [
        row for row in eval_rows if str(row.get("eval_mode", "deterministic")) == "deterministic"
    ]
    if not deterministic_rows:
        deterministic_rows = eval_rows

    scenarios = [str(row.get("scenario", f"scenario_{idx}")) for idx, row in enumerate(deterministic_rows)]
    reward = np.asarray([float(row.get("mean_reward", np.nan)) for row in deterministic_rows], dtype=np.float64)
    se = np.asarray([float(row.get("mean_se_bps_hz", np.nan)) for row in deterministic_rows], dtype=np.float64)
    jfi = np.asarray([float(row.get("mean_jfi_all", np.nan)) for row in deterministic_rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    metrics = [
        (axes[0], reward, "Mean reward", "tab:blue"),
        (axes[1], se, "Mean SE", "tab:green"),
        (axes[2], jfi, "Mean JFI(all)", "tab:orange"),
    ]
    y_pos = np.arange(len(scenarios))

    for ax, values, title, color in metrics:
        ax.barh(y_pos, values, color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_yticks(y_pos, scenarios)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Evaluation summary: {run_name}", fontsize=14)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def build_summary(
    *,
    train_rows: list[dict[str, Any]],
    update_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_config": run_config,
        "num_train_rows": len(train_rows),
        "num_update_rows": len(update_rows),
        "num_eval_rows": len(eval_rows),
    }

    if train_rows:
        last_train = train_rows[-1]
        summary["last_train"] = {
            "episode": int(last_train.get("episode", len(train_rows))),
            "mean_reward": float(last_train.get("mean_reward", np.nan)),
            "mean_se_bps_hz": float(last_train.get("mean_se_bps_hz", np.nan)),
            "mean_jfi_all": float(last_train.get("mean_jfi_all", np.nan)),
            "probe_mean_reward": float(last_train.get("probe_mean_reward", np.nan)),
            "probe_mean_se_bps_hz": float(last_train.get("probe_mean_se_bps_hz", np.nan)),
            "probe_mean_jfi_all": float(last_train.get("probe_mean_jfi_all", np.nan)),
        }

        best_reward_row = max(
            train_rows,
            key=lambda row: float(row.get("mean_reward", -np.inf)),
        )
        best_jfi_row = max(
            train_rows,
            key=lambda row: float(row.get("mean_jfi_all", -np.inf)),
        )
        summary["best_train_reward"] = {
            "episode": int(best_reward_row.get("episode", -1)),
            "mean_reward": float(best_reward_row.get("mean_reward", np.nan)),
        }
        summary["best_train_jfi_all"] = {
            "episode": int(best_jfi_row.get("episode", -1)),
            "mean_jfi_all": float(best_jfi_row.get("mean_jfi_all", np.nan)),
        }

        probe_candidates = [
            row for row in train_rows if math.isfinite(float(row.get("probe_mean_reward", np.nan)))
        ]
        if probe_candidates:
            best_probe_row = max(
                probe_candidates,
                key=lambda row: float(row.get("probe_mean_reward", -np.inf)),
            )
            summary["best_probe_reward"] = {
                "episode": int(best_probe_row.get("episode", -1)),
                "probe_scenario": str(best_probe_row.get("probe_scenario", "")),
                "probe_mean_reward": float(best_probe_row.get("probe_mean_reward", np.nan)),
                "probe_mean_se_bps_hz": float(best_probe_row.get("probe_mean_se_bps_hz", np.nan)),
                "probe_mean_jfi_all": float(best_probe_row.get("probe_mean_jfi_all", np.nan)),
            }

    if update_rows:
        last_update = update_rows[-1]
        summary["last_update"] = {
            "update_idx": int(last_update.get("update_idx", len(update_rows))),
            "loss": float(last_update.get("loss", np.nan)),
            "actor_loss": float(last_update.get("actor_loss", np.nan)),
            "critic_loss": float(last_update.get("critic_loss", np.nan)),
            "entropy": float(last_update.get("entropy", np.nan)),
        }

    if eval_rows:
        eval_by_mode: dict[str, list[dict[str, Any]]] = {}
        for row in eval_rows:
            mode = str(row.get("eval_mode", "deterministic"))
            eval_by_mode.setdefault(mode, []).append(row)

        summary["eval"] = {}
        for mode, rows in eval_by_mode.items():
            best_reward_row = max(
                rows,
                key=lambda row: float(row.get("mean_reward", -np.inf)),
            )
            best_jfi_row = max(
                rows,
                key=lambda row: float(row.get("mean_jfi_all", -np.inf)),
            )
            summary["eval"][mode] = {
                "mean_reward_avg": float(np.mean([float(row.get("mean_reward", np.nan)) for row in rows])),
                "mean_se_avg": float(np.mean([float(row.get("mean_se_bps_hz", np.nan)) for row in rows])),
                "mean_jfi_all_avg": float(np.mean([float(row.get("mean_jfi_all", np.nan)) for row in rows])),
                "best_reward_scenario": str(best_reward_row.get("scenario", "")),
                "best_reward_value": float(best_reward_row.get("mean_reward", np.nan)),
                "best_jfi_scenario": str(best_jfi_row.get("scenario", "")),
                "best_jfi_value": float(best_jfi_row.get("mean_jfi_all", np.nan)),
            }

    return summary


def generate_report(
    *,
    run_dir: Path,
    output_dir: Path,
    rolling_window: int,
) -> dict[str, Any]:
    train_rows = load_csv_rows(run_dir / "train_metrics.csv")
    update_rows = load_csv_rows(run_dir / "ppo_update_metrics.csv")
    eval_rows = load_csv_rows(run_dir / "eval_metrics.csv")
    run_config = load_json(run_dir / "run_config.json")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_dir.name

    generated_files: dict[str, str] = {}
    if plot_train_metrics(
        train_rows,
        output_path=output_dir / "train_overview.png",
        rolling_window=rolling_window,
        run_name=run_name,
    ):
        generated_files["train_overview"] = str(output_dir / "train_overview.png")

    if plot_ppo_update_metrics(
        update_rows,
        output_path=output_dir / "ppo_updates.png",
        rolling_window=rolling_window,
        run_name=run_name,
    ):
        generated_files["ppo_updates"] = str(output_dir / "ppo_updates.png")

    if plot_eval_summary(
        eval_rows,
        output_path=output_dir / "eval_summary.png",
        run_name=run_name,
    ):
        generated_files["eval_summary"] = str(output_dir / "eval_summary.png")

    summary = build_summary(
        train_rows=train_rows,
        update_rows=update_rows,
        eval_rows=eval_rows,
        run_config=run_config,
    )
    summary["generated_files"] = generated_files

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def main() -> None:
    args = parse_args()

    if plt is None:
        raise ModuleNotFoundError(
            "plot_training_metrics.py requires matplotlib to be installed."
        )

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else run_dir / "analysis"
    summary = generate_report(
        run_dir=run_dir,
        output_dir=output_dir,
        rolling_window=int(args.rolling_window),
    )

    print(f"Analysis directory: {output_dir}")
    for label, path in summary.get("generated_files", {}).items():
        print(f"{label}: {path}")
    print(f"summary: {summary['summary_path']}")


if __name__ == "__main__":
    main()
