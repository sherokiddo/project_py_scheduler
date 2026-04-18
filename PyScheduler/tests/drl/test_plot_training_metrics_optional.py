import csv
import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


pytest.importorskip("matplotlib")


from drl.scripts.plot_training_metrics import generate_report


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_plot_training_metrics_generates_analysis_artifacts():
    workspace_tmp = Path("d:/project_py_scheduler/.tmp_test_artifacts")
    workspace_tmp.mkdir(parents=True, exist_ok=True)
    run_dir = workspace_tmp / f"lte_ppo_ranker_test_{uuid4().hex}"
    run_dir.mkdir()

    _write_csv(
        run_dir / "train_metrics.csv",
        [
            {
                "episode": 1,
                "mean_reward": 1.0,
                "mean_se_bps_hz": 2.5,
                "mean_jfi_active": 0.7,
                "mean_jfi_all": 0.65,
                "probe_mean_reward": "",
                "probe_mean_se_bps_hz": "",
                "probe_mean_jfi_all": "",
                "mean_loss_last_20": 10.0,
                "mean_entropy_last_20": 0.8,
                "probe_scenario": "anchor",
            },
            {
                "episode": 2,
                "mean_reward": 1.2,
                "mean_se_bps_hz": 2.8,
                "mean_jfi_active": 0.75,
                "mean_jfi_all": 0.7,
                "probe_mean_reward": 1.15,
                "probe_mean_se_bps_hz": 2.7,
                "probe_mean_jfi_all": 0.68,
                "mean_loss_last_20": 9.0,
                "mean_entropy_last_20": 0.75,
                "probe_scenario": "anchor",
            },
        ],
    )
    _write_csv(
        run_dir / "ppo_update_metrics.csv",
        [
            {
                "update_idx": 1,
                "loss": 5.0,
                "actor_loss": 1.0,
                "critic_loss": 3.5,
                "entropy": 0.9,
            },
            {
                "update_idx": 2,
                "loss": 4.2,
                "actor_loss": 0.8,
                "critic_loss": 3.0,
                "entropy": 0.85,
            },
        ],
    )
    _write_csv(
        run_dir / "eval_metrics.csv",
        [
            {
                "scenario": "anchor",
                "eval_mode": "deterministic",
                "mean_reward": 1.1,
                "mean_se_bps_hz": 3.0,
                "mean_jfi_all": 0.6,
            },
            {
                "scenario": "mid",
                "eval_mode": "deterministic",
                "mean_reward": 1.0,
                "mean_se_bps_hz": 2.9,
                "mean_jfi_all": 0.55,
            },
        ],
    )
    (run_dir / "run_config.json").write_text(
        '{"seed": 42, "run_dir": "dummy"}',
        encoding="utf-8",
    )

    try:
        output_dir = run_dir / "analysis"
        summary = generate_report(
            run_dir=run_dir,
            output_dir=output_dir,
            rolling_window=2,
        )

        assert Path(summary["summary_path"]).exists()
        assert (output_dir / "train_overview.png").exists()
        assert (output_dir / "ppo_updates.png").exists()
        assert (output_dir / "eval_summary.png").exists()
        assert summary["best_train_reward"]["episode"] == 2
        assert summary["eval"]["deterministic"]["best_reward_scenario"] == "anchor"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
