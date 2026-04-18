"""
Подготовить sample-pack для LibTorch smoke-test PPO ranker.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
from pathlib import Path

from drl.libtorch_ranker_smoke import prepare_ranker_libtorch_smoke_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare sample input/output pack for PPO ranker LibTorch smoke-test.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Путь к checkpoint .pt от LTEPPORankerAgent.",
    )
    parser.add_argument(
        "--torchscript-path",
        type=Path,
        required=True,
        help="Путь к уже экспортированному TorchScript artifact (.ts).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Каталог для smoke-test файлов.",
    )
    parser.add_argument(
        "--scenario-key",
        default="anchor_5ue_10mhz_wb5_umi_fb",
        help="Runtime-сценарий, из которого снять observation/action_mask.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed для reproducible sample-pack.",
    )
    parser.add_argument(
        "--rank-weight-beta",
        type=float,
        default=0.3,
        help="Параметр среды ranker, чтобы sample соответствовал runtime semantics.",
    )
    parser.add_argument(
        "--pf-epsilon-bps",
        type=float,
        default=1e-6,
        help="Параметр среды ranker, чтобы sample соответствовал runtime semantics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = prepare_ranker_libtorch_smoke_pack(
        checkpoint_path=args.checkpoint_path,
        torchscript_path=args.torchscript_path,
        output_dir=args.output_dir,
        scenario_key=str(args.scenario_key),
        seed=int(args.seed),
        rank_weight_beta=float(args.rank_weight_beta),
        pf_epsilon_bps=float(args.pf_epsilon_bps),
    )

    print(f"Smoke manifest saved: {result['manifest_path']}")
    print(f"obs: {result['obs_path']}")
    print(f"action_mask: {result['action_mask_path']}")
    print(f"expected_scores: {result['expected_scores_path']}")
    print(f"obs_dim: {result['obs_dim']}")
    print(f"max_n_ue: {result['max_n_ue']}")


if __name__ == "__main__":
    main()
