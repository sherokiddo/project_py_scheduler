"""
Экспорт PPO ranker checkpoint в TorchScript artifact для LibTorch/C++ inference.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
from pathlib import Path

import torch

from drl.torchscript_ranker_export import export_ranker_to_torchscript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PPO ranker checkpoint to TorchScript for LibTorch inference.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Путь к checkpoint .pt от LTEPPORankerAgent.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Куда сохранить TorchScript artifact (.ts).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Куда сохранить metadata JSON. По умолчанию рядом с .ts.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Устройство, на котором загружать checkpoint перед export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but torch.cuda.is_available() == False.")

    result = export_ranker_to_torchscript(
        checkpoint_path=args.checkpoint_path,
        torchscript_path=args.output_path,
        metadata_path=args.metadata_path,
        device=args.device,
    )

    print(f"TorchScript saved: {result['torchscript_path']}")
    print(f"Metadata saved: {result['metadata_path']}")
    print(f"obs_dim: {result['metadata']['obs_dim']}")
    print(f"max_n_ue: {result['metadata']['max_n_ue']}")
    print(f"ue_feature_dim: {result['metadata']['ue_feature_dim']}")
    print(f"context_dim: {result['metadata']['context_dim']}")


if __name__ == "__main__":
    main()
