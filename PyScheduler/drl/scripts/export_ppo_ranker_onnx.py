"""
Экспорт PPO ranker checkpoint в ONNX artifact для ONNX Runtime inference.
"""

from __future__ import annotations

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
from pathlib import Path

import torch

from drl.onnx_ranker_export import (
    DEFAULT_ONNX_OPSET_VERSION,
    export_ranker_to_onnx,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PPO ranker checkpoint to ONNX for ONNX Runtime inference.",
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
        help="Куда сохранить ONNX artifact (.onnx).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Куда сохранить metadata JSON. По умолчанию рядом с .onnx.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Устройство, на котором загружать checkpoint перед export.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=DEFAULT_ONNX_OPSET_VERSION,
        help="ONNX opset version для export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested, but torch.cuda.is_available() == False.")

    result = export_ranker_to_onnx(
        checkpoint_path=args.checkpoint_path,
        onnx_path=args.output_path,
        metadata_path=args.metadata_path,
        device=args.device,
        opset_version=args.opset_version,
    )

    print(f"ONNX saved: {result['onnx_path']}")
    print(f"Metadata saved: {result['metadata_path']}")
    print(f"obs_dim: {result['metadata']['obs_dim']}")
    print(f"max_n_ue: {result['metadata']['max_n_ue']}")
    print(f"ue_feature_dim: {result['metadata']['ue_feature_dim']}")
    print(f"context_dim: {result['metadata']['context_dim']}")
    print(f"opset_version: {result['metadata']['onnx_opset_version']}")


if __name__ == "__main__":
    main()
