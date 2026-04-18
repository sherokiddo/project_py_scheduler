import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import drl.cpp_libtorch_ranker_bridge as cpp_ranker_bridge_module
from TEST_MODULES import _resolve_runtime_scheduler_config
from drl.model_runners import (
    BaseTorchAgentRunner,
    CppONNXPPORankerModelRunner,
    CppPPORankerModelRunner,
    DQNModelRunner,
    PPOModelRunner,
    PPORankerModelRunner,
)


def _workspace_tmp_dir() -> Path:
    base_dir = Path("d:/project_py_scheduler/.tmp_test_artifacts")
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / f"model_runners_{uuid4().hex}"
    run_dir.mkdir()
    return run_dir


def test_torch_model_runners_share_common_base():
    assert issubclass(DQNModelRunner, BaseTorchAgentRunner)
    assert issubclass(PPOModelRunner, BaseTorchAgentRunner)
    assert issubclass(PPORankerModelRunner, BaseTorchAgentRunner)


def test_test_modules_resolves_ranker_runtime_backends():
    python_model_path, python_kwargs = _resolve_runtime_scheduler_config(
        "PpoRankerScheduler",
        runtime_backend="python",
    )
    assert python_model_path.suffix == ".pt"
    assert python_kwargs["ppo_ranker_backend"] == "python"
    assert "ppo_ranker_inference_device" in python_kwargs

    cpp_model_path, cpp_kwargs = _resolve_runtime_scheduler_config(
        "PpoRankerScheduler",
        runtime_backend="cpp",
    )
    assert cpp_model_path.suffix == ".ts"
    assert cpp_kwargs["ppo_ranker_backend"] == "cpp"
    assert cpp_kwargs["ppo_ranker_runtime_library_path"].endswith(
        "ppo_ranker_runtime.dll"
    )
    assert cpp_kwargs["ppo_ranker_runtime_dll_search_paths"]

    onnx_model_path, onnx_kwargs = _resolve_runtime_scheduler_config(
        "PpoRankerScheduler",
        runtime_backend="onnx_cpp",
    )
    assert onnx_model_path.suffix == ".onnx"
    assert onnx_kwargs["ppo_ranker_backend"] == "onnx_cpp"
    assert onnx_kwargs["ppo_ranker_runtime_library_path"].endswith(
        "ppo_ranker_onnx_runtime.dll"
    )
    assert onnx_kwargs["ppo_ranker_runtime_dll_search_paths"]


def test_cpp_ppo_ranker_model_runner_uses_bridge(monkeypatch):
    workdir = _workspace_tmp_dir()
    model_path = workdir / "ranker.ts"
    runtime_library_path = workdir / "ppo_ranker_runtime.dll"
    model_path.write_bytes(b"ts")
    runtime_library_path.write_bytes(b"dll")

    class FakeBridge:
        def __init__(self, *, runtime_library_path, model_path, dll_search_paths=None):
            self.runtime_library_path = runtime_library_path
            self.model_path = model_path
            self.dll_search_paths = dll_search_paths

        def predict_scores(self, obs, action_mask):
            obs = np.asarray(obs, dtype=np.float32)
            action_mask = np.asarray(action_mask, dtype=bool)
            return np.where(action_mask, obs[: action_mask.size], -1e9).astype(
                np.float32
            )

        def close(self):
            return None

    monkeypatch.setattr(
        cpp_ranker_bridge_module,
        "CppPPORankerBridge",
        FakeBridge,
    )

    runner = None
    try:
        runner = CppPPORankerModelRunner(
            model_path=str(model_path),
            runtime_library_path=str(runtime_library_path),
            dll_search_paths=[workdir],
            deterministic=True,
            max_n_ue=4,
        )

        scores = runner.predict(
            obs=np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
            action_mask=np.asarray([1, 1, 0, 0], dtype=bool),
            deterministic=True,
        )

        assert np.allclose(
            scores, np.asarray([0.2, 0.4, -1e9, -1e9], dtype=np.float32)
        )
        assert runner.inference_device == "cpu_cpp_libtorch"
        assert runner.max_n_ue == 4
    finally:
        if runner is not None:
            runner.close()
        shutil.rmtree(workdir, ignore_errors=True)


def test_cpp_onnx_ppo_ranker_model_runner_uses_bridge(monkeypatch):
    workdir = _workspace_tmp_dir()
    model_path = workdir / "ranker.onnx"
    runtime_library_path = workdir / "ppo_ranker_onnx_runtime.dll"
    model_path.write_bytes(b"onnx")
    runtime_library_path.write_bytes(b"dll")

    class FakeBridge:
        def __init__(self, *, runtime_library_path, model_path, dll_search_paths=None):
            self.runtime_library_path = runtime_library_path
            self.model_path = model_path
            self.dll_search_paths = dll_search_paths

        def predict_scores(self, obs, action_mask):
            obs = np.asarray(obs, dtype=np.float32)
            action_mask = np.asarray(action_mask, dtype=bool)
            return np.where(action_mask, obs[: action_mask.size], -1e9).astype(
                np.float32
            )

        def close(self):
            return None

    import drl.cpp_onnx_ranker_bridge as cpp_onnx_ranker_bridge_module

    monkeypatch.setattr(
        cpp_onnx_ranker_bridge_module,
        "CppONNXPPORankerBridge",
        FakeBridge,
    )

    runner = None
    try:
        runner = CppONNXPPORankerModelRunner(
            model_path=str(model_path),
            runtime_library_path=str(runtime_library_path),
            dll_search_paths=[workdir],
            deterministic=True,
            max_n_ue=4,
        )

        scores = runner.predict(
            obs=np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
            action_mask=np.asarray([1, 1, 0, 0], dtype=bool),
            deterministic=True,
        )

        assert np.allclose(
            scores, np.asarray([0.2, 0.4, -1e9, -1e9], dtype=np.float32)
        )
        assert runner.inference_device == "cpu_cpp_onnxruntime"
        assert runner.max_n_ue == 4
    finally:
        if runner is not None:
            runner.close()
        shutil.rmtree(workdir, ignore_errors=True)
