import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import ctypes
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drl.cpp_onnx_ranker_bridge import CppONNXPPORankerBridge


def _workspace_tmp_dir() -> Path:
    base_dir = Path("d:/project_py_scheduler/.tmp_test_artifacts")
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / f"cpp_onnx_ranker_bridge_{uuid4().hex}"
    run_dir.mkdir()
    return run_dir


class _FakeFn:
    def __init__(self, impl):
        self.impl = impl
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.impl(*args)


class _FakeLib:
    def __init__(self):
        self.destroy_calls = 0
        self.ppo_ranker_onnx_create = _FakeFn(self._create)
        self.ppo_ranker_onnx_predict = _FakeFn(self._predict)
        self.ppo_ranker_onnx_destroy = _FakeFn(self._destroy)
        self.ppo_ranker_onnx_last_error = _FakeFn(self._last_error)

    def _create(self, model_path, out_handle):
        assert model_path.endswith(b".onnx")
        handle_ptr = ctypes.cast(out_handle, ctypes.POINTER(ctypes.c_void_p))
        handle_ptr[0] = ctypes.c_void_p(4321)
        return 0

    def _predict(
        self,
        handle,
        obs,
        obs_len,
        action_mask,
        action_mask_len,
        out_scores,
        out_scores_len,
    ):
        assert handle
        obs_arr = np.ctypeslib.as_array(obs, shape=(obs_len,))
        mask_arr = np.ctypeslib.as_array(action_mask, shape=(action_mask_len,))
        out_arr = np.ctypeslib.as_array(out_scores, shape=(out_scores_len,))
        out_arr[:] = np.where(mask_arr != 0, obs_arr[:out_scores_len], -1e9)
        return 0

    def _destroy(self, handle):
        if handle:
            self.destroy_calls += 1

    def _last_error(self):
        return b"fake error"


def test_cpp_onnx_ranker_bridge_predict_scores(monkeypatch):
    workdir = _workspace_tmp_dir()
    model_path = workdir / "ranker.onnx"
    runtime_library_path = workdir / "ppo_ranker_onnx_runtime.dll"
    model_path.write_bytes(b"onnx")
    runtime_library_path.write_bytes(b"dll")

    fake_lib = _FakeLib()
    monkeypatch.setattr(ctypes, "CDLL", lambda path: fake_lib)

    bridge = CppONNXPPORankerBridge(
        runtime_library_path=runtime_library_path,
        model_path=model_path,
        dll_search_paths=[workdir],
    )
    try:
        scores = bridge.predict_scores(
            obs=np.asarray([0.5, 0.7, 0.9, 1.1], dtype=np.float32),
            action_mask=np.asarray([1, 0, 1, 0], dtype=bool),
        )
        assert np.allclose(
            scores,
            np.asarray([0.5, -1e9, 0.9, -1e9], dtype=np.float32),
        )
        assert fake_lib.destroy_calls == 0
    finally:
        bridge.close()
        shutil.rmtree(workdir, ignore_errors=True)

    assert fake_lib.destroy_calls == 1
