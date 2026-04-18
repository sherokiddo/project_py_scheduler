"""
Python bridge для вызова C++ ONNX PPO ranker runtime через ctypes.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from drl.cpp_libtorch_ranker_bridge import normalize_cpp_runtime_search_paths


class CppONNXPPORankerBridge:
    """
    Тонкая Python-обертка над `ppo_ranker_onnx_runtime.dll`.
    """

    def __init__(
        self,
        *,
        runtime_library_path: str | Path,
        model_path: str | Path,
        dll_search_paths: Optional[Sequence[str | Path]] = None,
    ) -> None:
        self.runtime_library_path = Path(runtime_library_path)
        self.model_path = Path(model_path)
        self.dll_search_paths = normalize_cpp_runtime_search_paths(dll_search_paths)
        self._dll_directory_handles: list[object] = []
        self._handle = ctypes.c_void_p()

        if not self.runtime_library_path.exists():
            raise FileNotFoundError(
                f"C++ ONNX runtime library not found: {self.runtime_library_path}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found for C++ ranker backend: {self.model_path}"
            )

        self._register_dll_search_paths()
        try:
            self._lib = ctypes.CDLL(str(self.runtime_library_path))
        except OSError as error:
            self.close()
            raise RuntimeError(
                "Failed to load C++ ONNX PPO ranker runtime library. "
                "Check runtime_library_path and dll_search_paths for ONNX Runtime DLLs. "
                f"Original error: {error}"
            ) from error
        self._configure_signatures()
        self._create_runtime()

    def predict_scores(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32).reshape(-1)
        mask_array = np.asarray(action_mask, dtype=bool).reshape(-1)
        mask_i32 = mask_array.astype(np.int32, copy=False)
        out_scores = np.empty(mask_i32.shape[0], dtype=np.float32)

        status = self._lib.ppo_ranker_onnx_predict(
            self._handle,
            obs_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(obs_array.size),
            mask_i32.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            int(mask_i32.size),
            out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(out_scores.size),
        )
        if status != 0:
            raise RuntimeError(
                "C++ ONNX PPO ranker predict failed: "
                f"{self._get_last_error()}"
            )

        return out_scores

    def close(self) -> None:
        if getattr(self, "_handle", None) and self._handle.value:
            self._lib.ppo_ranker_onnx_destroy(self._handle)
            self._handle = ctypes.c_void_p()

        for handle in getattr(self, "_dll_directory_handles", []):
            close_fn = getattr(handle, "close", None)
            if callable(close_fn):
                close_fn()
        self._dll_directory_handles = []

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _configure_signatures(self) -> None:
        self._lib.ppo_ranker_onnx_create.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.ppo_ranker_onnx_create.restype = ctypes.c_int

        self._lib.ppo_ranker_onnx_predict.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self._lib.ppo_ranker_onnx_predict.restype = ctypes.c_int

        self._lib.ppo_ranker_onnx_destroy.argtypes = [ctypes.c_void_p]
        self._lib.ppo_ranker_onnx_destroy.restype = None

        self._lib.ppo_ranker_onnx_last_error.argtypes = []
        self._lib.ppo_ranker_onnx_last_error.restype = ctypes.c_char_p

    def _create_runtime(self) -> None:
        status = self._lib.ppo_ranker_onnx_create(
            str(self.model_path).encode("utf-8"),
            ctypes.byref(self._handle),
        )
        if status != 0:
            raise RuntimeError(
                "C++ ONNX PPO ranker create failed: "
                f"{self._get_last_error()}"
            )

    def _get_last_error(self) -> str:
        error_ptr = self._lib.ppo_ranker_onnx_last_error()
        if not error_ptr:
            return "unknown error"
        if isinstance(error_ptr, bytes):
            return error_ptr.decode("utf-8")
        return str(error_ptr)

    def _register_dll_search_paths(self) -> None:
        if os.name != "nt" or not hasattr(os, "add_dll_directory"):
            return

        paths_to_add = [self.runtime_library_path.parent, *self.dll_search_paths]
        for raw_path in paths_to_add:
            resolved = Path(raw_path)
            if resolved.exists():
                self._dll_directory_handles.append(
                    os.add_dll_directory(str(resolved))
                )
