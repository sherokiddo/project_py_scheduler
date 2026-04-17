"""
Python bridge для вызова C++ PPO ranker runtime через ctypes.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


class CppPPORankerBridge:
    """
    Тонкая Python-обертка над `ppo_ranker_runtime.dll`.

    Bridge:
    - загружает DLL;
    - создает runtime-хэндл с уже загруженной TorchScript моделью;
    - вызывает deterministic inference;
    - освобождает runtime при завершении.
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
        self.dll_search_paths = self._normalize_dll_search_paths(dll_search_paths)
        self._dll_directory_handles: list[object] = []
        self._handle = ctypes.c_void_p()

        if not self.runtime_library_path.exists():
            raise FileNotFoundError(
                f"C++ runtime library not found: {self.runtime_library_path}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"TorchScript model not found for C++ ranker backend: {self.model_path}"
            )

        self._register_dll_search_paths()
        try:
            self._lib = ctypes.CDLL(str(self.runtime_library_path))
        except OSError as error:
            self.close()
            raise RuntimeError(
                "Failed to load C++ PPO ranker runtime library. "
                "Check runtime_library_path and dll_search_paths for LibTorch DLLs. "
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

        status = self._lib.ppo_ranker_predict(
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
                "C++ PPO ranker predict failed: "
                f"{self._get_last_error()}"
            )

        return out_scores

    def close(self) -> None:
        if getattr(self, "_handle", None) and self._handle.value:
            self._lib.ppo_ranker_destroy(self._handle)
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
        self._lib.ppo_ranker_create.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.ppo_ranker_create.restype = ctypes.c_int

        self._lib.ppo_ranker_predict.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self._lib.ppo_ranker_predict.restype = ctypes.c_int

        self._lib.ppo_ranker_destroy.argtypes = [ctypes.c_void_p]
        self._lib.ppo_ranker_destroy.restype = None

        self._lib.ppo_ranker_last_error.argtypes = []
        self._lib.ppo_ranker_last_error.restype = ctypes.c_char_p

    def _create_runtime(self) -> None:
        status = self._lib.ppo_ranker_create(
            str(self.model_path).encode("utf-8"),
            ctypes.byref(self._handle),
        )
        if status != 0:
            raise RuntimeError(
                "C++ PPO ranker create failed: "
                f"{self._get_last_error()}"
            )

    def _get_last_error(self) -> str:
        error_ptr = self._lib.ppo_ranker_last_error()
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

    @staticmethod
    def _normalize_dll_search_paths(
        dll_search_paths: Optional[Sequence[str | Path]],
    ) -> list[Path]:
        if dll_search_paths is None:
            return []

        normalized: list[Path] = []
        for path in dll_search_paths:
            resolved = Path(path)
            if resolved not in normalized:
                normalized.append(resolved)
        return normalized


def normalize_cpp_runtime_search_paths(
    raw_value: Optional[str | Path | Iterable[str | Path]],
) -> list[Path]:
    """
    Нормализовать конфигурацию путей поиска DLL для C++ runtime.
    """

    if raw_value is None:
        return []

    if isinstance(raw_value, (str, Path)):
        return [Path(raw_value)]

    normalized: list[Path] = []
    for item in raw_value:
        path = Path(item)
        if path not in normalized:
            normalized.append(path)
    return normalized
