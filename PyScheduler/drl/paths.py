"""
Стандартные пути для DRL-артефактов внутри PyScheduler.
"""

from pathlib import Path


DRL_DIR = Path(__file__).resolve().parent
DRL_RUNS_DIR = DRL_DIR / "runs"

PYSCHEDULER_DQN_RUN_DIR = DRL_RUNS_DIR / "lte_dqn"
PYSCHEDULER_PPO_RUN_DIR = DRL_RUNS_DIR / "lte_ppo_tune_06_reference"
PYSCHEDULER_PPO_RANKER_RUN_DIR = DRL_RUNS_DIR / "lte_ppo_ranker_compact_v1"

# Legacy standalone playground path. Не относится к каноническому
# simulation-backed train/runtime-контуру PyScheduler.
PLAYGROUND_DQN_RUN_DIR = DRL_RUNS_DIR / "lte_dqn_playground"

PYSCHEDULER_DQN_MODEL_PATH = PYSCHEDULER_DQN_RUN_DIR / "lte_dqn_shared_q.pt"
PYSCHEDULER_PPO_MODEL_PATH = PYSCHEDULER_PPO_RUN_DIR / "lte_ppo_policy.pt"
PYSCHEDULER_PPO_RANKER_MODEL_PATH = (
    PYSCHEDULER_PPO_RANKER_RUN_DIR / "lte_ppo_ranker_policy.pt"
)
PYSCHEDULER_PPO_RANKER_ONNX_PATH = (
    PYSCHEDULER_PPO_RANKER_RUN_DIR / "lte_ppo_ranker_policy.onnx"
)
PYSCHEDULER_PPO_RANKER_TORCHSCRIPT_PATH = (
    PYSCHEDULER_PPO_RANKER_RUN_DIR / "lte_ppo_ranker_policy.ts"
)
PYSCHEDULER_PPO_RANKER_LIBTORCH_RUNTIME_DLL_PATH = (
    DRL_DIR / "cpp_ranker_runtime" / "build" / "Release" / "ppo_ranker_runtime.dll"
)
PYSCHEDULER_PPO_RANKER_ONNX_RUNTIME_DLL_PATH = (
    DRL_DIR / "cpp_onnx_runtime" / "build" / "Release" / "ppo_ranker_onnx_runtime.dll"
)

# Backward-compatible alias for older internal imports.
PYSCHEDULER_PPO_RANKER_CPP_RUNTIME_DLL_PATH = (
    PYSCHEDULER_PPO_RANKER_LIBTORCH_RUNTIME_DLL_PATH
)
PLAYGROUND_DQN_MODEL_PATH = PLAYGROUND_DQN_RUN_DIR / "lte_dqn_shared_q.pt"
