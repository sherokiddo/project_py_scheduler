"""
Стандартные пути для DRL-артефактов внутри PyScheduler.
"""

from pathlib import Path


DRL_DIR = Path(__file__).resolve().parent
DRL_RUNS_DIR = DRL_DIR / "runs"

PYSCHEDULER_DQN_RUN_DIR = DRL_RUNS_DIR / "lte_dqn"
PLAYGROUND_DQN_RUN_DIR = DRL_RUNS_DIR / "lte_dqn_playground"

PYSCHEDULER_DQN_MODEL_PATH = PYSCHEDULER_DQN_RUN_DIR / "lte_dqn_shared_q.pt"
PLAYGROUND_DQN_MODEL_PATH = PLAYGROUND_DQN_RUN_DIR / "lte_dqn_shared_q.pt"
