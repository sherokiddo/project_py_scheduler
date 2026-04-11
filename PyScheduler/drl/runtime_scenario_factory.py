"""
Фабрика runtime-сценариев для обучения и оценки DRL на реальном PyScheduler.

Модуль держит в одном месте:
- описание сценариев;
- curriculum-наборы;
- сборку `SimulationManager`;
- factory для `PySchedulerLteEnv`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

import GLOBALS
from BS_MODULE import BaseStation
from MOBILITY_MODEL import MapBorders
from SIMULATION_MANAGER import SimulationManager
from UE_MODULE import UECollection

try:
    import torch
except ModuleNotFoundError:
    torch = None


@dataclass(frozen=True)
class RuntimeTrainingScenario:
    """
    Конфигурация одного training/eval сценария для реального runtime PyScheduler.
    """

    name: str
    n_ue: int
    bandwidth_mhz: float
    sim_duration_tti: int
    wb_cqi_report_period_tti: int
    traffic_model_type: str = "Poisson"
    traffic_packet_rate: int = 5000
    update_interval_tti: int = 1
    mobility_update_interval_tti: int = 500
    channel_update_interval_tti: int = 10
    map_radius_m: float = 1000.0
    ue_class: str = "random"
    mobility_model: str = "RandomWaypoint"
    max_dl_ue_tti: Optional[int] = None
    pcfich: int = 2
    max_dl_cce_allowance: Optional[int] = None
    enable_window: bool = False
    window_size: int = 4
    ch_model_type: str = "UMa"
    enable_tdl: bool = False


SCENARIO_CONFIGS: Dict[str, RuntimeTrainingScenario] = {
    "train_3ue_10mhz_wb5": RuntimeTrainingScenario(
        name="train_3ue_10mhz_wb5",
        n_ue=3,
        bandwidth_mhz=10,
        sim_duration_tti=200,
        wb_cqi_report_period_tti=5,
    ),
    "mid_8ue_10mhz_wb5": RuntimeTrainingScenario(
        name="mid_8ue_10mhz_wb5",
        n_ue=8,
        bandwidth_mhz=10,
        sim_duration_tti=200,
        wb_cqi_report_period_tti=5,
    ),
    "mid_16ue_10mhz_wb5": RuntimeTrainingScenario(
        name="mid_16ue_10mhz_wb5",
        n_ue=16,
        bandwidth_mhz=10,
        sim_duration_tti=200,
        wb_cqi_report_period_tti=5,
    ),
    "target_40ue_10mhz_wb5": RuntimeTrainingScenario(
        name="target_40ue_10mhz_wb5",
        n_ue=40,
        bandwidth_mhz=10,
        sim_duration_tti=200,
        wb_cqi_report_period_tti=5,
    ),
    "bw_16ue_5mhz_wb5": RuntimeTrainingScenario(
        name="bw_16ue_5mhz_wb5",
        n_ue=16,
        bandwidth_mhz=5,
        sim_duration_tti=200,
        wb_cqi_report_period_tti=5,
    ),
    "bw_16ue_20mhz_wb5": RuntimeTrainingScenario(
        name="bw_16ue_20mhz_wb5",
        n_ue=16,
        bandwidth_mhz=20,
        sim_duration_tti=200,
        wb_cqi_report_period_tti=5,
    ),
    "cqi_16ue_10mhz_wb1": RuntimeTrainingScenario(
        name="cqi_16ue_10mhz_wb1",
        n_ue=16,
        bandwidth_mhz=10,
        sim_duration_tti=200,
        wb_cqi_report_period_tti=1,
    ),
    "cqi_16ue_10mhz_wb10": RuntimeTrainingScenario(
        name="cqi_16ue_10mhz_wb10",
        n_ue=16,
        bandwidth_mhz=10,
        sim_duration_tti=200,
        wb_cqi_report_period_tti=10,
    ),
}

CURRICULUM_STAGES = (
    (0.00, ("train_3ue_10mhz_wb5",)),
    (0.20, ("train_3ue_10mhz_wb5", "mid_8ue_10mhz_wb5")),
    (0.45, ("train_3ue_10mhz_wb5", "mid_8ue_10mhz_wb5", "mid_16ue_10mhz_wb5")),
    (
        0.70,
        (
            "train_3ue_10mhz_wb5",
            "mid_8ue_10mhz_wb5",
            "mid_16ue_10mhz_wb5",
            "target_40ue_10mhz_wb5",
            "bw_16ue_5mhz_wb5",
            "bw_16ue_20mhz_wb5",
            "cqi_16ue_10mhz_wb1",
            "cqi_16ue_10mhz_wb10",
        ),
    ),
)

EVAL_SCENARIO_KEYS = tuple(SCENARIO_CONFIGS.keys())


class NoOpDqnPolicyRunner:
    """
    Заглушка runner-а для training runtime.

    Нужна только для инициализации `DqnScheduler` внутри `SimulationManager`.
    Во время обучения `PySchedulerLteEnv` не вызывает inference модели через
    scheduler, а использует собственный RL loop поверх runtime.
    """

    def __init__(self, max_n_ue: int):
        self.max_n_ue = int(max_n_ue)

    def predict(self, _observation, action_mask, deterministic: bool = True) -> int:
        valid_actions = np.flatnonzero(np.asarray(action_mask, dtype=bool))
        if len(valid_actions) == 0:
            return 0
        return int(valid_actions[0])


def _set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return

    resolved_seed = int(seed)
    GLOBALS.SEED = resolved_seed
    random.seed(resolved_seed)
    np.random.seed(resolved_seed)
    if torch is not None:
        torch.manual_seed(resolved_seed)


def _reset_map_borders(map_radius_m: float) -> tuple[float, float, float, float]:
    MapBorders._instance = None
    MapBorders(-map_radius_m, map_radius_m, -map_radius_m, map_radius_m)
    return -map_radius_m, map_radius_m, -map_radius_m, map_radius_m


def _apply_scenario_options(
    scenario: RuntimeTrainingScenario,
    options: Optional[Dict],
) -> RuntimeTrainingScenario:
    if not options:
        return scenario

    allowed_fields = set(RuntimeTrainingScenario.__dataclass_fields__.keys()) - {"name"}
    overrides = {key: value for key, value in options.items() if key in allowed_fields}
    if not overrides:
        return scenario

    return replace(scenario, **overrides)


def create_training_manager(
    scenario: RuntimeTrainingScenario,
    *,
    max_n_ue: int,
    seed: Optional[int] = None,
) -> SimulationManager:
    """
    Собрать `SimulationManager` для одного DRL training episode.
    """

    if int(max_n_ue) < int(scenario.n_ue):
        raise ValueError(
            f"max_n_ue={max_n_ue} must be >= scenario.n_ue={scenario.n_ue}"
        )

    _set_global_seed(seed)
    x_min, x_max, y_min, y_max = _reset_map_borders(float(scenario.map_radius_m))

    bs = BaseStation(
        x=0.0,
        y=0.0,
        bandwidth=float(scenario.bandwidth_mhz),
        ch_model_type=scenario.ch_model_type,
        use_simple_buffer=True,
        enable_tdl=bool(scenario.enable_tdl),
    )

    ue_collection = UECollection()
    ue_collection.ADD_RANDOM_USERS(
        num_ue=int(scenario.n_ue),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        ue_class=scenario.ue_class,
    )
    ue_collection.SET_MOBILITY_MODEL(scenario.mobility_model)
    ue_collection.REG_USERS_TO_BS(bs)

    sim = SimulationManager()
    sim.set_base_station(bs)
    sim.set_ue_collection(ue_collection)
    sim.set_scheduler(
        algorithm="DqnScheduler",
        max_dl_ue_tti=scenario.max_dl_ue_tti,
        pcfich=int(scenario.pcfich),
        enable_window=bool(scenario.enable_window),
        window_size=int(scenario.window_size),
        max_dl_cce_allowance=scenario.max_dl_cce_allowance,
        algorithm_kwargs={
            "dqn_policy_runner": NoOpDqnPolicyRunner(max_n_ue=max_n_ue),
            "dqn_max_n_ue": int(max_n_ue),
            "dqn_episode_len_tti": int(scenario.sim_duration_tti),
            "dqn_wb_cqi_report_period_tti": int(scenario.wb_cqi_report_period_tti),
            "dqn_deterministic": True,
            "dqn_strict_observation": True,
        },
    )

    for ue in ue_collection.GET_ALL_USERS():
        sim.setup_ue_traffic(
            ue_id=ue.UE_ID,
            model_type=scenario.traffic_model_type,
            packet_rate=int(scenario.traffic_packet_rate),
        )

    sim.set_sim_duration(int(scenario.sim_duration_tti))
    sim.set_upd_interval(int(scenario.update_interval_tti))
    sim.set_mobility_interval(int(scenario.mobility_update_interval_tti))
    sim.set_channel_interval(int(scenario.channel_update_interval_tti))
    sim.set_stats_manager(enabled=False)
    return sim


def create_inference_manager(
    scenario: RuntimeTrainingScenario,
    *,
    model_path: str | Path,
    max_n_ue: int,
    seed: Optional[int] = None,
    deterministic: bool = True,
    options: Optional[Dict] = None,
) -> SimulationManager:
    """
    Собрать `SimulationManager` для инференса обученной DQN-модели внутри PyScheduler.
    """

    resolved_scenario = _apply_scenario_options(scenario, options)
    resolved_model_path = Path(model_path)

    if not resolved_model_path.exists():
        raise FileNotFoundError(f"DQN weights not found: {resolved_model_path}")

    sim = create_training_manager(
        resolved_scenario,
        max_n_ue=max_n_ue,
        seed=seed,
    )

    algorithm_kwargs = dict(sim.sched_config.algorithm_kwargs or {})
    algorithm_kwargs.pop("dqn_policy_runner", None)
    algorithm_kwargs["dqn_model_path"] = str(resolved_model_path)
    algorithm_kwargs["dqn_deterministic"] = bool(deterministic)

    sim.set_scheduler(
        algorithm="DqnScheduler",
        max_dl_ue_tti=resolved_scenario.max_dl_ue_tti,
        pcfich=int(resolved_scenario.pcfich),
        enable_window=bool(resolved_scenario.enable_window),
        window_size=int(resolved_scenario.window_size),
        max_dl_cce_allowance=resolved_scenario.max_dl_cce_allowance,
        algorithm_kwargs=algorithm_kwargs,
    )
    return sim


def build_simulation_manager_factory(
    scenario: RuntimeTrainingScenario,
    *,
    max_n_ue: int,
    default_seed: Optional[int] = None,
) -> Callable[..., SimulationManager]:
    """
    Построить factory, совместимую с `PySchedulerLteEnv`.
    """

    def factory(seed: Optional[int] = None, options: Optional[Dict] = None) -> SimulationManager:
        resolved_seed = default_seed if seed is None else seed
        resolved_scenario = _apply_scenario_options(scenario, options)
        return create_training_manager(
            resolved_scenario,
            max_n_ue=max_n_ue,
            seed=resolved_seed,
        )

    return factory


def make_pyscheduler_lte_env(
    scenario: RuntimeTrainingScenario,
    *,
    max_n_ue: int,
    seed: Optional[int] = None,
) -> Any:
    """
    Создать simulation-backed env для заданного runtime-сценария.
    """

    from drl.envs.pyscheduler_lte_env import PySchedulerLteEnv

    if int(max_n_ue) < int(scenario.n_ue):
        raise ValueError(
            f"max_n_ue={max_n_ue} must be >= scenario.n_ue={scenario.n_ue}"
        )

    return PySchedulerLteEnv(
        simulation_factory=build_simulation_manager_factory(
            scenario,
            max_n_ue=max_n_ue,
            default_seed=seed,
        ),
        max_n_ue=max_n_ue,
        reward_mode="per_tti",
        reward_window=1,
        alpha=1.0,
        beta=2.0,
        rate_scale_bps=1e6,
        jfi_target=0.70,
        lambda_jfi=2.0,
        strict_observation=True,
        wb_cqi_report_period_tti=int(scenario.wb_cqi_report_period_tti),
    )


def build_pyscheduler_env_pool(
    scenario_keys: tuple[str, ...],
    *,
    max_n_ue: int,
    seed_base: int = 0,
) -> Dict[str, Any]:
    env_pool: Dict[str, Any] = {}
    for idx, scenario_key in enumerate(scenario_keys):
        env_pool[scenario_key] = make_pyscheduler_lte_env(
            SCENARIO_CONFIGS[scenario_key],
            max_n_ue=max_n_ue,
            seed=seed_base + idx,
        )
    return env_pool


def get_curriculum_candidates(progress: float) -> tuple[str, ...]:
    candidates = CURRICULUM_STAGES[0][1]
    for min_progress, stage_candidates in CURRICULUM_STAGES:
        if progress >= min_progress:
            candidates = stage_candidates
    return candidates


def sample_curriculum_scenario(
    rng: np.random.Generator,
    candidates: tuple[str, ...],
) -> str:
    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]
