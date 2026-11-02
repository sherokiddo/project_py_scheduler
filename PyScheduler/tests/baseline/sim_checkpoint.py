#!/usr/bin/env python3
"""Контрольная точка full-SIM прогона (SPEC-FEAT-SCH-008 §8.2/§9.6).

Конфигурация повторяет sim_with_manager_qos (TEST_MODULES.py :234):
UMa, enable_tdl=True, LayeredBufferManager, 5 UE RandomWaypoint, Poisson-
профили, GLOBALS.SEED=42. Отличия: длительность задаётся аргументом,
verbose-лог и StatsManager отключены (wall-clock и лог в снимок не входят),
состояние планировщика записывается обёрткой schedule().

Команды (из корня репозитория):
    python PyScheduler/tests/baseline/sim_checkpoint.py --duration 1000 \
        --out PyScheduler/tests/baseline/baselines/sim1000_qosaware.json
    # детерминизм: два запуска с одним seed обязаны дать одинаковый sha256
    python PyScheduler/tests/baseline/sim_checkpoint.py --duration 50
    python PyScheduler/tests/baseline/sim_checkpoint.py --duration 50

Снимок: sha256 полной истории allocation по TTI + финальные агрегаты
(состояние UE, буферы, сумма выданных RB). Сравнение до/после миграции —
побитовое совпадение sha256 и точное совпадение агрегатов; wall-clock
не сравнивается (раздел 8.2).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent
for p in (str(PKG_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import GLOBALS  # noqa: E402
import numpy as np  # noqa: E402
from baseline_scenarios import DEV_REVISION, env_info  # noqa: E402
from BS_MODULE import BaseStation  # noqa: E402
from MOBILITY_MODEL import MapBorders  # noqa: E402
from SIMULATION_MANAGER import SimulationManager  # noqa: E402
from UE_MODULE import UECollection  # noqa: E402

TRAFFIC_CONFIG = [
    {"ue_id": 1, "bearers": [
        {"model_type": "Poisson", "qci": 1, "packet_rate": 100},
        {"model_type": "Poisson", "qci": 5, "packet_rate": 200},
        {"model_type": "Poisson", "qci": 7, "packet_rate": 1500}]},
    {"ue_id": 2, "bearers": [
        {"model_type": "Poisson", "qci": 4, "packet_rate": 500},
        {"model_type": "Poisson", "qci": 8, "packet_rate": 1000}]},
    {"ue_id": 3, "bearers": [
        {"model_type": "Poisson", "packet_rate": 1000}]},
    {"ue_id": 4, "bearers": [
        {"model_type": "Poisson", "qci": 6, "packet_rate": 1000},
        {"model_type": "Poisson", "qci": 7, "packet_rate": 1500},
        {"model_type": "Poisson", "qci": 8, "packet_rate": 1200}]},
    {"ue_id": 5, "bearers": [
        {"model_type": "Poisson", "qci": 1, "packet_rate": 100}]},
]


class _Recorder:
    """Перехват SchedulerInterface.schedule: полная история allocation по TTI.

    Планировщик создаётся внутри start_simulation() (SIMULATION_MANAGER.py
    :1357), поэтому перехват ставится на метод класса, а не на экземпляр.
    """

    def __init__(self):
        self.history = []
        self._orig = None

    def __enter__(self):
        from SCHEDULER import SchedulerInterface
        self._orig = SchedulerInterface.schedule
        recorder = self

        def schedule(self_sched, tti, users):
            result = recorder._orig(self_sched, tti, users)
            recorder.history.append(
                [tti, {str(k): v for k, v in sorted(result["allocation"].items())}])
            return result

        SchedulerInterface.schedule = schedule
        return self

    def __exit__(self, *exc):
        from SCHEDULER import SchedulerInterface
        SchedulerInterface.schedule = self._orig
        return False


def run(duration, seed, algorithm):
    # Гигиена глобального состояния: ChannelModel хранит кэши в КЛАССОВЫХ
    # dict — SHADOW_FADING_INFO / CHANNEL_COND_INFO / O2I_INFO
    # (CHANNEL_MODEL.py :158-160). Они переживают прогон и на втором запуске
    # в том же процессе меняют ПОТРЕБЛЕНИЕ ГСЧ (o2i/SF берутся из кэша вместо
    # розыгрыша), что рассинхронизирует весь поток random-чисел.
    # На снимки не влияет: standalone-прогон всегда начинает с пустых кэшей.
    from CHANNEL_MODEL import ChannelModel
    ChannelModel.SHADOW_FADING_INFO.clear()
    ChannelModel.CHANNEL_COND_INFO.clear()
    ChannelModel.O2I_INFO.clear()

    MapBorders._instance = None
    MapBorders(-1000, 1000, -1000, 1000)
    x_min, x_max, y_min, y_max = MapBorders().get_borders()

    bs = BaseStation(x=0, y=0, bandwidth=10, ch_model_type="UMa",
                     use_simple_buffer=False, enable_tdl=True)

    ue_collection = UECollection()
    GLOBALS.SEED = seed
    np.random.seed(seed)
    ue_collection.ADD_RANDOM_USERS(num_ue=5, x_min=x_min, x_max=x_max,
                                   y_min=y_min, y_max=y_max, ue_class="random")
    ue_collection.SET_MOBILITY_MODEL("RandomWaypoint")
    ue_collection.REG_USERS_TO_BS(bs)

    sim = SimulationManager()
    sim.set_base_station(bs)
    sim.set_ue_collection(ue_collection)
    sim.set_scheduler(algorithm=algorithm)
    sim.setup_traffic_profiles(TRAFFIC_CONFIG)
    sim.set_sim_duration(duration)
    # ОБЯЗАТЕЛЬНО: на 6c0aae2 тело start_simulation() выполнимо только при
    # self._to_file (SIMULATION_MANAGER.py :1331 — весь прогон внутри
    # `if self._to_file:`); вывод уходит в output.txt в cwd.
    sim.enable_verbose_log(to_file=True)

    cwd = Path.cwd()
    tmp = Path(tempfile.mkdtemp(prefix="sim_checkpoint_"))
    os.chdir(tmp)
    try:
        with _Recorder() as rec:
            sim.start_simulation()
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    history_json = json.dumps(rec.history, sort_keys=True)
    digest = hashlib.sha256(history_json.encode("utf-8")).hexdigest()

    ue_final = []
    for ue in sorted(ue_collection.users.values(), key=lambda u: u.UE_ID):
        ue_final.append({
            "ue_id": int(ue.UE_ID),
            "average_throughput": ue.average_throughput,
            "position": [ue.position[0], ue.position[1]],
        })

    return {
        "meta": {
            "checkpoint": f"sim{duration}_{algorithm.lower()}",
            "algorithm": algorithm,
            "dev_revision": DEV_REVISION,
            "duration_tti": duration,
            "seed": seed,
            "config": ("sim_with_manager_qos profile (TEST_MODULES.py :234); "
                       "verbose log и StatsManager отключены"),
            "compare_rules": ("sha256 истории allocation — побитово; агрегаты — "
                              "точно для дискретных, rel tol 1e-9 для float; "
                              "wall-clock не сравнивается (SPEC 8.2)"),
            "env": env_info(),
        },
        "allocation_history_sha256": digest,
        "tti_count_recorded": len(rec.history),
        "total_rb_allocated": sum(
            len(rbs) for _tti, alloc in rec.history
            for rbs in alloc.values()),
        "buffer_stats": bs.buffer_manager.get_stats(),
        "ue_final": ue_final,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--algorithm", default="QosAware")
    ap.add_argument("--out", default=None,
                    help="файл снимка (по умолчанию — печать в stdout)")
    args = ap.parse_args()

    snapshot = run(args.duration, args.seed, args.algorithm)
    text = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[OK] {args.out}: sha256={snapshot['allocation_history_sha256'][:16]}… "
              f"tti={snapshot['tti_count_recorded']} rb={snapshot['total_rb_allocated']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
