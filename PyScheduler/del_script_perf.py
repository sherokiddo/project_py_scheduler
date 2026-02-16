import time
import tracemalloc
from collections import namedtuple
from dataclasses import dataclass, replace
from typing import Dict, List
import SCHEDULER as SCHED


# ------------------------------------------------------------
#  Параметры бенчмарка
# ------------------------------------------------------------

NUM_UE = 2000          # размер priority_list (примерно как в твоём тесте)
REPEATS = 500          # сколько раз повторяем _apply_pdsch_estimation
RB_PER_SLOT = 100      # подстроим под твой сценарий, но важно, что одинаково


# ------------------------------------------------------------
#  Альтернативные представления CQIMap
# ------------------------------------------------------------

# 1) Текущий dataclass из твоего кода
CQIMapDC = SCHED.CQIMap

# 2) namedtuple с теми же полями
NamedCQI = namedtuple("NamedCQI", ["wb_cqi", "last_wb_update", "sb_cqi", "last_sb_update"])

# 3) dict со схожей структурой
#    {"wb_cqi": int, "last_wb_update": int, "sb_cqi": List[int], "last_sb_update": int}


# ------------------------------------------------------------
#  DummyGrid и DummyScheduler, имитирующие нужный кусок API
# ------------------------------------------------------------

class DummyGrid:
    def __init__(self, rb_per_slot: int):
        self.rb_per_slot = rb_per_slot


class DummyScheduler:
    """
    Минимальная имитация SchedulerInterface, чтобы прогнать
    реалистичный _apply_pdsch_estimation‑подобный код.
    """
    def __init__(self, cqi_map: Dict[int, object]):
        self.cqi_map = cqi_map
        self.lte_grid = DummyGrid(RB_PER_SLOT)
        self.amc = SCHED.AdaptiveModulationAndCoding(self)
        self.verbose = False

    def _get_wb_cqi(self, ueid: int) -> int:
        """
        Текущая реальная логика из SchedulerInterface, но адаптируем под
        три типа: dataclass / namedtuple / dict.
        """
        entry = self.cqi_map.get(ueid)
        if entry is None:
            return 0

        # dataclass / namedtuple имеют атрибут wb_cqi
        if hasattr(entry, "wb_cqi"):
            return entry.wb_cqi

        # dict‑вариант
        return entry.get("wb_cqi", 0)

    def apply_pdsch_estimation_like(self, priority_list: List[Dict], tti: int) -> List[Dict]:
        """
        Упрощённая копия твоего _apply_pdsch_estimation, чтобы бенчить
        CQI‑доступ + AMC.GET_BITS_PER_RB + арифметику.

        Оригинал (фрагмент):

            total_rb = self.lte_grid.rb_per_slot
            estimated_rb = 0.0
            selected_ues = []
            pdsch_threshold = total_rb * 0.95

            for idx, user in enumerate(priority_list):
                ue_id = user['UE_ID']
                cqi = self._get_wb_cqi(ue_id)
                bits_per_rb = self.amc.GET_BITS_PER_RB(cqi)
                buffer_bits = user['bs_buffer_size'] * 8
                rb_needed = min(buffer_bits / (bits_per_rb * 2), total_rb)
                ...

        Здесь мы не делаем логов, просто считаем rb_needed и threshold.
        """
        total_rb = self.lte_grid.rb_per_slot
        estimated_rb = 0.0
        selected_ues = []

        pdsch_threshold = total_rb * 0.95

        for idx, user in enumerate(priority_list):
            ue_id = user["UE_ID"]

            cqi = self._get_wb_cqi(ue_id)
            if not (1 <= cqi <= 15):
                # как и в реальном коде, UE с невалидным CQI по сути будут отфильтрованы
                continue

            bits_per_rb = self.amc.GET_BITS_PER_RB(cqi)

            buffer_bits = user["bs_buffer_size"] * 8
            if bits_per_rb > 0:
                rb_needed = min(buffer_bits / (bits_per_rb * 2), total_rb)
            else:
                rb_needed = 0

            if idx == 0:
                selected_ues.append(user)
                estimated_rb += rb_needed
                continue

            if estimated_rb + rb_needed >= pdsch_threshold:
                break

            selected_ues.append(user)
            estimated_rb += rb_needed

        return selected_ues


# ------------------------------------------------------------
#  Генерация synthetic priority_list, близкого к реальному
# ------------------------------------------------------------

def build_priority_list(num_ue: int) -> List[Dict]:
    priority_list = []
    for ueid in range(1, num_ue + 1):
        # CQI будет задаваться через cqi_map, здесь только UE_ID и buffer
        buffer_size = 1500 + (ueid % 50) * 100   # ~разброс буфера
        priority_list.append({
            "UE_ID": ueid,
            "bs_buffer_size": buffer_size,
            # остальное для нас не важно, _apply_pdsch_estimation его не трогает
        })
    return priority_list


# ------------------------------------------------------------
#  Построение трёх вариантов cqi_map
# ------------------------------------------------------------

def build_cqi_map_dc(num_ue: int) -> Dict[int, CQIMapDC]:
    m = {}
    for ueid in range(1, num_ue + 1):
        cqi = (ueid % 15) + 1  # 1..15
        m[ueid] = CQIMapDC(
            wb_cqi=cqi,
            last_wb_update=0,
            sb_cqi=[],
            last_sb_update=0,
        )
    return m


def build_cqi_map_named(num_ue: int) -> Dict[int, NamedCQI]:
    m = {}
    for ueid in range(1, num_ue + 1):
        cqi = (ueid % 15) + 1
        m[ueid] = NamedCQI(
            wb_cqi=cqi,
            last_wb_update=0,
            sb_cqi=[],
            last_sb_update=0,
        )
    return m


def build_cqi_map_dict(num_ue: int) -> Dict[int, Dict]:
    m = {}
    for ueid in range(1, num_ue + 1):
        cqi = (ueid % 15) + 1
        m[ueid] = {
            "wb_cqi": cqi,
            "last_wb_update": 0,
            "sb_cqi": [],
            "last_sb_update": 0,
        }
    return m


# ------------------------------------------------------------
#  Измерения памяти и времени
# ------------------------------------------------------------

def measure_memory(label: str, build_fn):
    tracemalloc.start()
    cqi_map = build_fn(NUM_UE)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{label:12s}: current={current/1024:.1f} KiB, peak={peak/1024:.1f} KiB, count={len(cqi_map)}")
    return cqi_map


def bench_scheduler(label: str, scheduler: DummyScheduler, priority_list: List[Dict]):
    t0 = time.perf_counter()
    total_len = 0
    for _ in range(REPEATS):
        selected = scheduler.apply_pdsch_estimation_like(priority_list, tti=0)
        total_len += len(selected)
    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"{label:20s}: {dt:.4f} s, total_selected={total_len}")
    return dt, total_len


def main():
    print(f"Realistic CQI benchmark: UE={NUM_UE}, REPEATS={REPEATS}, RB_PER_SLOT={RB_PER_SLOT}")
    print("\n--- MEMORY (cqi_map variants) ---")
    cqi_map_dc = measure_memory("dataclass", build_cqi_map_dc)
    cqi_map_named = measure_memory("namedtuple", build_cqi_map_named)
    cqi_map_dict = measure_memory("dict", build_cqi_map_dict)

    priority_list = build_priority_list(NUM_UE)

    print("\n--- TIME: _apply_pdsch_estimation‑like ---")
    sch_dc = DummyScheduler(cqi_map_dc)
    sch_named = DummyScheduler(cqi_map_named)
    sch_dict = DummyScheduler(cqi_map_dict)

    bench_scheduler("dataclass", sch_dc, priority_list)
    bench_scheduler("namedtuple", sch_named, priority_list)
    bench_scheduler("dict", sch_dict, priority_list)


if __name__ == "__main__":
    main()
