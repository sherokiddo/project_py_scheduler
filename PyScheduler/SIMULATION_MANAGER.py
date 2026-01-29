"""
#------------------------------------------------------------------------------
# Модуль: SIMULATION_MANAGER - Менеджер управления симуляцией.
#------------------------------------------------------------------------------
# Описание:
# Отвечает за настройку параметров симуляции, запуск основного цикла симуляции,
# а также логирование статистик в файл.
#
# Изменения v1.1.0:
# - Обновлен планировщик RR в связи с изменением принципа работы буфера
# и изменения UE_MODULE.
# - Добавлен модуль AMC
# - Добавлены тесты планировщика
# - По итогам тестов оказалось, что АМС не работает полноценно. Это печально.
# - Для дальнейшей коррекции работы, необходимо ввести фрагментацию пакетов.
#
# Версия: 1.1.0
# Дата последнего изменения: 2025-12-28
# Автор: Норицин Иван, Брагин Кирилл
# Версия Python Kernel: 3.12.9
#------------------------------------------------------------------------------
"""

import csv
import sys
import warnings
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional

import GLOBALS
import numpy as np
from BS_MODULE import BaseStation
from RES_GRID import RES_GRID_LTE
from SCHEDULER import SchedulerInterface
from TRAFFIC_MODEL import PacketManager, SimpleGenerator, TrafficType
from UE_MODULE import UECollection

# ==============================================================================
#                          ОБРАБОТЧИК СТАТИСТИКИ
# ==============================================================================


class MetricLevel(IntEnum):
    """
    Уровни детализации сбора метрик.

    - NONE: Не собирать (0% overhead)
    - BASIC: Только собираемые метрики (total, avg)
    - DETAILED: Собираемые метрики и расчетные (per-UE Dict, Jain's e.t.c)
    """

    NONE = 0
    BASIC = 1
    ADVANCED = 2
    FULL = 3


@dataclass
class LevelsConfig:
    """
    Конфигурация уровней детализации для каждого типа метрик.

    Attributes:
        scheduler (MetricLevel): Уровень для scheduler.get_stats()
        amc (MetricLevel): Уровень для amc.get_stats()
        pdcch (MetricLevel): Уровень для pdcch_manager.get_stats()
    """

    scheduler: MetricLevel = MetricLevel.BASIC
    amc: MetricLevel = MetricLevel.BASIC
    pdcch: MetricLevel = MetricLevel.BASIC


@dataclass
class StatisticsConfig:
    """
    Основная конфигурация системы статистики.

    Attributes:
        collect_interval (int): Интервал сбора в TTI (напр., 10 = каждые 10 TTI)
        levels (LevelsConfig): Уровни детализации per-метрика
        export_format (str): Формат экспорта ('csv' или 'json')
        history_max_len (int): Максимум snapshots в истории (для ограничения памяти)
        file_prefix (str): Префикс имени файла для экспорта
    """

    collect_interval: int = 10
    levels: LevelsConfig = None
    export_format: str = "csv"
    export_detailed_format: str = "csv"
    history_max_len: int = 10000
    file_prefix: str = "lte_stats"

    def __post_init__(self):
        """Инициализация levels, если не передан"""
        if self.levels is None:
            self.levels = LevelsConfig()

        if self.export_detailed_format not in ["csv", "json"]:
            raise ValueError(f"Invalid export_detailed_format: {self.export_detailed_format}")


class StatsManager:
    """
    Менеджер сбора и экспорта статистики LTE-симуляции.
    Собирает метрики с scheduler/amc/pdcch по настраиваемому интервалу,
    хранит историю и экспортирует в CSV/JSON.
    """

    def __init__(self, scheduler, config: StatisticsConfig = None):
        """
        Инициализация менеджера статистики.

        Args:
            scheduler (SchedulerInterface): Планировщик для опроса
            config (StatisticsConfig): Конфигурация (если None, используется default)
        """
        self.scheduler = scheduler  # Ссылка на scheduler (для вызова get_stats())
        self.config = config or StatisticsConfig()  # Default конфиг если None
        self.history = deque(maxlen=self.config.history_max_len)  # История snapshots
        self.detailed_history = deque(maxlen=self.config.history_max_len)

    def _validate_sched_stats(self, sched_stats: dict, tti: int) -> bool:
        """Валидация что данные sched_stats полноценные и корректные"""
        import warnings

        required_fields = [
            "tti",
            "sch_eligible_ue_count",
            "sch_active_ue_count",
            "dl_rb_allocated_count",
        ]

        for field in required_fields:
            if field not in sched_stats:
                warnings.warn(
                    f"[StatsManager] TTI {tti}: Missing required field '{field}' in sched_stats",
                    UserWarning,
                )
                return False

        if sched_stats["tti"] != tti:
            warnings.warn(
                f"[StatsManager] TTI mismatch at {tti}: sched_stats has TTI {sched_stats['tti']}",
                UserWarning,
            )
            return False

        return True

    def collect(self, tti: int) -> None:
        """
        Собрать метрики на текущем TTI.
        Опрашивает scheduler.get_stats() и amc.get_stats() в зависимости от
        уровней в config.levels, объединяет в один Dict (snapshot) и добавляет
        в self.history.

        Args:
            tti (int): Номер текущего TTI (для timestamp)
        """

        # Получить метрики из источников
        sched_stats = (
            self.scheduler.get_stats() if self.config.levels.scheduler != MetricLevel.NONE else {}
        )
        amc_stats = (
            self.scheduler.amc.get_stats() if self.config.levels.amc != MetricLevel.NONE else {}
        )
        pdcch_stats = (
            self.scheduler.pdcch_manager.get_stats()
            if self.config.levels.pdcch != MetricLevel.NONE
            else {}
        )

        # TTI validation
        if sched_stats and "tti" in sched_stats:
            if sched_stats["tti"] != tti:
                warnings.warn(
                    f"[StatsManager] TTI mismatch: expected {tti}, "
                    f"got {sched_stats['tti']}. Skipping collection."
                )
                return  # Пропустить невалидный snapshot

        # Проверка 2: Scheduler stats должны существовать (если level != NONE)
        if self.config.levels.scheduler != MetricLevel.NONE:
            if not sched_stats or "dl_rb_allocated_count" not in sched_stats:
                warnings.warn(
                    f"[StatsManager] Invalid sched_stats at TTI {tti}. Skipping collection."
                )
                return

        if sched_stats and not self._validate_sched_stats(sched_stats, tti):
            return  # Skip

        # Continue with snapshot creation
        snapshot = {
            "tti": tti,
            # BASIC метрики
            # Временная метка и базовые счетчики
            "sch_eligible_ue_count": sched_stats.get("sch_eligible_ue_count", 0),
            "sch_active_ue_count": sched_stats.get("sch_active_ue_count", 0),
            # Scheduler метрики (группа RB)
            "dl_rb_allocated_count": sched_stats.get("dl_rb_allocated_count", 0),
            "dl_rb_per_ue_avg": sched_stats.get("dl_rb_per_ue_avg", 0.0),
            # AMC метрики (группа throughput/bits)
            "dl_bits_per_rb_avg": amc_stats.get("dl_bits_per_rb_avg", 0.0),
            "dl_capacity_bits_sum_tti": amc_stats.get("dl_capacity_bits_sum_tti", 0),
            "dl_transmitted_bits_sum_tti": amc_stats.get("dl_transmitted_bits_sum_tti", 0),
            "dl_throughput_sum_kbps": amc_stats.get("dl_throughput_sum_kbps", 0.0),
            "dl_cqi_wb_avg_idx": amc_stats.get("dl_cqi_wb_avg_idx", 0.0),
            "dl_sinr_avg": amc_stats.get("dl_sinr_avg", 0.0),
            # Buffer метрики
            "buffer_size_sum_bytes": sched_stats.get("buffer_size_sum_bytes", 0),
            # Time метрики
            "sch_total_time_us": sched_stats.get("sch_total_time_us", 0.0),
            # "sch_priority_calc_time_us": sched_stats.get("sch_priority_calc_time_us", 0.0),
            # "sch_priority_sort_time_us": sched_stats.get("sch_priority_sort_time_us", 0.0),
            # PDCCH метрики (группа CCE)
            "pdcch_cce_total_count": pdcch_stats.get("pdcch_cce_total_count", 0),
            "pdcch_cce_allocated_count": pdcch_stats.get("pdcch_cce_allocated_count", 0),
        }

        if self.config.levels.pdcch >= MetricLevel.ADVANCED:
            snapshot.update(
                {"pdcch_cce_utilization_pct": pdcch_stats.get("pdcch_cce_utilization_pct", 0.0)}
            )

        if self.config.levels.scheduler >= MetricLevel.ADVANCED:
            snapshot.update(
                {
                    # Timing детализация
                    "sch_priority_calc_time_us": sched_stats.get("sch_priority_calc_time_us", 0.0),
                    "sch_priority_sort_time_us": sched_stats.get("sch_priority_sort_time_us", 0.0),
                    "dl_prb_utilization_pct": sched_stats.get("dl_prb_utilization_pct", 0.0),
                    "sch_priority_list_size": sched_stats.get("sch_priority_list_size", 0),
                    "sch_pdcch_blocked_count": sched_stats.get("sch_pdcch_blocked_count", 0),
                    "sch_avg_priority_value": sched_stats.get("sch_avg_priority_value", 0.0),
                    "sch_eligible_to_active_ratio": self._calculate_ratio(
                        sched_stats.get("sch_active_ue_count", 0),
                        sched_stats.get("sch_eligible_ue_count", 0),
                    ),
                }
            )

        priority_list = sched_stats.get("sch_priority_list", [])
        if priority_list:
            priority_values = [u.get("priority", 0) for u in priority_list]
            snapshot.update(
                {
                    "sch_priority_min": round(min(priority_values), 4),
                    "sch_priority_max": round(max(priority_values), 4),
                    "sch_priority_std": self._calculate_std(priority_values),
                }
            )
        else:
            snapshot.update(
                {
                    "sch_priority_min": 0.0,
                    "sch_priority_max": 0.0,
                    "sch_priority_std": 0.0,
                }
            )

        if self.config.levels.amc >= MetricLevel.ADVANCED:
            snapshot.update(
                {
                    "dl_capacity_bits_sum_tti": amc_stats.get("dl_capacity_bits_sum_tti", 0),
                }
            )
            ue_cqi = amc_stats.get("ue_cqi", {})
            ue_sinr = amc_stats.get("ue_sinr", {})
            ue_throughputs = amc_stats.get("dl_ue_throughputs", {})

            # Throughput min/max/std
            snapshot.update(
                self._safe_stats_from_dict(
                    ue_throughputs, "dl_throughput_kbps", scale=1000.0, filter_zeros=True
                )
            )

            # CQI min/max/std
            snapshot.update(
                self._safe_stats_from_dict(
                    ue_cqi,
                    "dl_cqi_wb_idx",
                    precision=0,  # CQI - целое число
                    filter_zeros=False,  # CQI=0 тоже валидное значение
                )
            )

            # SINR min/max/std
            snapshot.update(
                self._safe_stats_from_dict(
                    ue_sinr,
                    "dl_sinr",
                    precision=2,
                    filter_zeros=False,  # SINR может быть отрицательным
                )
            )

            rb_efficiency = self._calculate_rb_efficiency(
                capacity_bits=amc_stats.get("dl_capacity_bits_sum_tti", 0),
                transmitted_bits=amc_stats.get("dl_transmitted_bits_sum_tti", 0),
                allocated_rbs=sched_stats.get("dl_rb_allocated_count", 0),
                total_rbs=self.scheduler.lte_grid.rb_per_slot,
            )
            snapshot.update(rb_efficiency)

            # Spectral Efficiency
            ue_throughputs = amc_stats.get("dl_ue_throughputs", {})
            se_metrics = self._calculate_spectral_efficiency(
                throughput_sum_bps=amc_stats.get("dl_throughput_sum_kbps", 0.0) * 1000,
                ue_throughputs=ue_throughputs,
            )
            snapshot.update(se_metrics)

            # Coefficient of Variation
            cv_metrics = self._calculate_cv_metrics(
                ue_throughputs=ue_throughputs,
                ue_cqi=amc_stats.get("ue_cqi", {}),
                ue_sinr=amc_stats.get("ue_sinr", {}),
                ue_rb_allocated=amc_stats.get("ue_rb_allocated", {}),
            )
            snapshot.update(cv_metrics)

            # Fairness
            fairness_metrics = self._calculate_fairness(ue_throughputs)
            snapshot.update(fairness_metrics)

        self.history.append(snapshot)

        if (
            self.config.levels.scheduler >= MetricLevel.FULL
            or self.config.levels.amc >= MetricLevel.FULL
        ):
            detailed = self._collect_detailed_metrics(
                tti=tti,
                pdcch_stats=pdcch_stats,
                sched_stats=sched_stats,
                amc_stats=amc_stats,
            )
            self.detailed_history.append(detailed)

    def _collect_detailed_metrics(
        self, tti: int, sched_stats: dict, amc_stats: dict, pdcch_stats: dict
    ) -> dict:
        """
        Собрать детальную per-UE статистику для FULL level.

        Returns:
            Dict в формате:
            {
                "tti": 100,
                "ue_metrics": {
                    "1": {"priority": 0.95, "cqi": 15, ...},
                    "2": {"priority": 0.67, "cqi": 12, ...}
                }
            }
        """
        detailed = {"tti": tti, "ue_metrics": {}}

        # Priority list (если SCHEDULER == FULL)
        priority_list = sched_stats.get("sch_priority_list", [])
        ue_priorities = {u["UE_ID"]: u["priority"] for u in priority_list}

        ue_buffer_sizes = sched_stats.get("ue_buffer_sizes", {})
        ue_transmitted_bits = sched_stats.get("ue_transmitted_bits", {})

        # Per-UE данные (если AMC == FULL)
        ue_throughputs = amc_stats.get("dl_ue_throughputs", {})
        ue_cqi = amc_stats.get("ue_cqi", {})
        ue_sinr = amc_stats.get("ue_sinr", {})
        ue_rb = amc_stats.get("ue_rb_allocated", {})
        ue_cce_alloc = pdcch_stats.get("pdcch_ue_cce_allocations", {})

        # Собрать все unique UE IDs
        all_ue_ids = (
            set(ue_priorities.keys())
            | set(ue_throughputs.keys())
            | set(ue_cqi.keys())
            | set(ue_sinr.keys())
            | set(ue_rb.keys())
            | set(ue_buffer_sizes.keys())
            | set(ue_transmitted_bits.keys())
            | set(ue_cce_alloc.keys())
        )

        for ue_id in all_ue_ids:
            detailed["ue_metrics"][str(ue_id)] = {
                "priority": ue_priorities.get(ue_id, 0.0),
                "cqi": ue_cqi.get(ue_id, 0),
                "sinr": ue_sinr.get(ue_id, 0.0),
                "rb_allocated": ue_rb.get(ue_id, 0),
                "cce_allocated": ue_cce_alloc.get(ue_id, 0),
                "throughput_bps": ue_throughputs.get(ue_id, 0),
                "buffer_size_bytes": ue_buffer_sizes.get(ue_id, 0),
                "bits_transmitted": ue_transmitted_bits.get(ue_id, 0),
            }

        return detailed

    def _safe_stats_from_dict(
        self,
        values_dict: dict,
        prefix: str,
        precision: int = 2,
        scale: float = 1.0,
        filter_zeros: bool = True,
    ) -> dict:
        """Универсально считает min/max/avg/std из словаря значений."""

        # ЭТАП 1: Проверка пустого входа
        if not values_dict:
            return {
                f"{prefix}_min": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_avg": 0.0,
                f"{prefix}_std": 0.0,
            }

        # ЭТАП 2: Извлечь values из dict и применить фильтр
        if filter_zeros:
            values = [v / scale for v in values_dict.values() if v > 0]
        else:
            values = [v / scale for v in values_dict.values()]

        # ЭТАП 3: Проверка после фильтрации
        if not values:
            return {
                f"{prefix}_min": 0.0,
                f"{prefix}_max": 0.0,
                f"{prefix}_avg": 0.0,
                f"{prefix}_std": 0.0,
            }

        # ЭТАП 4: Рассчитать min/max/avg
        min_val = min(values)
        max_val = max(values)
        avg_val = sum(values) / len(values)

        # ЭТАП 5: Рассчитать std (standard deviation)
        variance = sum((x - avg_val) ** 2 for x in values) / len(values)
        std_val = variance**0.5

        # ЭТАП 6: Вернуть результат с округлением
        return {
            f"{prefix}_min": round(min_val, precision),
            f"{prefix}_max": round(max_val, precision),
            f"{prefix}_avg": round(avg_val, precision),
            f"{prefix}_std": round(std_val, precision),
        }

    def _calculate_ratio(self, active: int, eligible: int) -> float:
        """Рассчитать соотношение active/eligible (%)."""
        if eligible > 0:
            return round((active / eligible) * 100, 2)
        return 0.0  # Если eligible = 0, ratio = 0%

    def _calculate_rb_efficiency(
        self, capacity_bits: int, transmitted_bits: int, allocated_rbs: int, total_rbs: int
    ) -> dict:
        """
        Рассчитать метрики эффективности использования RB.

        Args:
            capacity_bits: Максимальная емкость выделенных RB (из CQI)
            transmitted_bits: Фактически переданные биты
            allocated_rbs: Число выделенных RB
            total_rbs: Всего RB в системе
        Returns:
            Dict с метриками:
                - dl_rb_utilization_pct: Эффективность использования capacity (%)
                - dl_rb_wasted_count: Недоиспользованные RB
                - dl_rb_idle_count: Невыделенные RB
        """
        # RB utilization (фактически переданные биты / capacity)
        if capacity_bits > 0:
            rb_utilization_pct = (transmitted_bits / capacity_bits) * 100
        else:
            rb_utilization_pct = 0.0

        # Wasted RB (выделены но недоиспользованы из-за пустого буфера)
        if capacity_bits > 0:
            wasted_capacity_ratio = 1 - (transmitted_bits / capacity_bits)
            wasted_rbs = int(allocated_rbs * wasted_capacity_ratio)
        else:
            wasted_rbs = 0

        # Idle RB (не выделены вообще)
        idle_rbs = total_rbs - allocated_rbs

        return {
            "dl_rb_utilization_pct": round(rb_utilization_pct, 2),
            "dl_rb_wasted_count": wasted_rbs,
            "dl_rb_idle_count": idle_rbs,
        }

    def _calculate_spectral_efficiency(
        self, throughput_sum_bps: float, ue_throughputs: dict
    ) -> dict:
        """
        Рассчитать спектральную эффективность (SE).

        SE = Throughput (bps) / Bandwidth (Hz)

        Args:
            throughput_sum_bps: Суммарный throughput системы (bps)
            ue_throughputs: Dict[ue_id -> throughput_bps]
        Returns:
            Dict с метриками SE (bps/Hz):
                - dl_spectral_efficiency_bps_hz: Общая SE системы
                - dl_spectral_efficiency_avg_ue: Средняя SE на UE
                - dl_spectral_efficiency_peak: Пиковая SE (лучший UE)
        """
        bandwidth_mhz = self.scheduler.lte_grid.bandwidth
        bandwidth_hz = bandwidth_mhz * 1_000_000  # МГц → Гц

        if bandwidth_hz == 0:
            return {
                "dl_spectral_efficiency_bps_hz": 0.0,
                "dl_spectral_efficiency_avg_ue": 0.0,
                "dl_spectral_efficiency_peak": 0.0,
            }

        # Общая SE системы
        se_total = throughput_sum_bps / bandwidth_hz

        # SE per-UE
        if ue_throughputs:
            active_throughputs = [tp for tp in ue_throughputs.values() if tp > 0]

            if active_throughputs:
                se_per_ue = [tp / bandwidth_hz for tp in active_throughputs]
                se_avg_ue = sum(se_per_ue) / len(se_per_ue)
                se_peak = max(se_per_ue)
            else:
                se_avg_ue = 0.0
                se_peak = 0.0
        else:
            se_avg_ue = 0.0
            se_peak = 0.0

        return {
            "dl_spectral_efficiency_bps_hz": round(se_total, 4),
            "dl_spectral_efficiency_avg_ue": round(se_avg_ue, 4),
            "dl_spectral_efficiency_peak": round(se_peak, 4),
        }

    def _calculate_cv(self, values: list) -> float:
        """
        Рассчитать коэффициент вариации (CV). Любой

        Args:
            values: Список значений
        Returns:
            CV в процентах (0-100+)
        """
        if not values or len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0

        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance**0.5
        cv = (std / mean) * 100

        return round(cv, 2)

    def _calculate_cv_metrics(
        self, ue_throughputs: dict, ue_cqi: dict, ue_sinr: dict, ue_rb_allocated: dict
    ) -> dict:
        """
        Рассчитать CV метрики для различных параметров.

        Args:
            ue_throughputs: Dict[ue_id -> throughput_bps]
            ue_cqi: Dict[ue_id -> cqi]
            ue_sinr: Dict[ue_id -> sinr]
            ue_rb_allocated: Dict[ue_id -> rb_count]
        Returns:
            Dict с CV метриками (%)
        """
        cv_throughput = (
            self._calculate_cv([tp for tp in ue_throughputs.values() if tp > 0])
            if ue_throughputs
            else 0.0
        )
        cv_cqi = self._calculate_cv([cqi for cqi in ue_cqi.values() if cqi > 0]) if ue_cqi else 0.0
        cv_sinr = (
            self._calculate_cv([sinr for sinr in ue_sinr.values() if sinr > 0]) if ue_sinr else 0.0
        )
        cv_rb = (
            self._calculate_cv([rb for rb in ue_rb_allocated.values() if rb > 0])
            if ue_rb_allocated
            else 0.0
        )
        cv_se = 0.0

        # CV для SE (рассчитать SE per-UE, затем CV)
        if ue_throughputs:
            bandwidth_hz = self.scheduler.lte_grid.bandwidth * 1_000_000
            if bandwidth_hz > 0:
                se_per_ue = [tp / bandwidth_hz for tp in ue_throughputs.values() if tp > 0]
                if se_per_ue:
                    cv_se = self._calculate_cv(se_per_ue)
                else:
                    cv_se = 0.0
            else:
                cv_se = 0.0

        return {
            "dl_throughput_cv_pct": cv_throughput,
            "dl_cqi_cv_pct": cv_cqi,
            "dl_sinr_cv_pct": cv_sinr,
            "dl_rb_allocation_cv_pct": cv_rb,
            "dl_spectral_efficiency_cv_pct": cv_se,
        }

    def _calculate_fairness(self, ue_throughputs: dict) -> dict:
        """
        Рассчитать Fairness метрики.
        Jain's Fairness Index = (Σx_i)² / (n × Σx_i²)
        Значение от 0 до 1, где 1 = идеальная справедливость.

        Args:
            ue_throughputs: Dict[ue_id -> throughput_bps]
        Returns:
            Dict с метриками:
                - dl_fairness_jain_index: Jain's Fairness Index [0-1]
                - dl_throughput_variance: Дисперсия throughput
                - dl_throughput_std: Стандартное отклонение
        """
        if not ue_throughputs or len(ue_throughputs) < 2:
            return {
                "dl_fairness_jain_index": 1.0,
                "dl_throughput_variance": 0.0,
                "dl_throughput_std": 0.0,
            }

        throughputs = list(ue_throughputs.values())
        n = len(throughputs)

        # Jain's Fairness Index
        sum_throughput = sum(throughputs)
        sum_squared = sum(x**2 for x in throughputs)

        if sum_squared > 0:
            jain_index = (sum_throughput**2) / (n * sum_squared)
        else:
            jain_index = 1.0

        # Variance and Std
        mean = sum_throughput / n
        variance = sum((x - mean) ** 2 for x in throughputs) / n
        std = variance**0.5

        return {
            "dl_fairness_jain_index": round(jain_index, 4),
            "dl_throughput_variance": round(variance, 2),
            "dl_throughput_std": round(std, 2),
        }

    def _calculate_std(self, values: list) -> float:
        """
        Рассчитать стандартное отклонение.

        Args:
            values: Список значений
        Returns:
            Стандартное отклонение (float)
        """
        if not values or len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = variance**0.5

        return round(std, 2)

    def export_csv(self, filename: str = None, locale: str = "ru") -> None:
        """
        Экспорт истории snapshots в CSV файл.

        Вложенные Dict (например, ue_throughputs, rb_distribution) сериализуются
        в JSON-строки для совместимости с CSV.

        Args:
            filename (str): Имя файла (если None, используется config.file_prefix + '.csv')
        """
        if not self.history:
            print("StatsManager: No data to export (history is empty)")
            return

        if filename is None:
            filename = f"{self.config.file_prefix}.csv"

        import json

        first_snapshot = self.history[0]
        headers = list(first_snapshot.keys())
        delimiter = ";" if locale == "ru" else ","
        rows = []
        for snapshot in self.history:
            row = {}
            for key, value in snapshot.items():
                if isinstance(value, dict):
                    json_str = json.dumps(value, ensure_ascii=False)
                    if locale == "ru":
                        json_str = json_str.replace(".", ",")
                    row[key] = json_str
                elif isinstance(value, float):
                    if locale == "ru":
                        row[key] = str(value).replace(".", ",")
                    else:
                        row[key] = value
                else:
                    row[key] = value
            rows.append(row)

        if locale == "ru":
            delimiter = ";"
        else:
            delimiter = ","

        # Запись в CSV
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)

        print(f"StatsManager: Exported {len(rows)} snapshots to {filename}")

    def export_detailed_csv(self, filename: str = None, locale: str = "ru") -> None:
        """
        Экспортировать детальную статистику в CSV (per-UE rows).

        Формат: tti, ue_id, priority, cqi, sinr, rb_allocated, throughput_bps
        """
        if not self.detailed_history:
            print("[StatsManager] No detailed data to export (detailed_history is empty)")
            return

        if filename is None:
            filename = f"{self.config.file_prefix}_detailed.csv"

        if locale == "ru":
            delimiter = ";"
        else:
            delimiter = ","

        with open(filename, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "tti",
                "ue_id",
                "priority",
                "cqi",
                "sinr",
                "rb_allocated",
                "cce_allocated",
                "throughput_bps",
                "buffer_size_bytes",
                "bits_transmitted",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()

            for snapshot in self.detailed_history:
                tti = snapshot["tti"]
                for ue_id, metrics in snapshot["ue_metrics"].items():
                    priority = metrics.get("priority", 0.0)
                    sinr = metrics.get("sinr", 0.0)

                    if locale == "ru":
                        priority_str = str(priority).replace(".", ",")
                        sinr_str = str(sinr).replace(".", ",")
                    else:
                        priority_str = priority
                        sinr_str = sinr

                    row = {
                        "tti": tti,
                        "ue_id": ue_id,
                        "priority": priority_str,
                        "cqi": metrics.get("cqi", 0),
                        "sinr": sinr_str,
                        "rb_allocated": metrics.get("rb_allocated", 0),
                        "cce_allocated": metrics.get("cce_allocated", 0),
                        "throughput_bps": metrics.get("throughput_bps", 0),
                        "buffer_size_bytes": metrics.get("buffer_size_bytes", 0),
                        "bits_transmitted": metrics.get("bits_transmitted", 0),
                    }
                    writer.writerow(row)

        print(
            f"[StatsManager] Exported {len(self.detailed_history)} TTI snapshots "
            f"to {filename} (per-UE format, locale={locale})"
        )

    def export_detailed_json(self, filename: str = None) -> None:
        """
        Экспортировать детальную статистику в JSON.

        Формат: {tti: {ue_id: {metrics}}}
        """
        if not self.detailed_history:
            print("[StatsManager] No detailed data to export (detailed_history is empty)")
            return

        if filename is None:
            filename = f"{self.config.file_prefix}_detailed.json"

        import json

        # Конвертировать в dict для JSON
        data = {str(snapshot["tti"]): snapshot["ue_metrics"] for snapshot in self.detailed_history}

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(
            f"[StatsManager] Exported {len(self.detailed_history)} TTI snapshots "
            f"to {filename} (JSON format)"
        )


# ==============================================================================
#                          КОНФИГУРАТОР СИМУЛЯЦИИ
# ==============================================================================


@dataclass
class SimulationConfig:
    """
    Конфигурация симуляции.

    Attributes:
        sim_duration (Optional[int]): Длительность симуляции (мс).
        update_interval (int): Интервал обновления состояния пользователей (мс).
        stats_log (bool): Включение логирования статистики в CSV-файл.
        verbose (bool): Включение подробного verbose логирования.

    """

    sim_duration: Optional[int] = None
    update_interval: int = 1
    stats_log: bool = False
    verbose: bool = False


@dataclass
class SchedulerConfig:
    """Конфигурация планировщика.

    Attributes:
        algorithm (Optional[str]): Название алгоритма планирования.
        max_dl_ue_tti (Optional[int]): Максимум UE за TTI.
        pcfich (int): Значаение PCFICH.
        max_dl_cce_allowance (Optional[int]): Максимальное число CCE для PDCCH.
        window_size (int): Размер скользящего окна.
        enable_window (bool): Включение скользящего окна.

    """

    algorithm: Optional[str] = None
    max_dl_ue_tti: Optional[int] = None
    pcfich: int = 2
    max_dl_cce_allowance: Optional[int] = None
    window_size: int = 100
    enable_window: bool = True


@dataclass
class StatsManagerConfig:
    """
    Конфигурация системы статистики.

    Attributes:
        enabled (bool): Включить StatsManager (False = отключить сбор)
        collect_interval (int): Интервал сбора в TTI
        scheduler_level (str): Уровень для scheduler ('none'/'basic'/'advanced'/'full')
        amc_level (str): Уровень для AMC
        pdcch_level (str): Уровень для PDCCH
        export_format (str): Формат экспорта ('csv' или 'json')
        file_prefix (str): Префикс файла для экспорта
    """

    enabled: bool = True
    collect_interval: int = 10
    scheduler_level: str = "basic"
    amc_level: str = "basic"
    pdcch_level: str = "none"
    export_format: str = "csv"
    export_detailed_format: str = "csv"
    file_prefix: str = "manager_stats"
    history_max_len: int = 10000


class SimulationManager:
    """
    Менеджер управления симуляцией.

    Реализует настройку и запуск симуляции.

    """

    def __init__(self):
        """
        Инициализация менеджера симуляции.

        Создаёт объекты конфигураций, а также контейнеры для базовой
        станции и коллекции пользователей.

        """
        self.sim_config = SimulationConfig()
        self.sched_config = SchedulerConfig()
        self.stats_config = StatsManagerConfig()

        self.ue_collection = None
        self.base_station = None
        self.stats_manager = None

        self._to_file = False
        self._log_file = None
        self._original_stdout = None
        self._original_stderr = None
        self.scheduler = None
        self.tti_duration = 1
        self.use_legacy_traffic = getattr(self.sim_config, "use_legacy_traffic", True)

        if self.use_legacy_traffic:
            self.traffic_gen = SimpleGenerator(default_qci=9)
        else:
            self.traffic_gen = PacketManager()

    def set_sim_duration(self, sim_duration: int) -> None:
        """
        Установить длительности всей симуляции.

        Args:
            sim_duration (int): Длительность симуляции (мс).

        Raises:
            ValueError: Если значение не является положительным целым числом.

        """
        if not isinstance(sim_duration, int) or sim_duration <= 0:
            raise ValueError(
                f"Значение длительности симуляции должно быть положительным "
                f"целым числом. Получено: {sim_duration} "
                f"({type(sim_duration).__name__})"
            )

        self.sim_config.sim_duration = sim_duration

    def set_upd_interval(self, update_interval: int) -> None:
        """
        Установить интервал обновления состояний пользователей.

        Args:
            update_interval (int): Интервал обновления (мс).

        Raises:
            ValueError: Если значение не является положительным целым числом.

        """
        if not isinstance(update_interval, int) or update_interval <= 0:
            raise ValueError(
                f"Значение интервала обновления должно быть положительным "
                f"целым числом. Получено: {update_interval} "
                f"({type(update_interval).__name__})"
            )

        self.sim_config.update_interval = update_interval

    def set_ue_collection(self, ue_collection: UECollection) -> None:
        """
        Установить коллекцию пользователей.

        Args:
            ue_collection (UECollection): Коллекция пользователей.

        Raises:
            TypeError: Если передан объект неверного типа.

        """
        if not isinstance(ue_collection, UECollection):
            raise TypeError(
                f"Коллекция пользователей должна быть типа UECollection. "
                f"Получено: {type(ue_collection).__name__}"
            )

        self.ue_collection = ue_collection

    def set_base_station(self, base_station: BaseStation) -> None:
        """
        Установить базовую станцию.

        Args:
            base_station (BaseStation): Экземпляр базовой станции.

        Raises:
            TypeError: Если передан объект неверного типа.

        """
        if not isinstance(base_station, BaseStation):
            raise TypeError(
                f"Экземпляр базовой станции должен быть типа BaseStation. "
                f"Получено: {type(base_station).__name__}"
            )

        self.base_station = base_station

    def set_scheduler(self, algorithm: str, **kwargs) -> None:
        """
        Установить настройки алгоритма планирования ресурсов.

        Args:
            algorithm (str): Название алгоритма планирования.
            **kwargs: Параметры планировщика (max_dl_ue_tti, verbose, etc.)

        Raises:
            TypeError: Если 'algorithm' не является строкой.
            ValueError: Получен недопустимый параметр настройки планировщика.

        """
        if not isinstance(algorithm, str):
            raise TypeError(
                f"Название алгоритма планирования должно быть типа str. "
                f"Получено: {type(algorithm).__name__}"
            )

        self.sched_config.algorithm = algorithm
        for key, value in kwargs.items():
            if hasattr(self.sched_config, key):
                setattr(self.sched_config, key, value)
            else:
                raise ValueError(
                    f"Недопустимый параметр '{key}' для настройки планировщика. "
                    f"Доступные параметры: "
                    f"{', '.join(self.sched_config.__dataclass_fields__.keys())}"
                )

    def set_stats_manager(self, **kwargs) -> None:
        """
        Настроить StatsManager.

        Args:
            **kwargs: Параметры StatsManagerConfig (enabled, collect_interval, etc.)

        Raises:
            ValueError: Если ключ не существует в StatsManagerConfig

        Example:
            manager.set_stats_manager(enabled=True, collect_interval=5, amc_level="detailed")
        """
        for key, value in kwargs.items():
            if hasattr(self.stats_config, key):
                setattr(self.stats_config, key, value)
            else:
                available = ", ".join(self.stats_config.__dataclass_fields__.keys())
                raise ValueError(
                    f"'{key}' не является параметром StatsManagerConfig. Доступные: {available}"
                )

    def enable_verbose_log(self, to_file: bool = False):
        """
        Включить подробное логирование симуляции (verbose).

        Args:
            to_file (bool, optional): Флаг, отвечающий за перевод консольного
            вывода в текстовый файл (output.txt). По умолчанию False.

        """
        self.sim_config.verbose = True
        self._to_file = to_file

    def start_simulation(self) -> None:
        """
        Запуск основной симуляции.

        Основные этапы:
            1. Проверка конфигурации.
            2. Создание ресурсной сетки (RES_GRID_LTE).
            3. Инициализация планировщика.
            4. Инициализация менеджера статистики (есть есть).
            4. Основной цикл по времени (TTI):
                - Периодическое обновление состояния пользователей.
                - Генерация трафика для пользователей при наличии модели.
                - Выполнение планирования ресурсов.
                - Сбор статистики через менеджер (если включен)
                - При необходимости запись статистики в CSV-файл.
            5. Завершение симуляции.

        """
        # Перевод консольного вывода в текстовый файл
        if self._to_file:
            self._log_file = open("output.txt", "w", buffering=1, encoding="utf-8")
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = self._log_file
            sys.stderr = self._log_file
            self._return_stdout = True

        try:
            # Проверка обязательных параметров симуляции
            self._check_required_parameters()

            # Расчёт числа кадров для ресурсной сетки
            num_frames = int(np.ceil(self.sim_config.sim_duration / 10))
            if self.sim_config.verbose:
                print(f"[SIMULATION] Calculated number of frames for resource grid: {num_frames}")

            # Создание ресурсной сетки
            lte_grid = RES_GRID_LTE(bandwidth=self.base_station.bandwidth, num_frames=num_frames)
            if self.sim_config.verbose and lte_grid:
                print(
                    f"[SIMULATION] The resource grid has been initialized. "
                    f"Bandwidth={lte_grid.bandwidth} MHz. RBs={lte_grid.rb_per_slot}"
                )

            # Создание планировщика
            self.scheduler = SchedulerInterface.create(
                algorithm=self.sched_config.algorithm,
                lte_grid=lte_grid,
                bs=self.base_station,
                max_dl_ue_tti=self.sched_config.max_dl_ue_tti,
                pcfich=self.sched_config.pcfich,
                max_dl_cce_allowance=self.sched_config.max_dl_cce_allowance,
                verbose_pdcch=self.sim_config.verbose,
                window_size=self.sched_config.window_size,
                enable_window=self.sched_config.enable_window,
                verbose=self.sim_config.verbose,
            )

            # Инициализация менеджера статистики
            if self.stats_config.enabled:
                level_map = {
                    "none": MetricLevel.NONE,
                    "basic": MetricLevel.BASIC,
                    "advanced": MetricLevel.ADVANCED,
                    "full": MetricLevel.FULL,
                }

                # Создание конфига для StatsManager
                stats_config = StatisticsConfig(
                    collect_interval=self.stats_config.collect_interval,
                    levels=LevelsConfig(
                        scheduler=level_map[self.stats_config.scheduler_level],
                        amc=level_map[self.stats_config.amc_level],
                        pdcch=level_map[self.stats_config.pdcch_level],
                    ),
                    export_format=self.stats_config.export_format,
                    file_prefix=self.stats_config.file_prefix,
                    history_max_len=self.stats_config.history_max_len,
                )

                self.stats_manager = StatsManager(self.scheduler, stats_config)

                if self.sim_config.verbose:
                    print(
                        f"[SIMULATION] StatsManager enabled "
                        f"(interval={self.stats_config.collect_interval} TTI, "
                        f"scheduler={self.stats_config.scheduler_level}, "
                        f"amc={self.stats_config.amc_level})"
                    )

            # Основной цикл симуляции
            for tti in range(self.sim_config.sim_duration):
                if self.sim_config.verbose:
                    print(f"\n[SIMULATION] Start TTI {tti}...")

                GLOBALS.CURRENT_TIME = tti

                sched_result = self.run_tti(current_time=tti)

                # Вывод статистики в CSV файл
                if self.stats_manager and tti % self.stats_manager.config.collect_interval == 0:
                    self.stats_manager.collect(tti)

            if self.stats_manager:
                # Экспорт в CSV
                output_filename = f"{self.stats_config.file_prefix}.csv"
                self.stats_manager.export_csv(output_filename, locale="ru")

                # Отладочный момент для глобального JI (Fairness)
                ue_avg_throughputs = {}
                for ue in self.ue_collection.GET_ALL_USERS():
                    ue_avg_throughputs[ue.UE_ID] = ue.average_throughput
                longterm_fairness_metrics = self.stats_manager._calculate_fairness(
                    ue_throughputs=ue_avg_throughputs
                )
                print(
                    f"[SIMULATION] Jain's Fairness Index: {longterm_fairness_metrics['dl_fairness_jain_index']:.4f}"
                )

                # Вывод summary (если verbose включен)
                if (
                    self.stats_config.scheduler_level == "full"
                    or self.stats_config.amc_level == "full"
                ):
                    if self.stats_config.export_detailed_format == "csv":
                        detailed_filename = f"{self.stats_config.file_prefix}_detailed.csv"
                        self.stats_manager.export_detailed_csv(detailed_filename, locale="ru")
                    elif self.stats_config.export_detailed_format == "json":
                        detailed_filename = f"{self.stats_config.file_prefix}_detailed.json"
                        self.stats_manager.export_detailed_json(detailed_filename)

        finally:
            # Возвращение консольного вывода
            if self._to_file:
                sys.stdout = self._original_stdout
                sys.stderr = self._original_stderr
                self._log_file.close()

    def _check_required_parameters(self) -> None:
        """
        Проверка наличия всех обязательных параметров симуляции.

        Raises:
            RuntimeError: Если не заданы один или несколько ключевых параметров.

        """
        errors = []

        if self.sim_config.sim_duration is None:
            errors.append(
                "Не задана длительность симуляции. "
                "Используйте set_sim_duration(*) для установки длительности (мс)."
            )

        if self.ue_collection is None:
            errors.append(
                "Не задана коллекция пользователей. "
                "Используйте set_ue_collection(*) для установки коллекции UE."
            )

        if self.base_station is None:
            errors.append(
                "Не задана базовая станция. "
                "Используйте set_base_station(*) для установки базовой станции."
            )

        if self.sched_config.algorithm is None:
            errors.append(
                "Не задан алгоритм планирования ресурсов. "
                "Используйте set_scheduler(*) для настройки планировщика."
            )

        if errors:
            msg = "Ошибка: не все обязательные параметры симуляции заданы:\n" + "\n".join(errors)
            raise RuntimeError(msg)

    def setup_ue_traffic(self, ue_id, model_type, **params):
        """
        Настройка трафика. Работает как для старого SimpleGenerator,
        так и для нового PacketManager.
        """
        if self.use_legacy_traffic:
            # Старый способ (Phase 1)
            self.traffic_gen.set_model(ue_id, model_type, **params)
        else:
            # Новый способ (Phase 2 - Multi-bearer)
            # Если передали старые параметры, адаптируем их под QoS
            qci = params.pop("qci", 9)  # Default Web
            traffic_type = params.pop("traffic_type", TrafficType.DEFAULT)

            self.traffic_gen.add_bearer(
                ue_id=ue_id, model_type=model_type, qci=qci, traffic_type=traffic_type, **params
            )

    def setup_traffic_profiles(self, traffic_profiles: List[Dict[str, Any]]):
        """
        Инициализирует и настраивает профили трафика (bearers) для списка пользователей.

        Принимает конфигурацию, проходит по каждому UE и создает соответствующие
        генераторы трафика через PacketManager/BaseStation.

        Args:
            traffic_profiles (List[Dict[str, Any]]): Список конфигураций для пользователей.
                Каждый элемент списка должен иметь следующую структуру:

                [
                    {
                        "ue_id": int,       # ID существующего пользователя
                        "bearers": [        # Список биреров (потоков) для этого UE
                            {
                                # Обязательные параметры:
                                "traffic_type": TrafficType, # Тип (VOIP, VIDEO_STREAM, BACKGROUND...)
                                "model_type": str,           # "Poisson", "OnOff", "MMPP"
                                "qci": int,                  # QCI индекс (например, 1, 5, 9)

                                # Параметры для модели Poisson:
                                "packet_rate": int,          # Частота пакетов (пак/сек)
                                "max_bitrate": float,        # (Опционально) Макс. битрейт (Мбит/с)

                                # Дополнительные параметры для модели OnOff:
                                "duration_on": float,        # (Опционально) Время активности (сек)
                                "duration_off": float,       # (Опционально) Время простоя (сек)
                            },
                            # ... другие биреры
                        ]
                    },
                    # ... другие пользователи
                ]

        Raises:
            RuntimeError: Если BaseStation не инициализирована.
            ValueError: Если передан ue_id, которого нет в коллекции пользователей.

        Example:
            >>> config = [{
            ...     "ue_id": 1,
            ...     "bearers": [{
            ...         "model_type": "OnOff",
            ...         "qci": 1,
            ...         "traffic_type": TrafficType.VOIP,
            ...         "packet_rate": 50,
            ...         "duration_on": 2.0,
            ...         "duration_off": 3.0
            ...     }]
            ... }]
            >>> sim.setup_traffic_profiles(config)
        """
        if not self.base_station:
            raise RuntimeError("BaseStation не инициализирована")

        print(f"\n[SIM] Настройка профилей трафика для {len(traffic_profiles)} пользователей...")

        for profile in traffic_profiles:
            ue_id = profile.get("ue_id")
            bearers = profile.get("bearers", [])

            if ue_id is None:
                print("[WARNING] Пропущен профиль без ue_id")
                continue

            try:
                self.base_station.setup_ue_traffic_multi_bearer(ue_id=ue_id, bearers=bearers)
                print(f"  -> UE {ue_id}: настроено {len(bearers)} биреров")
            except Exception as e:
                print(f"[ERROR] Ошибка настройки UE {ue_id}: {e}")

    def run_tti(self, current_time: int):
        """
        Выполнение одного TTI с поддержкой обоих режимов генерации трафика.
        """
        # Обновление физики и позиций UE
        if current_time % self.sim_config.update_interval == 0:
            if self.sim_config.verbose:
                print("[SIMULATION] Update UEs states")

            self.ue_collection.UPDATE_ALL_USERS(
                current_time=current_time,
                update_interval=self.sim_config.update_interval,
            )
        # БЛОК ГЕНЕРАЦИИ ТРАФИКА

        if self.use_legacy_traffic:
            # LEGACY MODE
            all_users = self.ue_collection.GET_ALL_USERS()

            for ue in all_users:
                ue_id = ue.UE_ID

                # Пропускаем UE без модели
                if ue_id not in self.traffic_gen.models:
                    continue

                # Генерируем пакеты по одному юзеру
                packets = self.traffic_gen.generate_packets(
                    ue_id=ue_id, current_time=current_time, update_interval=self.tti_duration
                )

                # Кладем в буфер вручную
                if packets and (ue_id in self.base_station.ue_buffers):
                    for pkt in packets:
                        self.base_station.ue_buffers[ue_id].ADD_PACKET(pkt, current_time)

        else:
            # PACKET MANAGER MODE
            # PacketManager сам обновляет состояния всех беареров всех юзеров
            # и возвращает общий список новых пакетов
            new_packets = self.traffic_gen.update_all(current_time, self.tti_duration)

            # Маршрутизатор: раскидываем полученные пакеты по буферам БС
            for pkt in new_packets:
                if pkt.ue_id in self.base_station.ue_buffers:
                    self.base_station.ue_buffers[pkt.ue_id].ADD_PACKET(pkt, current_time)

        # 3. Планировщик
        users = self.ue_collection.GET_USERS_FOR_SCHEDULER()
        sched_result = self.scheduler.schedule(current_time, users)

        return sched_result

    def generate_traffic_for_ue(self, ue_id, current_time, interval):
        """Генерация трафика"""
        packets = self.traffic_gen.generate_packets(ue_id, current_time, interval)
        for pkt in packets:
            self.base_station.ue_buffers[ue_id].ADD_PACKET(pkt, current_time)
