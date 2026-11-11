"""Телеметрия LTE: независимые контракты, конфигурация, сбор и экспорт."""

import csv
import json
import warnings
from pathlib import Path
from collections.abc import Mapping
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ILteGridStatsProvider(Protocol):
    """Параметры сетки: полоса в МГц и число RB в слоте."""
    bandwidth: float
    rb_per_slot: int


@runtime_checkable
class IAmcStatsProvider(Protocol):
    def get_stats(self) -> Dict[str, Any]: ...


@runtime_checkable
class IPdcchStatsProvider(Protocol):
    def get_stats(self) -> Dict[str, Any]: ...


@runtime_checkable
class ISchedulerStatsProvider(Protocol):
    lte_grid: ILteGridStatsProvider
    amc: IAmcStatsProvider
    pdcch_manager: IPdcchStatsProvider

    def get_stats(self) -> Dict[str, Any]: ...


class MetricLevel(IntEnum):
    """
    Уровни детализации сбора метрик.

    - NONE: Не собирать (0% overhead)
    - BASIC: Только собираемые метрики (total, avg)
    - ADVANCED: Расчетные агрегаты и индексы
    - FULL: Поабонентная детализация
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
    qos: MetricLevel = MetricLevel.BASIC


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
    levels: LevelsConfig = field(default_factory=LevelsConfig)
    export_format: str = "csv"
    export_detailed_format: str = "csv"
    history_max_len: int = 10000
    file_prefix: str = "lte_stats"

    def __post_init__(self):
        """Инициализация levels, если не передан"""
        if self.levels is None:
            self.levels = LevelsConfig()

        if (isinstance(self.collect_interval, bool)
                or not isinstance(self.collect_interval, int)
                or self.collect_interval <= 0):
            raise ValueError("collect_interval must be a positive integer")
        if (isinstance(self.history_max_len, bool)
                or not isinstance(self.history_max_len, int)
                or self.history_max_len <= 0):
            raise ValueError("history_max_len must be a positive integer")

        if self.export_format not in ["csv", "json"]:
            raise ValueError(f"Invalid export_format: {self.export_format}")

        if self.export_detailed_format not in ["csv", "json"]:
            raise ValueError(f"Invalid export_detailed_format: {self.export_detailed_format}")


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
        qos_level (str): Уровень для QoS статистики
        export_format (str): Формат экспорта ('csv' или 'json')
        file_prefix (str): Префикс файла для экспорта
    """

    enabled: bool = True
    collect_interval: int = 10
    scheduler_level: str = "basic"
    amc_level: str = "basic"
    pdcch_level: str = "none"
    qos_level: str = "none"
    export_format: str = "csv"
    export_detailed_format: str = "csv"
    file_prefix: str = "manager_stats"
    history_max_len: int = 10000

    def to_statistics_config(self) -> StatisticsConfig:
        """Преобразовать фасад настроек, включая QoS, в канонический конфиг."""
        level_map = {level.name.lower(): level for level in MetricLevel}
        def parse_level(value, name):
            if isinstance(value, MetricLevel):
                return value
            if not isinstance(value, str) or value.lower() not in level_map:
                raise ValueError(f"Invalid {name}: {value!r}")
            return level_map[value.lower()]

        return StatisticsConfig(
            collect_interval=self.collect_interval,
            levels=LevelsConfig(
                scheduler=parse_level(self.scheduler_level, "scheduler_level"),
                amc=parse_level(self.amc_level, "amc_level"),
                pdcch=parse_level(self.pdcch_level, "pdcch_level"),
                qos=parse_level(self.qos_level, "qos_level"),
            ),
            export_format=self.export_format,
            export_detailed_format=self.export_detailed_format,
            history_max_len=self.history_max_len,
            file_prefix=self.file_prefix,
        )


class StatsManager:
    """
    Менеджер сбора и экспорта статистики LTE-симуляции.
    Собирает метрики с scheduler/amc/pdcch по настраиваемому интервалу,
    хранит историю и экспортирует в CSV/JSON.
    """

    def __init__(self, scheduler: ISchedulerStatsProvider,
                 config: Optional[StatisticsConfig] = None, *, buffer_provider=None):
        """
        Инициализация менеджера статистики.

        Args:
            scheduler (ISchedulerStatsProvider): Планировщик для опроса
            config (StatisticsConfig): Конфигурация (если None, используется default)
        """
        self.scheduler = scheduler  # Ссылка на scheduler (для вызова get_stats())
        self.config = config or StatisticsConfig()  # Default конфиг если None
        # Explicit dependency from SimulationManager; legacy lookup remains for old callers.
        self.buffer_provider = buffer_provider
        if self.buffer_provider is None:
            grid = getattr(scheduler, "lte_grid", None)
            bs = getattr(grid, "bs", None)
            if bs is not None and not getattr(bs, "use_simple_buffer", True):
                self.buffer_provider = getattr(bs, "buffer_manager", None)
        self._qos_unavailable_warned = False
        self.history = deque(maxlen=self.config.history_max_len)  # История snapshots
        self.detailed_history = deque(maxlen=self.config.history_max_len)
        self.qos_history = deque(maxlen=self.config.history_max_len)

    def _validate_sched_stats(self, sched_stats: dict, tti: int) -> bool:
        """Валидация что данные sched_stats полноценные и корректные"""
        import warnings

        if not isinstance(sched_stats, Mapping):
            warnings.warn(f"[StatsManager] TTI {tti}: scheduler snapshot must be a mapping.")
            return False

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

    @staticmethod
    def _read_snapshot(provider, name, tti):
        """Read real telemetry; never fabricate data for an unavailable source."""
        getter = getattr(provider, "get_stats", None)
        if not callable(getter):
            warnings.warn(f"[StatsManager] TTI {tti}: {name}.get_stats is unavailable.")
            return None
        snapshot = getter()
        if not isinstance(snapshot, Mapping):
            warnings.warn(f"[StatsManager] TTI {tti}: {name} returned a non-mapping snapshot.")
            return None
        if "tti" in snapshot and snapshot["tti"] != tti:
            warnings.warn(f"[StatsManager] TTI mismatch in {name}: expected {tti}, "
                          f"got {snapshot['tti']}. Skipping collection.")
            return None
        # Do not clone live UE objects contained in scheduler priority lists.
        return dict(snapshot)

    def collect(self, tti: int) -> None:
        """
        Собрать метрики на текущем TTI.
        Опрашивает scheduler.get_stats() и amc.get_stats() в зависимости от
        уровней в config.levels, объединяет в один Dict (snapshot) и добавляет
        в self.history.

        Args:
            tti (int): Номер текущего TTI (для timestamp)
        """

        sched_stats = {}
        amc_stats = {}
        pdcch_stats = {}
        if self.config.levels.scheduler != MetricLevel.NONE:
            sched_stats = self._read_snapshot(self.scheduler, "scheduler", tti)
            if sched_stats is None:
                return
        if self.config.levels.amc != MetricLevel.NONE:
            amc_stats = self._read_snapshot(getattr(self.scheduler, "amc", None), "AMC", tti)
            if amc_stats is None:
                return
        if self.config.levels.pdcch != MetricLevel.NONE:
            pdcch_stats = self._read_snapshot(
                getattr(self.scheduler, "pdcch_manager", None), "PDCCH", tti)
            if pdcch_stats is None:
                return

        if (self.config.levels.scheduler != MetricLevel.NONE
                and not self._validate_sched_stats(sched_stats, tti)):
            return

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

        if self.config.levels.scheduler >= MetricLevel.ADVANCED:
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

        self.history.append(deepcopy(snapshot))

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
            self.detailed_history.append(deepcopy(detailed))

        # QoS is optional; a missing source is visible and is not replaced by zeros.
        if self.config.levels.qos != MetricLevel.NONE:
            if self.buffer_provider is None:
                if not self._qos_unavailable_warned:
                    warnings.warn("[StatsManager] QoS enabled but no layered-buffer "
                                  "statistics provider is available; QoS collection skipped.")
                    self._qos_unavailable_warned = True
            else:
                buffer_stats = self._read_snapshot(self.buffer_provider, "buffer", tti)
                if buffer_stats is not None:
                    self.qos_history.append(deepcopy(self._collect_qos_metrics(tti, buffer_stats)))

    def get_history(self) -> List[Dict[str, Any]]:
        """Вернуть независимую копию агрегированной истории."""
        return deepcopy(list(self.history))

    def clear_history(self) -> None:
        """Очистить агрегированную, поабонентную и QoS-историю."""
        self.history.clear()
        self.detailed_history.clear()
        self.qos_history.clear()

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
        ue_cqi_sbb = amc_stats.get('ue_cqi_sbb', {})
        ue_sinr = amc_stats.get("ue_sinr", {})
        ue_rb = amc_stats.get("ue_rb_allocated", {})
        ue_cce_alloc = pdcch_stats.get("pdcch_ue_cce_allocations", {})

        # Собрать все unique UE IDs
        all_ue_ids = (
            set(ue_priorities.keys())
            | set(ue_throughputs.keys())
            | set(ue_cqi.keys())
            | set(ue_cqi_sbb.keys())
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
                "cqi_sbb": ue_cqi_sbb.get(ue_id, 0),
                "sinr": ue_sinr.get(ue_id, 0.0),
                "rb_allocated": ue_rb.get(ue_id, 0),
                "cce_allocated": ue_cce_alloc.get(ue_id, 0),
                "throughput_bps": ue_throughputs.get(ue_id, 0),
                "buffer_size_bytes": ue_buffer_sizes.get(ue_id, 0),
                "bits_transmitted": ue_transmitted_bits.get(ue_id, 0),
            }

        return detailed

    def _collect_qos_metrics(self, tti: int, buffer_stats: dict) -> dict:
        """
        Собрать QoS per-UE-per-QCI статистику.

        Returns:
            Dict в формате:
            {
                "tti": 100,
                "qos_ue_metrics": {
                    "1": {
                    "buffer_size": {"1": 1500, "5": 500, "7": 3200},
                    "bytes_transmitted": {"1": 1000, "5": 200, "7": 2000},
                    ...
                    },
                    "2": {
                    "buffer_size": {"7": 0, "8": 5000, "9": 200},
                    "bytes_transmitted": {"7": 0, "8": 2500, 9": 50},
                    ...
                    },
                }
            }
        """
        qos_snapshot = {"tti": tti, "qos_ue_metrics": {}}

        ue_buffer_size_per_qci = buffer_stats.get("buffer_size_per_ue_per_qci", {})
        ue_transmitted_bytes_per_qci = buffer_stats.get("extracted_bytes_per_ue_per_qci", {})
        ue_packets_ext_per_qci = buffer_stats.get("packets_ext_per_ue_per_qci", {})
        ue_packets_ext_late_per_qci = buffer_stats.get("packets_ext_late_per_ue_per_qci", {})
        ue_packets_added_per_qci = buffer_stats.get("packets_added_per_ue_per_qci", {})
        ue_packets_dropped_per_qci = buffer_stats.get("packets_dropped_per_ue_per_qci", {})
        ue_packets_expired_per_qci = buffer_stats.get("packets_expired_per_ue_per_qci", {})
        ue_oldest_delay_per_qci = buffer_stats.get("oldest_delay_per_ue_per_qci", {})
        avg_delay_per_ue_per_qci = buffer_stats.get("avg_delay_per_ue_per_qci", {})

        all_ue_ids = (
            set(ue_buffer_size_per_qci.keys())
            | set(ue_transmitted_bytes_per_qci.keys())
            | set(ue_packets_ext_per_qci.keys())
            | set(ue_packets_ext_late_per_qci.keys())
            | set(ue_packets_added_per_qci.keys())
            | set(ue_packets_dropped_per_qci.keys())
            | set(ue_packets_expired_per_qci.keys())
            | set(ue_oldest_delay_per_qci.keys())
            | set(avg_delay_per_ue_per_qci.keys())
        )

        for ue_id in all_ue_ids:
            qos_snapshot["qos_ue_metrics"][str(ue_id)] = {
                "buffer_size": ue_buffer_size_per_qci.get(ue_id, {}),
                "bytes_transmitted": ue_transmitted_bytes_per_qci.get(ue_id, {}),
                "packets_transmitted": ue_packets_ext_per_qci.get(ue_id, {}),
                "late_packets_transmitted": ue_packets_ext_late_per_qci.get(ue_id, {}),
                "packets_added": ue_packets_added_per_qci.get(ue_id, {}),
                "packets_dropped": ue_packets_dropped_per_qci.get(ue_id, {}),
                "packets_expired": ue_packets_expired_per_qci.get(ue_id, {}),
                "hol_delay": ue_oldest_delay_per_qci.get(ue_id, {}),
                "avg_delay": avg_delay_per_ue_per_qci.get(ue_id, {}),
            }

        return qos_snapshot

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

        if sum_throughput == 0 or sum_squared == 0:
            jain_index = None
        else:
            jain_index = (sum_throughput ** 2) / (n * sum_squared)

        # Variance and Std
        mean = sum_throughput / n
        variance = sum((x - mean) ** 2 for x in throughputs) / n
        std = variance**0.5

        return {
            'dl_fairness_jain_index':   round(jain_index, 4) if jain_index is not None else None,
            'dl_throughput_variance':   round(variance, 2),
            'dl_throughput_std':        round(std, 2)
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

    def calculate_fairness(self, ue_throughputs: dict) -> dict:
        """Public facade retaining the existing numerical implementation."""
        return self._calculate_fairness(ue_throughputs)

    def export_json(self, filename: str = None) -> None:
        """Export aggregate history as a list of TTI snapshots."""
        filename = filename or f"{self.config.file_prefix}.json"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as stream:
            json.dump(self.get_history(), stream, ensure_ascii=False, indent=2)

    def export_all(self, locale: str = "ru") -> List[str]:
        """Export available histories and return the actual output paths."""
        paths = []
        prefix = self.config.file_prefix
        if self.history:
            filename = f"{prefix}.{self.config.export_format}"
            if self.config.export_format == "json":
                self.export_json(filename)
            else:
                self.export_csv(filename, locale=locale)
            paths.append(filename)
        if self.detailed_history:
            filename = f"{prefix}_detailed.{self.config.export_detailed_format}"
            if self.config.export_detailed_format == "json":
                self.export_detailed_json(filename)
            else:
                self.export_detailed_csv(filename, locale=locale)
            paths.append(filename)
        if self.qos_history:
            filename = f"{prefix}_qos.csv"
            self.export_qos_csv(filename, locale=locale)
            paths.append(filename)
        return paths

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
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
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

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "tti",
                "ue_id",
                "priority",
                "cqi",
                "cqi_sbb",
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
                        "cqi_sbb": metrics.get('cqi_sbb', 0),
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

    def export_qos_csv(self, filename: str = None, locale: str = "ru") -> None:
        """
        Экспортировать QoS статистику в CSV (per-UE-per-QCI rows).

        Формат: tti, ue_id, qci, cqi, buffer_size, bits_transmitted и т.д.
        """
        if not self.qos_history:
            print("[StatsManager] No QoS data to export (qos_history is empty)")
            return

        if filename is None:
            filename = f"{self.config.file_prefix}_qos.csv"

        if locale == "ru":
            delimiter = ";"
        else:
            delimiter = ","

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "tti",
                "ue_id",
                "qci",
                "buffer_size",
                "bits_transmitted",
                "packets_transmitted",
                "late_packets_transmitted",
                "packets_added",
                "packets_dropped",
                "packets_expired",
                "hol_delay",
                "avg_delay",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()

            for snapshot in self.qos_history:
                tti = snapshot["tti"]
                for ue_id, metrics in snapshot["qos_ue_metrics"].items():
                    buffer_size = metrics.get("buffer_size", {})
                    bytes_transmitted = metrics.get("bytes_transmitted", {})
                    packets_transmitted = metrics.get("packets_transmitted", {})
                    late_packets_transmitted = metrics.get("late_packets_transmitted", {})
                    packets_added = metrics.get("packets_added", {})
                    packets_dropped = metrics.get("packets_dropped", {})
                    packets_expired = metrics.get("packets_expired", {})
                    hol_delay = metrics.get("hol_delay", {})
                    avg_delay = metrics.get("avg_delay", {})

                    qci_keys = sorted(
                        set(buffer_size.keys())
                        | set(bytes_transmitted.keys())
                        | set(packets_transmitted.keys())
                        | set(late_packets_transmitted.keys())
                        | set(packets_added.keys())
                        | set(packets_dropped.keys())
                        | set(packets_expired.keys())
                        | set(hol_delay.keys())
                        | set(avg_delay.keys()),
                        key=str,
                    )

                    for qci in qci_keys:
                        row = {
                            "tti": tti,
                            "ue_id": ue_id,
                            "qci": qci,
                            "buffer_size": buffer_size.get(qci, 0),
                            "bits_transmitted": bytes_transmitted.get(qci, 0) * 8,
                            "packets_transmitted": packets_transmitted.get(qci, 0),
                            "late_packets_transmitted": late_packets_transmitted.get(qci, 0),
                            "packets_added": packets_added.get(qci, 0),
                            "packets_dropped": packets_dropped.get(qci, 0),
                            "packets_expired": packets_expired.get(qci, 0),
                            "hol_delay": hol_delay.get(qci, 0),
                            "avg_delay": avg_delay.get(qci, 0),
                        }
                        writer.writerow(row)

        print(
            f"[StatsManager] Exported {len(self.qos_history)} TTI snapshots "
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

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(
            f"[StatsManager] Exported {len(self.detailed_history)} TTI snapshots "
            f"to {filename} (JSON format)"
        )

__all__ = [
    "MetricLevel", "LevelsConfig", "StatisticsConfig", "StatsManagerConfig",
    "StatsManager", "ISchedulerStatsProvider", "IAmcStatsProvider",
    "IPdcchStatsProvider", "ILteGridStatsProvider",
]
