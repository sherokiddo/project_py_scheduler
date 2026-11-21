"""
Модуль STATS_MANAGER: универсальная событийно-коллекторная телеметрия LTE-симуляции.

===============================================================================
Архитектурный статус: Универсальная событийно-коллекторная архитектура (OCP).
===============================================================================
Модуль реализует Open-Closed Principle (OCP):
- Ядро StatsManager больше НЕ привязано к конкретному набору модулей симулятора.
- Владелец любого доменного модуля (Scheduler, AMC, PDCCH, Buffer/QoS, Channel,
  Mobility, Traffic, HARQ и др.) реализует универсальный контракт IStatsCollector
  (или наследуется от BaseStatsCollector) внутри своего модуля.
- Новый коллектор подключается через stats_manager.register_collector(collector).
- Добавление новой телеметрии НЕ ТРЕБУЕТ изменений кода StatsManager,
  модификации метода collect() или переписывания конфигурационных структур.

Для сохранения 100% обратной совместимости с существующим кодом и тестовыми стендами
предусмотрены встроенные адаптеры (LegacySchedulerCollector, LegacyAmcCollector,
LegacyBufferCollector, LegacyPdcchCollector), которые регистрируются автоматически
при передаче scheduler и base_station в StatsManager.
"""

import csv
import json
import math
import os
import warnings
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import numpy as np


# =============================================================================
# Уровни детализации телеметрии
# =============================================================================

class MetricLevel(IntEnum):
    """
    Уровни детализации сбора метрик.

    - NONE: Не собирать (0% накладных расходов).
    - BASIC: Только базовые агрегаты и счетчики (total, avg, count).
    - ADVANCED: Расчетные индексы, мин/макс/стд, распределения и эффективность.
    - FULL: Полная детализация, включая поабонентные (per-UE) метрики.
    """

    NONE = 0
    BASIC = 1
    ADVANCED = 2
    FULL = 3

    @classmethod
    def from_value(cls, val: Union["MetricLevel", int, str]) -> "MetricLevel":
        """Безопасное преобразование строки или числа в MetricLevel."""
        if isinstance(val, MetricLevel):
            return val
        if isinstance(val, int):
            try:
                return cls(val)
            except ValueError:
                return cls.BASIC
        if isinstance(val, str):
            clean = val.strip().lower()
            mapping = {
                "none": cls.NONE,
                "0": cls.NONE,
                "basic": cls.BASIC,
                "1": cls.BASIC,
                "advanced": cls.ADVANCED,
                "2": cls.ADVANCED,
                "full": cls.FULL,
                "3": cls.FULL,
            }
            return mapping.get(clean, cls.BASIC)
        return cls.BASIC


# =============================================================================
# Универсальный контракт коллектора телеметрии
# =============================================================================

@runtime_checkable
class IStatsCollector(Protocol):
    """
    Универсальный контракт коллектора метрик (Protocol).
    Каждый модуль системы (Scheduler, AMC, PDCCH, Buffer/QoS, Channel,
    Mobility, HARQ, Traffic и др.) реализует этот контракт.
    """

    def get_collector_name(self) -> str:
        """
        Уникальный строковый идентификатор коллектора (например, 'scheduler', 'channel').
        Используется как ключ в реестре и для настройки уровня детализации.
        """
        ...

    def collect_aggregated(self, tti: int, level: MetricLevel) -> Dict[str, Any]:
        """
        Собрать агрегированные системные метрики модуля на текущем TTI.

        Args:
            tti (int): Номер текущего TTI симуляции.
            level (MetricLevel): Запрошенный уровень детализации (BASIC, ADVANCED, FULL).

        Returns:
            Dict[str, Any]: Плоский словарь метрик вида {'metric_name': value}.
            Если на данном уровне метрики не собираются, возвращается пустой dict {}.
        """
        ...

    def collect_detailed(self, tti: int, level: MetricLevel) -> Optional[Dict[Any, Dict[str, Any]]]:
        """
        (Опционально) Собрать поабонентные (per-UE) метрики модуля на текущем TTI.

        Args:
            tti (int): Номер текущего TTI симуляции.
            level (MetricLevel): Запрошенный уровень детализации (обычно FULL).

        Returns:
            Dict[Any, Dict[str, Any]] вида {ue_id: {'metric_name': value, ...}},
            или None / {}, если модуль не предоставляет поабонентных данных.
        """
        ...

    def collect_custom_records(self, tti: int, level: MetricLevel) -> Optional[Any]:
        """
        (Опционально) Собрать специфичные многомерные записи (например, QoS per-UE-per-QCI).

        Returns:
            Любая структура данных (например, словарь с записями) или None.
        """
        ...


class BaseStatsCollector:
    """
    Базовый класс для коллекторов с реализациями по умолчанию.
    Разработчикам модулей рекомендуется наследоваться от него.
    """

    default_level: MetricLevel = MetricLevel.BASIC

    def __init__(self, name: str):
        self.name = str(name).lower().strip()

    def get_collector_name(self) -> str:
        return self.name

    def collect_aggregated(self, tti: int, level: MetricLevel) -> Dict[str, Any]:
        return {}

    def collect_detailed(self, tti: int, level: MetricLevel) -> Optional[Dict[Any, Dict[str, Any]]]:
        return None

    def collect_custom_records(self, tti: int, level: MetricLevel) -> Optional[Any]:
        return None


# =============================================================================
# Legacy Protocols для этапа E0 (Extraction с сохранением legacy polling)
# =============================================================================

@runtime_checkable
class ILteGridStatsProvider(Protocol):
    """
    Legacy-протокол доступа к параметрам сетки LTE (Этап E0).
    Явно объявляет bs для соответствия реальным зависимостям fallback-пути.
    """
    num_rb: int
    bandwidth: float
    bs: Optional[Any]


@runtime_checkable
class IAmcStatsProvider(Protocol):
    """Legacy-протокол доступа к статистике AMC (Этап E0)."""

    def get_stats(self) -> Dict[str, Any]:
        ...


@runtime_checkable
class IPdcchStatsProvider(Protocol):
    """Legacy-протокол доступа к статистике PDCCH (Этап E0)."""

    def get_stats(self) -> Dict[str, Any]:
        ...


@runtime_checkable
class ISchedulerStatsProvider(Protocol):
    """
    Legacy-протокол доступа к планировщику (Этап E0).
    Отражает существующие связи Scheduler с AMC, PDCCH и Grid.
    В этапе E1 эти зависимости будут полностью разорваны через реестр коллекторов.
    """
    lte_grid: Optional[ILteGridStatsProvider]
    amc: Optional[IAmcStatsProvider]
    pdcch_manager: Optional[IPdcchStatsProvider]

    def get_stats(self) -> Dict[str, Any]:
        ...



# =============================================================================
# Конфигурационные структуры
# =============================================================================

@dataclass
class LevelsConfig:
    """Конфигурация уровней детализации для зарегистрированных коллекторов."""

    scheduler: MetricLevel = MetricLevel.BASIC
    amc: MetricLevel = MetricLevel.BASIC
    pdcch: MetricLevel = MetricLevel.NONE
    qos: MetricLevel = MetricLevel.BASIC
    buffer: MetricLevel = MetricLevel.BASIC
    channel: MetricLevel = MetricLevel.NONE
    mobility: MetricLevel = MetricLevel.NONE
    traffic: MetricLevel = MetricLevel.NONE
    harq: MetricLevel = MetricLevel.NONE

    custom_levels: Dict[str, MetricLevel] = field(default_factory=dict)

    def get_level(self, collector_name: str, default: MetricLevel = MetricLevel.BASIC) -> MetricLevel:
        """Получить уровень детализации для коллектора по его имени."""
        cname = collector_name.lower().strip()
        if cname in self.custom_levels:
            return self.custom_levels[cname]
        if hasattr(self, cname):
            val = getattr(self, cname)
            if isinstance(val, MetricLevel):
                return val
            return MetricLevel.from_value(val)
        return default

    def set_level(self, collector_name: str, level: Union[MetricLevel, int, str]) -> None:
        """Установить уровень детализации для коллектора."""
        cname = collector_name.lower().strip()
        lvl = MetricLevel.from_value(level)
        if hasattr(self, cname):
            setattr(self, cname, lvl)
        else:
            self.custom_levels[cname] = lvl


@dataclass
class StatisticsConfig:
    """Конфигурация сбора и экспорта статистики."""

    enabled: bool = True
    collect_interval: int = 1
    file_prefix: str = "lte_stats"
    output_dir: str = "."
    export_csv: bool = True
    export_json: bool = False
    export_detailed_csv: bool = True
    export_detailed_json: bool = False
    export_format: str = "csv"
    export_detailed_format: str = "csv"
    history_max_len: Optional[int] = None
    levels: LevelsConfig = field(default_factory=LevelsConfig)
    auto_register_legacy: bool = True

    def __post_init__(self):
        if self.collect_interval <= 0:
            raise ValueError(f"collect_interval must be >= 1, got {self.collect_interval}")
        if self.history_max_len is not None and self.history_max_len <= 0:
            raise ValueError(f"history_max_len must be > 0, got {self.history_max_len}")


@dataclass
class StatsManagerConfig:
    """Конфигурация StatsManager для обратной совместимости."""

    enabled: bool = True
    collect_interval: int = 1
    file_prefix: str = "lte_stats"
    output_dir: str = "."
    export_csv: bool = True
    export_json: bool = False
    export_detailed_csv: bool = True
    export_detailed_json: bool = False
    export_format: str = "csv"
    export_detailed_format: str = "csv"
    history_max_len: Optional[int] = None

    scheduler_level: Union[MetricLevel, str] = MetricLevel.BASIC
    amc_level: Union[MetricLevel, str] = MetricLevel.BASIC
    pdcch_level: Union[MetricLevel, str] = MetricLevel.NONE
    qos_level: Union[MetricLevel, str] = MetricLevel.BASIC
    buffer_level: Union[MetricLevel, str] = MetricLevel.BASIC
    channel_level: Union[MetricLevel, str] = MetricLevel.NONE
    mobility_level: Union[MetricLevel, str] = MetricLevel.NONE
    traffic_level: Union[MetricLevel, str] = MetricLevel.NONE
    harq_level: Union[MetricLevel, str] = MetricLevel.NONE

    def to_statistics_config(self) -> StatisticsConfig:
        """Преобразование legacy-конфига в актуальный StatisticsConfig."""
        levels = LevelsConfig(
            scheduler=MetricLevel.from_value(self.scheduler_level),
            amc=MetricLevel.from_value(self.amc_level),
            pdcch=MetricLevel.from_value(self.pdcch_level),
            qos=MetricLevel.from_value(self.qos_level),
            buffer=MetricLevel.from_value(self.buffer_level),
            channel=MetricLevel.from_value(self.channel_level),
            mobility=MetricLevel.from_value(self.mobility_level),
            traffic=MetricLevel.from_value(self.traffic_level),
            harq=MetricLevel.from_value(self.harq_level),
        )
        return StatisticsConfig(
            enabled=self.enabled,
            collect_interval=self.collect_interval,
            file_prefix=self.file_prefix,
            output_dir=self.output_dir,
            export_csv=self.export_csv,
            export_json=self.export_json,
            export_detailed_csv=self.export_detailed_csv,
            export_detailed_json=self.export_detailed_json,
            export_format=self.export_format,
            export_detailed_format=self.export_detailed_format,
            history_max_len=self.history_max_len,
            levels=levels,
        )


# =============================================================================
# Математические и вспомогательные функции
# =============================================================================

def calculate_std(values: list) -> float:
    """Вычисление стандартного отклонения (выборочного или генерального)."""
    if not values or len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    return float(np.std(arr, ddof=1 if len(arr) > 1 else 0))


def calculate_cv(values: list) -> float:
    """Вычисление коэффициента вариации CV = std / mean."""
    if not values:
        return 0.0
    mean_val = float(np.mean(values))
    if mean_val == 0.0:
        return 0.0
    return calculate_std(values) / mean_val


def calculate_fairness_dict(ue_throughputs: dict) -> dict:
    """
    Расчет метрик справедливости распределения ресурсов (Jain's Fairness Index, Gini).

    Args:
        ue_throughputs (dict): Словарь вида {ue_id: throughput_bps}

    Returns:
        dict: Метрики справедливости и распределения пропускной способности.
    """
    if not ue_throughputs:
        return {
            "dl_fairness_jain_index": None,
            "dl_throughput_total_bps": 0.0,
            "dl_throughput_mean_bps": 0.0,
            "dl_throughput_min_bps": 0.0,
            "dl_throughput_max_bps": 0.0,
            "dl_throughput_std_bps": 0.0,
            "dl_throughput_cv": 0.0,
            "dl_fairness_gini_index": 0.0,
        }

    vals = [float(v) for v in ue_throughputs.values()]
    n = len(vals)
    total = sum(vals)

    if total == 0.0 or n == 0:
        return {
            "dl_fairness_jain_index": None,
            "dl_throughput_total_bps": 0.0,
            "dl_throughput_mean_bps": 0.0,
            "dl_throughput_min_bps": 0.0,
            "dl_throughput_max_bps": 0.0,
            "dl_throughput_std_bps": 0.0,
            "dl_throughput_cv": 0.0,
            "dl_fairness_gini_index": 0.0,
        }

    sum_sq = sum(x * x for x in vals)
    jain = (total * total) / (n * sum_sq) if sum_sq > 0 else 0.0
    mean = total / n

    # Gini coefficient calculation
    sorted_vals = sorted(vals)
    diff_sum = sum(abs(xi - xj) for xi in sorted_vals for xj in sorted_vals)
    gini = diff_sum / (2 * n * total) if total > 0 else 0.0

    return {
        "dl_fairness_jain_index": round(float(jain), 4),
        "dl_throughput_total_bps": round(float(total), 2),
        "dl_throughput_mean_bps": round(float(mean), 2),
        "dl_throughput_min_bps": round(float(min(vals)), 2),
        "dl_throughput_max_bps": round(float(max(vals)), 2),
        "dl_throughput_std_bps": round(calculate_std(vals), 2),
        "dl_throughput_cv": round(calculate_cv(vals), 4),
        "dl_fairness_gini_index": round(float(gini), 4),
    }


# =============================================================================
# Встроенные адаптеры существующих модулей (Legacy Adapters)
# =============================================================================

class LegacySchedulerCollector(BaseStatsCollector):
    """
    Адаптер телеметрии планировщика ресурсов (SCHEDULER).
    Совместим как со старыми интерфейсами (scheduler.get_stats()),
    так и с результатами вызова scheduler.schedule(tti, users).
    """

    def __init__(self, scheduler: Any):
        super().__init__(name="scheduler")
        self.scheduler = scheduler
        self._last_result: Optional[dict] = None
        self._last_tti: Optional[int] = None

    def record_schedule_result(self, tti: int, result: dict) -> None:
        """Сохранить результат работы планировщика за текущий TTI."""
        self._last_tti = tti
        self._last_result = result

    def _read_snapshot(self, tti: int) -> Dict[str, Any]:
        """Получить снимок состояния планировщика."""
        # 1. Если у объекта есть метод get_stats()
        if hasattr(self.scheduler, "get_stats") and callable(self.scheduler.get_stats):
            try:
                snap = self.scheduler.get_stats()
                if isinstance(snap, Mapping):
                    return dict(snap)
            except Exception:
                pass

        # 2. Если результат передан через record_schedule_result
        res = self._last_result or getattr(self.scheduler, "_last_schedule_result", None)
        if isinstance(res, Mapping):
            allocation = res.get("allocation", {})
            stats = res.get("statistics", {})

            active_ues = [ue_id for ue_id, rbs in allocation.items() if len(rbs) > 0]
            active_count = len(active_ues)
            eligible_count = len(allocation) if allocation else active_count

            # Вычисление выделенных RB
            if "total_allocated_rbs" in stats:
                allocated_rbs = stats["total_allocated_rbs"]
            else:
                allocated_rbs = sum(len(rbs) * 2 for rbs in allocation.values())

            avg_rbs = (allocated_rbs / active_count) if active_count > 0 else 0.0

            # Расчет использования PRB из сетки, если доступна
            total_rbs = 50  # дефолт для 10 MHz
            grid = getattr(self.scheduler, "lte_grid", None)
            if grid is not None and hasattr(grid, "num_rb"):
                total_rbs = getattr(grid, "num_rb", 50)
            prb_util = (allocated_rbs / (total_rbs * 2) * 100.0) if total_rbs > 0 else 0.0

            return {
                "tti": tti,
                "sch_eligible_ue_count": eligible_count,
                "sch_active_ue_count": active_count,
                "dl_rb_allocated_count": allocated_rbs,
                "dl_rb_per_ue_avg": round(avg_rbs, 2),
                "dl_prb_utilization_pct": round(min(prb_util, 100.0), 2),
                "buffer_size_sum_bytes": stats.get("total_effective_bits", 0) // 8,
                "sch_total_time_us": stats.get("sch_total_time_us", 0.0),
                "raw_result": res,
            }

        return {}

    def collect_aggregated(self, tti: int, level: MetricLevel) -> Dict[str, Any]:
        if level == MetricLevel.NONE:
            return {
                "sch_eligible_ue_count": 0,
                "sch_active_ue_count": 0,
                "dl_rb_allocated_count": 0,
                "dl_rb_per_ue_avg": 0.0,
                "dl_prb_utilization_pct": 0.0,
                "buffer_size_sum_bytes": 0,
                "sch_total_time_us": 0.0,
            }

        data = self._read_snapshot(tti)
        if not data:
            return {
                "sch_eligible_ue_count": 0,
                "sch_active_ue_count": 0,
                "dl_rb_allocated_count": 0,
                "dl_rb_per_ue_avg": 0.0,
                "dl_prb_utilization_pct": 0.0,
                "buffer_size_sum_bytes": 0,
                "sch_total_time_us": 0.0,
            }

        metrics = {
            "sch_eligible_ue_count": data.get("sch_eligible_ue_count", 0),
            "sch_active_ue_count": data.get("sch_active_ue_count", 0),
            "dl_rb_allocated_count": data.get("dl_rb_allocated_count", 0),
            "dl_rb_per_ue_avg": data.get("dl_rb_per_ue_avg", 0.0),
            "dl_prb_utilization_pct": data.get("dl_prb_utilization_pct", 0.0),
            "buffer_size_sum_bytes": data.get("buffer_size_sum_bytes", 0),
            "sch_total_time_us": data.get("sch_total_time_us", 0.0),
        }

        if level >= MetricLevel.ADVANCED:
            active = metrics["sch_active_ue_count"]
            eligible = metrics["sch_eligible_ue_count"]
            ratio = round((active / eligible) * 100.0, 2) if eligible > 0 else 0.0
            algo_name = getattr(self.scheduler, "algorithm", None) or self.scheduler.__class__.__name__

            metrics.update({
                "sch_algorithm": str(algo_name),
                "sch_eligible_to_active_ratio": ratio,
                "sch_priority_calc_time_us": data.get("sch_priority_calc_time_us", 0.0),
                "sch_priority_sort_time_us": data.get("sch_priority_sort_time_us", 0.0),
                "sch_priority_list_size": data.get("sch_priority_list_size", eligible),
                "sch_pdcch_blocked_count": data.get("sch_pdcch_blocked_count", 0),
                "sch_avg_priority_value": data.get("sch_avg_priority_value", 0.0),
                "sch_priority_min": data.get("sch_priority_min", 0.0),
                "sch_priority_max": data.get("sch_priority_max", 0.0),
                "sch_priority_std": data.get("sch_priority_std", 0.0),
            })

        return metrics

    def collect_detailed(self, tti: int, level: MetricLevel) -> Optional[Dict[Any, Dict[str, Any]]]:
        if level < MetricLevel.FULL:
            return None

        detailed = {}
        # 1. Попытка чтения из get_stats()["detailed"]
        if hasattr(self.scheduler, "get_stats") and callable(self.scheduler.get_stats):
            try:
                snap = self.scheduler.get_stats()
                if "detailed" in snap and isinstance(snap["detailed"], Mapping):
                    for ue_id, u_info in snap["detailed"].items():
                        detailed[ue_id] = {
                            "rb_allocated": u_info.get("rb_allocated", 0),
                            "cce_allocated": u_info.get("cce_allocated", 0),
                            "priority": u_info.get("priority", 0.0),
                        }
                    return detailed
            except Exception:
                pass

        # 2. Чтение из _last_result
        res = self._last_result or getattr(self.scheduler, "_last_schedule_result", None)
        if isinstance(res, Mapping):
            allocation = res.get("allocation", {})
            stats = res.get("statistics", {})
            user_tp = stats.get("user_throughput", {})

            for ue_id, rbs in allocation.items():
                rb_cnt = len(rbs) * 2
                detailed[ue_id] = {
                    "rb_allocated": rb_cnt,
                    "throughput_bps": user_tp.get(ue_id, 0),
                }

        return detailed if detailed else None


class LegacyAmcCollector(BaseStatsCollector):
    """
    Адаптер телеметрии модуля Адаптивной Модуляции и Кодирования (AMC).
    """

    def __init__(self, scheduler: Any):
        super().__init__(name="amc")
        self.scheduler = scheduler

    def _get_amc_module(self):
        return getattr(self.scheduler, "amc", None)

    def collect_aggregated(self, tti: int, level: MetricLevel) -> Dict[str, Any]:
        if level == MetricLevel.NONE:
            return {
                "amc_total_bits_transmitted": 0,
                "amc_avg_cqi": 0.0,
                "amc_avg_bits_per_rb": 0.0,
                "amc_spectral_efficiency_bps_hz": 0.0,
            }

        amc = self._get_amc_module()
        # 1. Если есть amc.get_stats()
        if amc is not None and hasattr(amc, "get_stats") and callable(amc.get_stats):
            try:
                snap = amc.get_stats()
                if isinstance(snap, Mapping):
                    return {
                        "amc_total_bits_transmitted": snap.get("amc_total_bits_transmitted", 0),
                        "amc_avg_cqi": snap.get("amc_avg_cqi", 0.0),
                        "amc_avg_bits_per_rb": snap.get("amc_avg_bits_per_rb", 0.0),
                        "amc_spectral_efficiency_bps_hz": snap.get("amc_spectral_efficiency_bps_hz", 0.0),
                    }
            except Exception:
                pass

        # 2. Извлечение из результатов планирования
        res = getattr(self.scheduler, "_last_schedule_result", None)
        if isinstance(res, Mapping):
            stats = res.get("statistics", {})
            eff_bits = stats.get("total_effective_bits", 0)
            avg_tp = stats.get("average_dl_throughput", 0.0)

            grid = getattr(self.scheduler, "lte_grid", None)
            bw_mhz = getattr(grid, "bandwidth", 10.0) if grid else 10.0
            spec_eff = (avg_tp / (bw_mhz * 1e6)) if bw_mhz > 0 else 0.0

            return {
                "amc_total_bits_transmitted": eff_bits,
                "amc_avg_cqi": round(stats.get("avg_cqi", 0.0), 2),
                "amc_avg_bits_per_rb": round(stats.get("avg_bits_per_rb", 0.0), 2),
                "amc_spectral_efficiency_bps_hz": round(spec_eff, 4),
            }

        return {
            "amc_total_bits_transmitted": 0,
            "amc_avg_cqi": 0.0,
            "amc_avg_bits_per_rb": 0.0,
            "amc_spectral_efficiency_bps_hz": 0.0,
        }

    def collect_detailed(self, tti: int, level: MetricLevel) -> Optional[Dict[Any, Dict[str, Any]]]:
        if level < MetricLevel.FULL:
            return None

        detailed = {}
        amc = self._get_amc_module()
        if amc is not None and hasattr(amc, "get_stats") and callable(amc.get_stats):
            try:
                snap = amc.get_stats()
                for ue_id, u in snap.get("detailed", {}).items():
                    detailed[ue_id] = {
                        "cqi": u.get("cqi", 0),
                        "bits_per_rb": u.get("bits_per_rb", 0),
                        "modulation_order": u.get("modulation_order", 0),
                        "code_rate": u.get("code_rate", 0.0),
                    }
                return detailed
            except Exception:
                pass

        # Извлечение из пользователей планировщика
        res = getattr(self.scheduler, "_last_schedule_result", None)
        if isinstance(res, Mapping):
            stats = res.get("statistics", {})
            u_tps = stats.get("user_throughput", {})
            u_max = stats.get("user_max_throughput", {})
            for ue_id, tp in u_tps.items():
                detailed[ue_id] = {
                    "throughput_bps": tp,
                    "max_throughput_bps": u_max.get(ue_id, 0),
                }

        return detailed if detailed else None


class LegacyPdcchCollector(BaseStatsCollector):
    """
    Адаптер телеметрии PDCCH (Control Channel Elements).
    """

    def __init__(self, scheduler: Any):
        super().__init__(name="pdcch")
        self.scheduler = scheduler

    def _get_pdcch(self):
        return getattr(self.scheduler, "pdcch_manager", None)

    def collect_aggregated(self, tti: int, level: MetricLevel) -> Dict[str, Any]:
        if level == MetricLevel.NONE:
            return {}

        pdcch = self._get_pdcch()
        if pdcch is not None and hasattr(pdcch, "get_stats") and callable(pdcch.get_stats):
            try:
                snap = pdcch.get_stats()
                return {
                    "pdcch_cce_total": snap.get("pdcch_cce_total", 0),
                    "pdcch_cce_used": snap.get("pdcch_cce_used", 0),
                    "pdcch_cce_utilization_pct": snap.get("pdcch_cce_utilization_pct", 0.0),
                    "pdcch_blocked_ue_count": snap.get("pdcch_blocked_ue_count", 0),
                }
            except Exception:
                pass

        return {
            "pdcch_cce_total": 0,
            "pdcch_cce_used": 0,
            "pdcch_cce_utilization_pct": 0.0,
            "pdcch_blocked_ue_count": 0,
        }


class LegacyBufferCollector(BaseStatsCollector):
    """
    Адаптер телеметрии буферов базовой станции и очередей QoS.
    Поддерживает как BaseStation.GET_GLOBAL_BUFFER_STATUS(),
    так и IBufferManager / LayeredBufferManager.
    """

    def __init__(self, buffer_provider: Any, manager: Any = None):
        super().__init__(name="buffer")
        self.buffer_provider = buffer_provider
        self.manager = manager

    def collect_aggregated(self, tti: int, level: MetricLevel) -> Dict[str, Any]:
        if level == MetricLevel.NONE:
            return {
                "buf_occupancy_total_bytes": 0,
                "buf_total_packets": 0,
                "buf_dropped_packets": 0,
                "buf_expired_packets": 0,
                "buf_avg_packet_delay_ms": 0.0,
                "buf_max_hol_delay_ms": 0.0,
            }

        bp = self.buffer_provider
        if bp is None:
            return {}

        # 1. Если это BaseStation с методом GET_GLOBAL_BUFFER_STATUS
        if hasattr(bp, "GET_GLOBAL_BUFFER_STATUS") and callable(bp.GET_GLOBAL_BUFFER_STATUS):
            try:
                status = bp.GET_GLOBAL_BUFFER_STATUS(tti)
                return {
                    "buf_occupancy_total_bytes": status.get("total_size", 0),
                    "buf_total_packets": status.get("total_packets", 0),
                    "buf_dropped_packets": status.get("total_dropped", 0),
                    "buf_expired_packets": status.get("total_expired", 0),
                    "buf_avg_packet_delay_ms": round(status.get("avg_delay", 0.0), 2),
                    "buf_max_hol_delay_ms": round(float(status.get("max_delay", 0.0)), 2),
                }
            except Exception:
                pass

        # 2. Если это BufferManager с методом get_summary()
        if hasattr(bp, "get_summary") and callable(bp.get_summary):
            try:
                summary = bp.get_summary()
                return {
                    "buf_occupancy_total_bytes": summary.get("total_bytes", 0),
                    "buf_total_packets": summary.get("total_packets", 0),
                    "buf_dropped_packets": summary.get("dropped_packets", 0),
                    "buf_expired_packets": summary.get("expired_packets", 0),
                    "buf_avg_packet_delay_ms": round(summary.get("avg_delay_ms", 0.0), 2),
                    "buf_max_hol_delay_ms": round(summary.get("max_hol_delay_ms", 0.0), 2),
                }
            except Exception:
                pass

        # 3. Прямой обход словаря буферов ue_buffers
        buffers = getattr(bp, "ue_buffers", None)
        if isinstance(buffers, Mapping):
            total_size = 0
            total_pkts = 0
            for b in buffers.values():
                if hasattr(b, "GET_UE_STATUS"):
                    st = b.GET_UE_STATUS(tti)
                    total_size += st.get("total_size", 0)
                    total_pkts += st.get("total_packets", 0)
                elif hasattr(b, "sizes"):
                    total_size += sum(b.sizes.values()) if isinstance(b.sizes, dict) else 0

            return {
                "buf_occupancy_total_bytes": total_size,
                "buf_total_packets": total_pkts,
                "buf_dropped_packets": 0,
                "buf_expired_packets": 0,
                "buf_avg_packet_delay_ms": 0.0,
                "buf_max_hol_delay_ms": 0.0,
            }

        return {}

    def collect_detailed(self, tti: int, level: MetricLevel) -> Optional[Dict[Any, Dict[str, Any]]]:
        if level < MetricLevel.FULL:
            return None

        bp = self.buffer_provider
        if bp is None:
            return None

        detailed = {}
        # 1. BaseStation с ue_buffers
        buffers = getattr(bp, "ue_buffers", None)
        if isinstance(buffers, Mapping):
            for ue_id, b in buffers.items():
                if hasattr(b, "GET_UE_STATUS") and callable(b.GET_UE_STATUS):
                    st = b.GET_UE_STATUS(tti)
                    per_ue = st.get("per_ue", {}).get(ue_id, {})
                    detailed[ue_id] = {
                        "buffer_size_bytes": per_ue.get("size", st.get("total_size", 0)),
                        "packet_count": per_ue.get("packet_count", st.get("total_packets", 0)),
                        "avg_delay_ms": per_ue.get("avg_delay", 0.0),
                        "dropped_packets": per_ue.get("dropped", 0),
                    }
                elif hasattr(b, "sizes"):
                    sz = sum(b.sizes.values()) if isinstance(b.sizes, dict) else 0
                    detailed[ue_id] = {
                        "buffer_size_bytes": sz,
                    }
            return detailed if detailed else None

        # 2. BufferManager с get_all_ue_buffers()
        if hasattr(bp, "get_all_ue_buffers") and callable(bp.get_all_ue_buffers):
            for ue_id, ub in bp.get_all_ue_buffers().items():
                detailed[ue_id] = {
                    "buffer_size_bytes": ub.get_size_bytes() if hasattr(ub, "get_size_bytes") else 0,
                    "avg_delay_ms": ub.get_avg_delay_ms() if hasattr(ub, "get_avg_delay_ms") else 0.0,
                }
            return detailed if detailed else None

        return None


# Алиас для обратной совместимости:
LegacyQosCollector = LegacyBufferCollector


# =============================================================================
# Готовые коллекторы физики (Channel, Mobility, Traffic)
# =============================================================================

class ChannelStatsCollector(BaseStatsCollector):
    """Коллектор метрик радиоканала (SINR, CQI, Pathloss)."""

    def __init__(self, channel_source: Any):
        super().__init__(name="channel")
        self.channel_source = channel_source

    def collect_aggregated(self, tti: int, level: MetricLevel) -> Dict[str, Any]:
        if level == MetricLevel.NONE:
            return {}

        sinrs = []
        # Источник может быть UECollection или ChannelModel
        if hasattr(self.channel_source, "GET_ALL_USERS"):
            for ue in self.channel_source.GET_ALL_USERS():
                if hasattr(ue, "SINR") and ue.SINR is not None:
                    sinrs.append(float(ue.SINR))
        elif hasattr(self.channel_source, "get_all_ue_sinr"):
            sinrs = list(self.channel_source.get_all_ue_sinr().values())

        if not sinrs:
            return {"ch_sinr_avg": 0.0, "ch_active_links": 0}

        avg_sinr = sum(sinrs) / len(sinrs)
        res = {
            "ch_sinr_avg": round(avg_sinr, 2),
            "ch_active_links": len(sinrs),
        }
        if level >= MetricLevel.ADVANCED:
            res["ch_sinr_min"] = round(min(sinrs), 2)
            res["ch_sinr_max"] = round(max(sinrs), 2)
        return res

    def collect_detailed(self, tti: int, level: MetricLevel) -> Optional[Dict[Any, Dict[str, Any]]]:
        if level < MetricLevel.FULL:
            return None

        detailed = {}
        if hasattr(self.channel_source, "GET_ALL_USERS"):
            for ue in self.channel_source.GET_ALL_USERS():
                detailed[ue.UE_ID] = {
                    "sinr": round(float(getattr(ue, "SINR", 0.0)), 2),
                    "cqi": getattr(ue, "cqi", 0),
                }
        return detailed if detailed else None


class MobilityStatsCollector(BaseStatsCollector):
    """Коллектор метрик перемещения терминалов (координаты, скорость, дистанция)."""

    def __init__(self, mobility_source: Any):
        super().__init__(name="mobility")
        self.mobility_source = mobility_source

    def collect_aggregated(self, tti: int, level: MetricLevel) -> Dict[str, Any]:
        if level == MetricLevel.NONE:
            return {}

        speeds = []
        distances = []
        if hasattr(self.mobility_source, "GET_ALL_USERS"):
            for ue in self.mobility_source.GET_ALL_USERS():
                if hasattr(ue, "velocity"):
                    speeds.append(float(ue.velocity))
                if hasattr(ue, "dist_to_BS_2D"):
                    distances.append(float(ue.dist_to_BS_2D))

        if not speeds and not distances:
            return {}

        res = {}
        if speeds:
            res["mob_speed_avg_ms"] = round(sum(speeds) / len(speeds), 2)
        if distances:
            res["mob_distance_avg_m"] = round(sum(distances) / len(distances), 1)

        return res

    def collect_detailed(self, tti: int, level: MetricLevel) -> Optional[Dict[Any, Dict[str, Any]]]:
        if level < MetricLevel.FULL:
            return None

        detailed = {}
        if hasattr(self.mobility_source, "GET_ALL_USERS"):
            for ue in self.mobility_source.GET_ALL_USERS():
                pos = getattr(ue, "position", (0.0, 0.0))
                detailed[ue.UE_ID] = {
                    "mob_x": round(float(pos[0]), 2),
                    "mob_y": round(float(pos[1]), 2),
                    "mob_speed_ms": round(float(getattr(ue, "velocity", 0.0)), 2),
                }
        return detailed if detailed else None


# =============================================================================
# Главный фасад: StatsManager (Универсальный реестр коллекторов)
# =============================================================================

class StatsManager:
    """
    Менеджер сбора и экспорта статистики LTE-симуляции.

    Архитектура:
    - Универсальный реестр коллекторов (IStatsCollector).
    - Опрашивает зарегистрированные коллекторы на каждом TTI в соответствии
      с их уровнем детализации (MetricLevel).
    - Автоматически агрегирует системные метрики (history) и поабонентные метрики (detailed_history).
    - Предоставляет экспорт в CSV и JSON с динамическим формированием столбцов.
    """

    def __init__(
        self,
        scheduler: Optional[Any] = None,
        config: Optional[Union[StatisticsConfig, StatsManagerConfig]] = None,
        *,
        buffer_provider: Optional[Any] = None,
    ):
        """
        Инициализация менеджера статистики.

        Args:
            scheduler (Optional[Any]): Планировщик ресурсов (для legacy адаптеров).
            config (Optional[Union[StatisticsConfig, StatsManagerConfig]]): Конфигурация.
            buffer_provider (Optional[Any]): Базовая станция или провайдер буферов.
        """
        if isinstance(config, StatsManagerConfig):
            self.config = config.to_statistics_config()
        elif isinstance(config, StatisticsConfig):
            self.config = config
        else:
            self.config = StatisticsConfig()

        self._collectors: Dict[str, IStatsCollector] = {}

        self.history = deque(maxlen=self.config.history_max_len)
        self.detailed_history = deque(maxlen=self.config.history_max_len)
        self.qos_history = deque(maxlen=self.config.history_max_len)

        self.scheduler = scheduler
        self.buffer_provider = buffer_provider

        # Fallback для buffer_provider из scheduler (временный legacy path этапа E0)
        if self.buffer_provider is None and self.scheduler is not None:
            grid = getattr(self.scheduler, "lte_grid", None)
            bs = getattr(grid, "bs", None)
            if bs is not None:
                warnings.warn(
                    "[StatsManager] Неявное извлечение buffer_provider через scheduler.lte_grid.bs "
                    "является временным legacy fallback (этап E0) и будет удалено в этапе E1. "
                    "Передавайте buffer_provider явно в StatsManager.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if not getattr(bs, "use_simple_buffer", True) and hasattr(bs, "buffer_manager"):
                    self.buffer_provider = getattr(bs, "buffer_manager", None)
                else:
                    self.buffer_provider = bs

        # Автоматическая регистрация адаптеров существующих модулей
        if self.config.auto_register_legacy:
            self._register_legacy_adapters()

    def _register_legacy_adapters(self) -> None:
        """Регистрация встроенных адаптеров для переданных scheduler и buffer_provider."""
        if self.scheduler is not None:
            self.register_collector(LegacySchedulerCollector(self.scheduler))
            self.register_collector(LegacyAmcCollector(self.scheduler))
            self.register_collector(LegacyPdcchCollector(self.scheduler))

        if self.buffer_provider is not None:
            self.register_collector(LegacyBufferCollector(self.buffer_provider, manager=self))

    # -------------------------------------------------------------------------
    # API Реестра коллекторов (Collector Registry)
    # -------------------------------------------------------------------------

    def register_collector(
        self,
        collector: IStatsCollector,
        name: Optional[str] = None,
        level: Optional[Union[MetricLevel, int, str]] = None,
    ) -> None:
        """
        Зарегистрировать коллектор в реестре StatsManager.

        Args:
            collector (IStatsCollector): Экземпляр коллектора.
            name (Optional[str]): Имя коллектора (если None, берется из get_collector_name()).
            level (Optional[Union[MetricLevel, int, str]]): Уровень детализации сбора.
        """
        if name is not None:
            col_name = str(name).lower().strip()
        elif hasattr(collector, "get_collector_name") and callable(collector.get_collector_name):
            col_name = str(collector.get_collector_name()).lower().strip()
        else:
            col_name = collector.__class__.__name__.lower().strip()

        self._collectors[col_name] = collector

        if level is not None:
            self.config.levels.set_level(col_name, MetricLevel.from_value(level))
        elif hasattr(collector, "default_level"):
            # Применяем default_level только если для коллектора нет явно заданного уровня
            if not hasattr(self.config.levels, col_name) and col_name not in self.config.levels.custom_levels:
                self.config.levels.set_level(col_name, getattr(collector, "default_level"))

    def unregister_collector(self, name: str) -> Optional[IStatsCollector]:
        """Удалить коллектор по имени."""
        return self._collectors.pop(name.lower().strip(), None)

    def get_collector(self, name: str) -> Optional[IStatsCollector]:
        """Получить коллектор по имени."""
        return self._collectors.get(name.lower().strip(), None)

    def has_collector(self, name: str) -> bool:
        """Проверить, зарегистрирован ли коллектор с данным именем."""
        return name.lower().strip() in self._collectors

    def set_collector_level(self, name: str, level: Union[MetricLevel, int, str]) -> None:
        """Установить уровень детализации для коллектора."""
        self.config.levels.set_level(name, level)

    @property
    def collectors(self) -> Dict[str, IStatsCollector]:
        """Словарь всех зарегистрированных коллекторов."""
        return dict(self._collectors)

    def record_schedule_result(self, tti: int, sched_result: dict) -> None:
        """
        Передать результат работы планировщика в адаптеры.
        Позволяет LegacySchedulerCollector и LegacyAmcCollector извлекать метрики.
        """
        if hasattr(self.scheduler, "_last_schedule_result"):
            self.scheduler._last_schedule_result = sched_result

        sched_col = self.get_collector("scheduler")
        if isinstance(sched_col, LegacySchedulerCollector):
            sched_col.record_schedule_result(tti, sched_result)

    # -------------------------------------------------------------------------
    # Сбор метрик
    # -------------------------------------------------------------------------

    def collect(self, tti: int) -> None:
        """
        Собрать метрики на текущем TTI со всех зарегистрированных коллекторов.
        """
        if not self.config.enabled:
            return

        snapshot: Dict[str, Any] = {"tti": tti}
        detailed_ue_metrics: Dict[str, Dict[str, Any]] = {}

        for name, collector in list(self._collectors.items()):
            level = self.config.levels.get_level(name, MetricLevel.BASIC)

            if level == MetricLevel.NONE:
                # При NONE некоторые коллекторы возвращают дефолтные нули
                if hasattr(collector, "collect_aggregated"):
                    try:
                        zero_snap = collector.collect_aggregated(tti, MetricLevel.NONE)
                        if isinstance(zero_snap, Mapping):
                            snapshot.update(zero_snap)
                    except Exception:
                        pass
                continue

            # 1. Агрегированные системные метрики
            if hasattr(collector, "collect_aggregated") and callable(collector.collect_aggregated):
                try:
                    agg = collector.collect_aggregated(tti, level)
                    if isinstance(agg, Mapping):
                        snapshot.update(agg)
                except Exception as e:
                    warnings.warn(f"[StatsManager] Ошибка collect_aggregated в коллекторе '{name}': {e}")

            # 2. Поабонентные метрики (FULL)
            if level >= MetricLevel.FULL and hasattr(collector, "collect_detailed") and callable(collector.collect_detailed):
                try:
                    detailed = collector.collect_detailed(tti, level)
                    if isinstance(detailed, Mapping):
                        for ue_id, u_metrics in detailed.items():
                            ue_key = str(ue_id)
                            if ue_key not in detailed_ue_metrics:
                                detailed_ue_metrics[ue_key] = {}
                            if isinstance(u_metrics, Mapping):
                                detailed_ue_metrics[ue_key].update(u_metrics)
                except Exception as e:
                    warnings.warn(f"[StatsManager] Ошибка collect_detailed в коллекторе '{name}': {e}")

            # 3. Специфичные многомерные структуры (QoS и др.)
            if hasattr(collector, "collect_custom_records") and callable(collector.collect_custom_records):
                try:
                    custom = collector.collect_custom_records(tti, level)
                    if custom and isinstance(custom, Mapping) and "qos_ue_metrics" in custom:
                        self.qos_history.append(deepcopy(custom))
                except Exception as e:
                    warnings.warn(f"[StatsManager] Ошибка collect_custom_records в коллекторе '{name}': {e}")

        # Сохранение агрегированного снимка
        self.history.append(deepcopy(snapshot))

        # Сохранение поабонентной детализации
        if detailed_ue_metrics:
            self.detailed_history.append(deepcopy({
                "tti": tti,
                "ue_metrics": detailed_ue_metrics,
            }))

    def get_history(self) -> List[Dict[str, Any]]:
        """Вернуть независимую копию агрегированной истории."""
        return deepcopy(list(self.history))

    def get_detailed_history(self) -> List[Dict[str, Any]]:
        """Вернуть независимую копию детальной истории."""
        return deepcopy(list(self.detailed_history))

    def clear_history(self) -> None:
        """Очистить накопленную историю."""
        self.history.clear()
        self.detailed_history.clear()
        self.qos_history.clear()

    def calculate_fairness(self, ue_throughputs: dict) -> dict:
        """Расчет коэффициентов справедливости (Jain's index, Gini)."""
        return calculate_fairness_dict(ue_throughputs)

    # -------------------------------------------------------------------------
    # Экспорт телеметрии
    # -------------------------------------------------------------------------

    def export_all(self, locale: str = "ru") -> List[str]:
        """Экспортировать всю накопленную статистику и вернуть список созданных файлов."""
        paths = []
        prefix = self.config.file_prefix
        out_dir = Path(self.config.output_dir)

        if self.history:
            filename = str(out_dir / f"{prefix}.{self.config.export_format}")
            if self.config.export_format == "json":
                self.export_json(filename)
            else:
                self.export_csv(filename, locale=locale)
            paths.append(filename)

        if self.detailed_history:
            filename = str(out_dir / f"{prefix}_detailed.{self.config.export_detailed_format}")
            if self.config.export_detailed_format == "json":
                self.export_detailed_json(filename)
            else:
                self.export_detailed_csv(filename, locale=locale)
            paths.append(filename)

        if self.qos_history:
            filename = str(out_dir / f"{prefix}_qos.csv")
            self.export_qos_csv(filename, locale=locale)
            paths.append(filename)

        return paths

    def export_csv(self, filename: Optional[str] = None, locale: str = "ru") -> None:
        """
        Экспорт агрегированной истории в CSV.
        Заголовки формируются динамически на основе ключей зарегистрированных коллекторов.
        """
        if not self.history:
            print("[StatsManager] Нет данных для экспорта (history пуста)")
            return

        filename = filename or f"{self.config.file_prefix}.csv"

        # Сбор уникальных заголовков с сохранением порядка
        headers = []
        for snapshot in self.history:
            for k in snapshot.keys():
                if k not in headers:
                    headers.append(k)

        delimiter = ";" if locale == "ru" else ","
        rows = []
        for snapshot in self.history:
            row = {}
            for key in headers:
                val = snapshot.get(key, 0)
                if isinstance(val, dict):
                    row[key] = json.dumps(val, ensure_ascii=False)
                elif isinstance(val, float):
                    row[key] = str(val).replace(".", ",") if locale == "ru" else val
                else:
                    row[key] = val
            rows.append(row)

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[StatsManager] Экспортировано {len(rows)} снимков TTI в {filename}")

    def export_json(self, filename: Optional[str] = None) -> None:
        """Экспорт агрегированной истории в JSON."""
        filename = filename or f"{self.config.file_prefix}.json"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as stream:
            json.dump(self.get_history(), stream, ensure_ascii=False, indent=2)

    def export_detailed_csv(self, filename: Optional[str] = None, locale: str = "ru") -> None:
        """
        Экспорт детальной поабонентной истории в CSV.
        Динамически включает столбцы из любых коллекторов.
        """
        if not self.detailed_history:
            return

        filename = filename or f"{self.config.file_prefix}_detailed.csv"
        delimiter = ";" if locale == "ru" else ","

        standard_order = [
            "tti", "ue_id", "rb_allocated", "throughput_bps",
            "buffer_size_bytes", "cqi", "sinr", "cce_allocated",
        ]
        extra_keys = set()
        for snap in self.detailed_history:
            for u_metrics in snap.get("ue_metrics", {}).values():
                for k in u_metrics.keys():
                    if k not in standard_order:
                        extra_keys.add(k)

        fieldnames = standard_order + sorted(extra_keys)

        rows = []
        for snap in self.detailed_history:
            tti = snap.get("tti", 0)
            for ue_id, u_metrics in snap.get("ue_metrics", {}).items():
                row = {"tti": tti, "ue_id": ue_id}
                for k in fieldnames:
                    if k in ("tti", "ue_id"):
                        continue
                    val = u_metrics.get(k, 0)
                    if isinstance(val, float):
                        row[k] = str(val).replace(".", ",") if locale == "ru" else val
                    else:
                        row[k] = val
                rows.append(row)

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[StatsManager] Экспортировано {len(rows)} детальных записей в {filename}")

    def export_detailed_json(self, filename: Optional[str] = None) -> None:
        """Экспорт детальной истории в JSON."""
        filename = filename or f"{self.config.file_prefix}_detailed.json"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as stream:
            json.dump(self.get_detailed_history(), stream, ensure_ascii=False, indent=2)

    def export_qos_csv(self, filename: Optional[str] = None, locale: str = "ru") -> None:
        """Экспорт специфичной статистики QoS (per-UE-per-QCI) в CSV."""
        if not self.qos_history:
            return

        filename = filename or f"{self.config.file_prefix}_qos.csv"
        delimiter = ";" if locale == "ru" else ","

        fieldnames = [
            "tti", "ue_id", "qci", "tx_packets", "tx_bytes",
            "dropped_packets", "dropped_bytes", "avg_delay_ms",
        ]

        rows = []
        for snap in self.qos_history:
            tti = snap.get("tti", 0)
            for item in snap.get("qos_ue_metrics", []):
                row = {"tti": tti}
                for fld in fieldnames:
                    if fld == "tti":
                        continue
                    val = item.get(fld, 0)
                    if isinstance(val, float):
                        row[fld] = str(val).replace(".", ",") if locale == "ru" else val
                    else:
                        row[fld] = val
                rows.append(row)

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)


__all__ = [
    "MetricLevel",
    "IStatsCollector",
    "BaseStatsCollector",
    "LevelsConfig",
    "StatisticsConfig",
    "StatsManagerConfig",
    "StatsManager",
    "LegacySchedulerCollector",
    "LegacyAmcCollector",
    "LegacyPdcchCollector",
    "LegacyBufferCollector",
    "LegacyQosCollector",
    "ChannelStatsCollector",
    "MobilityStatsCollector",
    "calculate_fairness_dict",
    "calculate_std",
    "calculate_cv",
]
