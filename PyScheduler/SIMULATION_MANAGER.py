"""
#------------------------------------------------------------------------------
# Модуль: SIMULATION_MANAGER - Менеджер управления симуляцией.
#------------------------------------------------------------------------------
# Описание:
# Отвечает за настройку параметров симуляции, запуск основного цикла симуляции,
# а также логирование статистик в файл.
#
# Версия: 1.0.0
# Дата последнего изменения: 2025-10-31
# Автор: Норицин Иван
# Версия Python Kernel: 3.12.9
#------------------------------------------------------------------------------
"""
import csv
import sys
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

import GLOBALS
from UE_MODULE import UECollection
from BS_MODULE import BaseStation
from RES_GRID import RES_GRID_LTE
from SCHEDULER import SchedulerInterface

from enum import Enum
from collections import deque

#==============================================================================
#                          ОБРАБОТЧИК СТАТИСТИКИ
#==============================================================================

class MetricLevel(Enum):
    """
    Уровни детализации сбора метрик.

    - NONE: Не собирать (0% overhead)
    - BASIC: Только собираемые метрики (total, avg)
    - DETAILED: Собираемые метрики и расчетные (per-UE Dict, Jain's e.t.c)
    """
    NONE = 0
    BASIC = 1
    DETAILED = 2

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
    pdcch: MetricLevel = MetricLevel.NONE

@dataclass
class StatisticsConfig:
    """
    Основная конфигурация системы статистики.

    Attributes:
        collect_interval (int): Интервал сбора в TTI (напр., 10 = каждые 10 TTI)
        levels (LevelsConfig): Уровни детализации per-метрика
        export_format (str): Формат экспорта ('csv' или 'json')
        history_maxlen (int): Максимум snapshots в истории (для ограничения памяти)
        file_prefix (str): Префикс имени файла для экспорта
    """
    collect_interval: int = 10
    levels: LevelsConfig = None
    export_format: str = "csv"
    history_maxlen: int = 10000
    file_prefix: str = "lte_stats"

    def __post_init__(self):
        """Инициализация levels, если не передан"""
        if self.levels is None:
            self.levels = LevelsConfig()

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
        self.history = deque(maxlen=self.config.history_maxlen)  # История snapshots

    def collect(self, tti: int) -> None:
        """
        Собрать метрики на текущем TTI.
        Опрашивает scheduler.get_stats() и amc.get_stats() в зависимости от
        уровней в config.levels, объединяет в один Dict (snapshot) и добавляет
        в self.history.

        Args:
            tti (int): Номер текущего TTI (для timestamp)
        """
        snapshot = {"tti": tti}  # Базовый Dict с timestamp

        # 1. Scheduler metrics
        if self.config.levels.scheduler != MetricLevel.NONE:
            sched_stats = self.scheduler.get_stats()

            if self.config.levels.scheduler == MetricLevel.BASIC:
                snapshot.update({
                    "eligible_ue_count": sched_stats["eligible_ue_count"],
                    "active_ue_count": sched_stats["active_ue_count"],
                    "allocated_rb_count": sched_stats["allocated_rb_count"],
                    "allocation_efficiency": sched_stats["allocation_efficiency"]
                })
            else:  # DETAILED
                # Все метрики (включая rb_distribution, если добавлен)
                snapshot.update(sched_stats)

        # 2. AMC metrics
        if self.config.levels.amc != MetricLevel.NONE:
            amc_stats = self.scheduler.amc.get_stats()
            print(f"[DEBUG AMC] amc_stats: {amc_stats}") ############

            if self.config.levels.amc == MetricLevel.BASIC:
                snapshot.update({
                    "total_throughput_bps": amc_stats.get("total_throughput", 0),  # ← Fallback на 0
                    "avg_bits_per_rb": amc_stats.get("avg_bits_per_rb", 0.0)
                })

            else:  # DETAILED
                # Включая per-UE throughputs
                snapshot.update(amc_stats)

        # 3. PDCCH metrics (опционально)
        if self.config.levels.pdcch != MetricLevel.NONE:
            pdcch_stats = self.scheduler.pdcch_manager.get_stats()
            snapshot["pdcch"] = pdcch_stats  # Вложенный Dict

        # Добавить snapshot в историю
        self.history.append(snapshot)

    def export_csv(self, filename: str = None) -> None:
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
        rows = []

        for snapshot in self.history:
            row = {}
            for key, value in snapshot.items():
                if isinstance(value, dict):
                    row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    row[key] = value
            rows.append(row)

        headers = sorted({header for row in rows for header in row.keys()})

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        print(f"StatsManager: Exported {len(rows)} snapshots to {filename}")

    def get_summary(self) -> Dict:
        """
        Получить агрегированную статистику по всей истории.
        Рассчитывает среднее/макс/мин для ключевых метрик (throughput, RB, UE).
        Полезно для быстрого анализа эффективности симуляции.

        Returns:
            Dict: Агрегированные метрики {
                "total_ttis_collected": int,
                "avg_throughput_bps": float,
                "max_throughput_bps": float,
                "min_throughput_bps": float,
                "avg_active_ue_count": float,
                "first_tti": int,
                "last_tti": int
            }
        """
        if not self.history:
            return {"error": "No data collected"}

        # Извлечение списков значений для метрик
        throughputs = [s.get("total_throughput_bps", 0) for s in self.history]
        active_ue_counts = [s.get("active_ue_count", 0) for s in self.history]
        allocated_rbs = [s.get("allocated_rb_count", 0) for s in self.history]

        # Агрегация
        summary = {
            "total_ttis_collected": len(self.history),

            # Throughput
            "avg_throughput_bps": sum(throughputs) / len(throughputs) if throughputs else 0,
            "max_throughput_bps": max(throughputs) if throughputs else 0,
            "min_throughput_bps": min(throughputs) if throughputs else 0,

            # Active UE
            "avg_active_ue_count": sum(active_ue_counts) / len(active_ue_counts) if active_ue_counts else 0,
            "max_active_ue_count": max(active_ue_counts) if active_ue_counts else 0,

            # RB allocation
            "avg_allocated_rbs": sum(allocated_rbs) / len(allocated_rbs) if allocated_rbs else 0,

            # Временной диапазон
            "first_tti": self.history[0]["tti"],
            "last_tti": self.history[-1]["tti"]
        }

        return summary

#==============================================================================
#                          КОНФИГУРАТОР СИМУЛЯЦИИ
#==============================================================================

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
        scheduler_level (str): Уровень для scheduler ('none'/'basic'/'detailed')
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
    file_prefix: str = "manager_stats"


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
                available = ', '.join(self.stats_config.__dataclass_fields__.keys())
                raise ValueError(
                    f"'{key}' не является параметром StatsManagerConfig. "
                    f"Доступные: {available}"
                )

    def enable_stats_log(self):
        """
        Включить логирование статистики симуляции в CSV-файл.

        """
        self.sim_config.stats_log = True

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
                print(f"[SIMULATION] Calculated number of frames for resource "
                      f"grid: {num_frames}")

            # Создание ресурсной сетки
            lte_grid = RES_GRID_LTE(
                bandwidth=self.base_station.bandwidth,
                num_frames=num_frames
            )
            if self.sim_config.verbose and lte_grid:
                print(f"[SIMULATION] The resource grid has been initialized. "
                      f"Bandwidth={lte_grid.bandwidth} MHz. RBs={lte_grid.rb_per_slot}")

            # Создание планировщика
            scheduler = SchedulerInterface.create(
                algorithm=self.sched_config.algorithm,
                lte_grid=lte_grid,
                bs=self.base_station,
                max_dl_ue_tti=self.sched_config.max_dl_ue_tti,
                pcfich=self.sched_config.pcfich,
                max_dl_cce_allowance=self.sched_config.max_dl_cce_allowance,
                verbose_pdcch=self.sim_config.verbose,
                window_size=self.sched_config.window_size,
                enable_window=self.sched_config.enable_window,
                verbose=self.sim_config.verbose
            )

            # Инициализация менеджера статистики
            if self.stats_config.enabled:
                level_map = {
                    "none": MetricLevel.NONE,
                    "basic": MetricLevel.BASIC,
                    "detailed": MetricLevel.DETAILED
                }

                # Создание конфига для StatsManager
                stats_config = StatisticsConfig(
                    collect_interval=self.stats_config.collect_interval,
                    levels=LevelsConfig(
                        scheduler=level_map[self.stats_config.scheduler_level],
                        amc=level_map[self.stats_config.amc_level],
                        pdcch=level_map[self.stats_config.pdcch_level]
                    ),
                    export_format=self.stats_config.export_format,
                    file_prefix=self.stats_config.file_prefix
                )

                self.stats_manager = StatsManager(scheduler, stats_config)

                if self.sim_config.verbose:
                    print(f"[SIMULATION] StatsManager enabled "
                          f"(interval={self.stats_config.collect_interval} TTI, "
                          f"scheduler={self.stats_config.scheduler_level}, "
                          f"amc={self.stats_config.amc_level})")

            # Основной цикл симуляции
            for tti in range(self.sim_config.sim_duration):

                if self.sim_config.verbose:
                    print(f"\n[SIMULATION] Start TTI {tti}...")

                # Обновление глобальной переменной текущего времени
                GLOBALS.CURRENT_TIME = tti

                # Условие для обновления состояния пользователей
                if tti % self.sim_config.update_interval == 0:

                    if self.sim_config.verbose:
                        print("[SIMULATION] Update UEs states")

                    # Обновление состояния пользователей
                    self.ue_collection.UPDATE_ALL_USERS(
                        current_time=tti,
                        update_interval=self.sim_config.update_interval
                    )

                # Подготовка данных для планировщика
                users = self.ue_collection.GET_USERS_FOR_SCHEDULER()

                # Планирование ресурсов
                sched_result = scheduler.schedule(tti, users)

                # Вывод статистики в CSV файл
                if self.sim_config.stats_log:
                    self._stats_logging(sched_result)

                if self.stats_manager and tti % self.stats_manager.config.collect_interval == 0:
                    self.stats_manager.collect(tti)

            if self.stats_manager:
                # Экспорт в CSV
                output_filename = f"{self.stats_config.file_prefix}.csv"
                self.stats_manager.export_csv(output_filename)

                # Вывод summary (если verbose включен)
                if self.sim_config.verbose:
                    summary = self.stats_manager.get_summary()
                    print(f"\n[StatsManager] Collected {summary.get('total_ttis_collected', 0)} snapshots, "
                          f"avg throughput {summary.get('avg_throughput_bps', 0) / 1e6:.2f} Mbps, "
                          f"max throughput {summary.get('max_throughput_bps', 0) / 1e6:.2f} Mbps")

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
            msg = (
                "Ошибка: не все обязательные параметры симуляции заданы:\n"
                + "\n".join(errors)
            )
            raise RuntimeError(msg)

    def _stats_logging(self, sched_result: dict) -> None:
        """
        Логирование статистики симуляции в CSV-файл (stats.csv).

        Args:
            sched_result (dict): Результаты работы планировщика.

        """
        allocation = sched_result["allocation"]
        pdcch_allocation = sched_result["pdcch_stats"]["allocations"]

        filename = 'stats.csv'

        # Названия колонок при первом запуске
        if not hasattr(self, "_stats_file_initialized"):
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["TTI", "UE_ID", "Num_RBs", "RBs", "Num_CCE",
                                 "Tx_Bits", "Buffer_Size" ,"CQI", "SINR"])
            self._stats_file_initialized = True

        with open(filename, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            # Сбор данных
            for ue in self.ue_collection.GET_ALL_USERS():
                ue_id = ue.UE_ID
                num_rbs = len(allocation.get(ue_id, []))
                rbs = allocation.get(ue_id, 0)
                num_cce = pdcch_allocation.get(ue_id, 0)
                tx_bits = int(ue.last_transmitted_bits)
                buf_size = self.base_station.ue_buffers[ue_id].sizes[ue_id] * 8
                cqi = ue.cqi
                sinr = round(ue.SINR, 4)

                # Запись в файл
                row = [GLOBALS.CURRENT_TIME, ue_id, num_rbs, rbs, num_cce,
                       tx_bits, buf_size, cqi, sinr]
                writer.writerow(row)
