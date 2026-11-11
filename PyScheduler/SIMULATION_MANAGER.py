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

import sys
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import GLOBALS
import numpy as np
from BS_MODULE import BaseStation
from RES_GRID import RES_GRID_LTE
from SCHEDULER import SchedulerInterface
from TRAFFIC_MODEL import PacketManager, SimpleGenerator, QCI
from UE_MODULE import UECollection
from MOBILITY_MODEL import MapBorders
from tqdm import tqdm
from STATS_MANAGER import (
    MetricLevel, LevelsConfig, StatisticsConfig, StatsManagerConfig, StatsManager,
)

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
    mobility_update_interval: int = 500
    channel_update_interval: int = 1
    use_legacy_traffic: bool = True
    map_x_min: float = -500
    map_x_max: float = 500
    map_y_min: float = -500
    map_y_max: float = 500
    stats_log: bool = False
    verbose: bool = False


@dataclass
class SchedulerConfig:
    """
    Конфигурация планировщика.

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
        self.exported_stats_files = []

        self._to_file = False
        self._log_file = None
        self._original_stdout = None
        self._original_stderr = None
        self.scheduler = None

        self.traffic_gen = None

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

    def set_mobility_interval(self, mobility_update_interval: int) -> None:
        """
        Установить интервал обновления позиции UE (модели мобильности).

        Args:
            mobility_update_interval (int): Интервал в мс. Рекомендуется 100–1000 мс.
        """
        if not isinstance(mobility_update_interval, int) or mobility_update_interval <= 0:
            raise ValueError(
                f"Интервал обновления мобильности должен быть положительным целым числом. "
                f"Получено: {mobility_update_interval}"
            )
        self.sim_config.mobility_update_interval = mobility_update_interval

    def set_channel_interval(self, channel_update_interval: int) -> None:
        """
        Установить интервал обновления качества канала (SINR/CQI).

        Args:
            channel_update_interval (int): Интервал в мс. Рекомендуется 1–100 мс.
        """
        if not isinstance(channel_update_interval, int) or channel_update_interval <= 0:
            raise ValueError(
                f"Интервал обновления канала должен быть положительным целым числом. "
                f"Получено: {channel_update_interval}"
            )
        self.sim_config.channel_update_interval = channel_update_interval

    #TODO: возможно эти интервальные сеттеры можно объединить в один метод

    def set_map_borders(self, x_min: float, x_max: float,
                        y_min: float, y_max: float) -> None:
        """
        Устанавливает границы карты для моделей мобильности и визуализации.
        Сбрасывает синглтон MapBorders перед установкой новых значений.
        """
        MapBorders._instance = None
        MapBorders(x_min, x_max, y_min, y_max)

        self.sim_config.map_x_min = x_min
        self.sim_config.map_x_max = x_max
        self.sim_config.map_y_min = y_min
        self.sim_config.map_y_max = y_max

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

        if self.base_station.use_simple_buffer:
            self.sim_config.use_legacy_traffic = True
            self.traffic_gen = SimpleGenerator()
        else:
            self.sim_config.use_legacy_traffic = False
            self.traffic_gen = PacketManager()

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
            manager.set_stats_manager(enabled=True, collect_interval=5, amc_level="advanced")
        """
        unknown = set(kwargs) - self.stats_config.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown statistics settings: {sorted(unknown)}")
        # Validate before replacing the live configuration.
        candidate = replace(self.stats_config, **kwargs)
        candidate.to_statistics_config()
        self.stats_config = candidate

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
        self.stats_manager = None
        self.exported_stats_files = []
        self._check_required_parameters()
        stats_config = (self.stats_config.to_statistics_config()
                        if self.stats_config.enabled else None)

        JFI_INTERVAL_TTI    = 500
        STATS_REFRESH_TTI   = 100

        progress_stream = getattr(sys, "__stderr__", sys.stderr)

        sbar        = None
        pbar        = None
        jfi_cached  = None

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

            # Инициализация буферов пользователей
            for ue in self.ue_collection.GET_ALL_USERS():
                # Режим Simple Buffer
                if self.sim_config.use_legacy_traffic:
                    if not self.base_station.buffer_manager.ue_has_buffer(ue.UE_ID):
                        self.base_station.buffer_manager.create_ue_buffer(
                            ue.UE_ID, self.base_station.per_ue_max
                        )
                # Режим Layered Buffer
                else:
                    bearers_info = self.traffic_gen.get_bearers_info(ue.UE_ID)
                    self.base_station.buffer_manager.create_ue_buffer(
                        ue.UE_ID, self.base_station.per_ue_max, bearers_info
                    )

            # Инициализация менеджера статистики
            self.stats_manager = None
            if self.stats_config.enabled:
                buffer_provider = (None if self.base_station.use_simple_buffer
                                   else self.base_station.buffer_manager)
                self.stats_manager = StatsManager(
                    self.scheduler, stats_config, buffer_provider=buffer_provider)

                if self.sim_config.verbose:
                    print(
                        f"[SIMULATION] StatsManager enabled "
                        f"(interval={self.stats_config.collect_interval} TTI, "
                        f"scheduler={self.stats_config.scheduler_level}, "
                        f"amc={self.stats_config.amc_level})"
                    )

            pbar = tqdm(
                total=self.sim_config.sim_duration,
                desc="Simulation Progress",
                unit="TTI",
                dynamic_ncols=True,
                colour='cyan',
                bar_format=(
                    "{desc}: {percentage:3.0f}%|{bar}| "
                    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                ),
                file=progress_stream,
                position = 0,
                leave = True,
            )

            sbar = tqdm(
                total=0,
                desc="Stats",
                bar_format="{desc}",
                dynamic_ncols=True,
                file=progress_stream,
                position=1,
                leave=True,
                )

            # Основной цикл симуляции
            for tti in range(self.sim_config.sim_duration):
                if self.sim_config.verbose:
                    print(f"\n[SIMULATION] Start TTI {tti}...")

                GLOBALS.CURRENT_TIME = tti

                _ = self.run_tti(current_time=tti)

                pbar.update(1)
                if self.stats_manager and (tti % JFI_INTERVAL_TTI == 0):
                    ue_avg_throughputs = {
                        ue.UE_ID: ue.average_throughput
                        for ue in self.ue_collection.GET_ALL_USERS()
                    }
                    longterm_fairness_metrics = self.stats_manager.calculate_fairness(
                        ue_throughputs=ue_avg_throughputs
                    )
                    jfi_cached = longterm_fairness_metrics.get("dl_fairness_jain_index", None)

                if (tti % STATS_REFRESH_TTI == 0) and self.scheduler:
                    s = self.scheduler.get_stats()
                    a = self.scheduler.amc.get_stats()

                    tput = a.get('dl_throughput_sum_kbps', 0)
                    prb = s.get("dl_prb_utilization_pct", 0.0)
                    ue_cnt = s.get("sch_active_ue_count", 0)
                    us = s.get("sch_total_time_us", 0.0)
                    jfi_str = f"{jfi_cached:.4f}" if jfi_cached is not None else "N/A"

                    sbar.set_description_str(
                        f"TTI={tti} | UE={ue_cnt} | "
                        f"Tput={tput:.0f} kbps | PRB={prb:.1f}% | "
                        f"JFI(long)={jfi_str} | sch_total_time_us={us:.1f}"
                    )
                    sbar.refresh()

                # Вывод статистики в CSV файл
                if self.stats_manager and tti % self.stats_manager.config.collect_interval == 0:
                    self.stats_manager.collect(tti)

            if self.stats_manager:
                self.exported_stats_files = self.stats_manager.export_all(locale="ru")

                # Отладочный момент для глобального JI (Fairness)
                ue_avg_throughputs = {}
                for ue in self.ue_collection.GET_ALL_USERS():
                    ue_avg_throughputs[ue.UE_ID] = ue.average_throughput

                longterm_fairness_metrics = self.stats_manager.calculate_fairness(
                    ue_throughputs=ue_avg_throughputs
                )
                jfi = longterm_fairness_metrics["dl_fairness_jain_index"]
                jfi_str = f"{jfi:.4f}" if jfi is not None else "N/A (no throughput data)"
                print(f"[SIMULATION] Jain's Fairness Index: {jfi_str}")

        finally:
            # Возвращение консольного вывода
            if pbar is not None:
                pbar.close()
            if sbar is not None:
                sbar.close()
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

    def setup_ue_traffic(self, ue_id: int, model_type: str, qci: Optional[QCI | int] = None, **params):
        """
        Задать модель генерации трафика для пользователя. Работает как для старого SimpleGenerator,
        так и для нового PacketManager.
        1) При SimpleGenerator: задаёт одну модель генерации трафика для пользователя.
        2) При PacketManager: задаёт модель генерации трафика для выбранного QCI пользователя.
        Если не указать значение QCI, то модель будет создана для QCI по умолчанию (QCI 9).

        Args:
            ue_id (int):  Уникальный идентификатор UE.
            model_type (str): Тип модели ('Poisson', 'OnOff', 'MMPP').
            qci (Optional[QCI | int], optional): Идентификатор класса QoS. По умолчанию None.
            **params: Параметры модели генерации трафика.

        """
        self.traffic_gen.set_model(ue_id=ue_id, model_type=model_type, qci=qci, **params)

    def setup_traffic_profiles(self, traffic_profiles: List[Dict[str, Any]]):
        """
        Инициализирует и настраивает профили трафика (bearers) для списка пользователей.

        Принимает конфигурацию, проходит по каждому UE и создает соответствующие
        генераторы трафика через PacketManager.

        Args:
            traffic_profiles (List[Dict[str, Any]]): Список конфигураций для пользователей.
                Каждый элемент списка должен иметь следующую структуру:

                [
                    {
                        "ue_id": int,       # ID существующего пользователя
                        "bearers": [        # Список биреров (потоков) для этого UE
                            {
                                # Обязательные параметры:
                                "model_type": str,           # "Poisson", "OnOff", "MMPP"
                                "qci": QCI | int,            # QCI для данного bearer

                                # Параметры для модели Poisson:
                                "packet_rate": int,          # Частота пакетов (пак/сек)

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
            ValueError: Если данная функция вызывается при режиме Simple Buffer, не передано
            значение ue_id в конфигурации трафика или отсутсвует настройка трафика для bearer'ов
            в конфигурации.

        """
        if self.sim_config.use_legacy_traffic:
            raise ValueError(
                "This function is intended only for configuring traffic in Layered Buffer mode. " 
                "To configure the traffic for the Simple Buffer mode, use the setup_ue_traffic "
                "function."
            )

        for profile in traffic_profiles:
            ue_id = profile.get("ue_id")

            if ue_id is None:
                raise ValueError(
                    "The ue_id value is missing for the traffic profile."
                )

            bearers = profile.get("bearers", [])

            if bearers is None:
                raise ValueError(
                    f"The bearers list is missing for the UE {ue_id} traffic profile."
                )

            for bearer in bearers:
                model_type = bearer.get("model_type")
                qci = bearer.get("qci")

                model_params = {k: v for k, v in bearer.items() if k not in ("model_type", "qci")}

                self.traffic_gen.set_model(ue_id=ue_id, model_type=model_type, qci=qci, **model_params)


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
                mobility_update_interval=self.sim_config.mobility_update_interval,
                channel_update_interval=self.sim_config.channel_update_interval,
            )

        # Блок обновления буферов и генерации трафика
        # Обновляем буферы всех пользователей
        self.base_station.buffer_manager.upd_buffers_all()

        all_users = self.ue_collection.GET_ALL_USERS()
        for ue in all_users:
            ue_id = ue.UE_ID

            # Пропускаем UE без модели
            if self.sim_config.use_legacy_traffic:
                if ue_id not in self.traffic_gen.models:
                    continue
            else:
                if ue_id not in self.traffic_gen.ue_profiles:
                    continue

            # Генерируем пакеты по одному юзеру
            packets = self.traffic_gen.generate_packets(
                ue_id=ue_id, current_time=current_time, update_interval=self.sim_config.update_interval
            )

            # Кладем пакеты в буфер
            if packets:
                for pkt in packets:
                    self.base_station.buffer_manager.add_packet(ue_id, pkt)

        # 3. Планировщик
        users = self.ue_collection.GET_USERS_FOR_SCHEDULER()
        sched_result = self.scheduler.schedule(current_time, users)

        return sched_result

__all__ = [
    "SimulationConfig", "SchedulerConfig", "StatsManagerConfig",
    "SimulationManager", "StatsManager", "MetricLevel", "LevelsConfig", "StatisticsConfig",
]
