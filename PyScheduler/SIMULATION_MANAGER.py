"""
Модуль SIMULATION_MANAGER: оркестратор LTE-симуляции.

===============================================================================
Архитектурный статус:
- Этап E0 (Extraction): Полный вынос телеметрии в StatsManager.
- Инверсия управления (IoC): SimulationManager является чистым оркестратором цикла
  и НЕ создает внутри себя ресурсную сетку, планировщик и менеджер статистики.
  Все компоненты инжектируются извне через сеттеры (или конструктор) в готовом виде.
===============================================================================
"""

import inspect
import sys
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Union

import GLOBALS
import numpy as np
from tqdm import tqdm

from BS_MODULE import BaseStation
from RES_GRID import RES_GRID_LTE

# Импорт планировщика и его интерфейса
try:
    from SCHEDULER import SchedulerInterface
except ImportError:
    try:
        from RES_GRID import SchedulerInterface
    except ImportError:
        SchedulerInterface = object

try:
    from SCHEDULER import RoundRobinScheduler
except ImportError:
    RoundRobinScheduler = None

try:
    from SCHEDULER import BestCQIScheduler
except ImportError:
    BestCQIScheduler = None

try:
    from SCHEDULER import ProportionalFairScheduler
except ImportError:
    ProportionalFairScheduler = None

try:
    from SCHEDULER import ProportionalFairScheduler_v2
except ImportError:
    ProportionalFairScheduler_v2 = None

from UE_MODULE import UECollection, UserEquipment

# Экспорт контрактов и менеджера телеметрии
from STATS_MANAGER import (
    BaseStatsCollector,
    ChannelStatsCollector,
    IStatsCollector,
    LevelsConfig,
    MetricLevel,
    MobilityStatsCollector,
    StatisticsConfig,
    StatsManager,
    StatsManagerConfig,
)


# =============================================================================
# Конфигурация сценария симуляции
# =============================================================================

@dataclass
class SimulationConfig:
    """Конфигурация параметров прогона симуляции."""

    sim_duration: Optional[int] = 20  # Длительность в миллисекундах (TTI)
    update_interval: int = 1          # Интервал обновления физики/канала/буферов (мс)
    mobility_update_interval: int = 1
    channel_update_interval: int = 1
    bandwidth: int = 10               # Ширина полосы (МГц)
    verbose: bool = False
    enable_traffic_gen: bool = True
    use_legacy_traffic: bool = True


@dataclass
class SchedulerConfig:
    """Конфигурация параметров планирования (для передачи параметров алгоритму)."""

    algorithm: Optional[str] = "RoundRobin"
    max_dl_ue_tti: Optional[int] = None
    pcfich: int = 2
    max_dl_cce_allowance: Optional[int] = None
    verbose_pdcch: bool = False
    window_size: int = 10
    enable_window: bool = True
    verbose: bool = False


# =============================================================================
# Класс SimulationManager
# =============================================================================

class SimulationManager:
    """
    Главный оркестратор LTE-симулятора.
    Отвечает исключительно за координацию цикла симуляции по времени (TTI).
    Не инстанциирует внутри себя инфраструктурные объекты (сетку, планировщик, StatsManager),
    а получает их уже готовыми через методы-сеттеры.
    """

    def __init__(
        self,
        sim_config: Optional[SimulationConfig] = None,
        sched_config: Optional[SchedulerConfig] = None,
        stats_config: Optional[StatsManagerConfig] = None,
    ):
        self.sim_config = sim_config or SimulationConfig()
        self.sched_config = sched_config or SchedulerConfig()
        self.stats_config = stats_config or StatsManagerConfig()

        self.ue_collection: Optional[UECollection] = None
        self.base_station: Optional[BaseStation] = None
        self.lte_grid: Optional[RES_GRID_LTE] = None
        self.scheduler: Optional[Any] = None
        self.stats_manager: Optional[StatsManager] = None
        self.traffic_gen: Optional[Any] = None

        self._pending_collectors: List[Tuple[IStatsCollector, Optional[MetricLevel]]] = []
        self.exported_stats_files: List[str] = []

        self._to_file: bool = False
        self._log_file = None
        self._original_stdout = None
        self._original_stderr = None

    # -------------------------------------------------------------------------
    # Методы настройки и передачи готовых компонентов
    # -------------------------------------------------------------------------

    def set_sim_duration(self, sim_duration: int) -> None:
        """Установить длительность симуляции в миллисекундах (TTI)."""
        if not isinstance(sim_duration, int) or sim_duration <= 0:
            raise ValueError(f"Длительность симуляции должна быть положительным int, получено: {sim_duration}")
        self.sim_config.sim_duration = sim_duration

    def set_sim_config(self, config: SimulationConfig) -> None:
        """Задать конфигурацию симуляции."""
        if not isinstance(config, SimulationConfig):
            raise TypeError("Ожидается экземпляр SimulationConfig")
        self.sim_config = config

    def set_ue_collection(self, ue_collection: UECollection) -> None:
        """Установить готовую коллекцию пользовательских устройств (UE)."""
        if not isinstance(ue_collection, UECollection):
            raise TypeError(f"Ожидается UECollection, получено: {type(ue_collection).__name__}")
        self.ue_collection = ue_collection

    def set_base_station(self, base_station: BaseStation) -> None:
        """Установить готовую базовую станцию (eNodeB)."""
        if not isinstance(base_station, BaseStation):
            raise TypeError(f"Ожидается BaseStation, получено: {type(base_station).__name__}")
        self.base_station = base_station
        if hasattr(base_station, "use_simple_buffer"):
            self.sim_config.use_legacy_traffic = base_station.use_simple_buffer

    def set_mobility_interval(self, mobility_update_interval: int) -> None:
        """Установить интервал обновления модели мобильности (в TTI)."""
        if not isinstance(mobility_update_interval, int) or mobility_update_interval <= 0:
            raise ValueError(f"Интервал мобильности должен быть положительным int, получено: {mobility_update_interval}")
        self.sim_config.mobility_update_interval = mobility_update_interval

    def set_channel_interval(self, channel_update_interval: int) -> None:
        """Установить интервал обновления модели радиоканала (в TTI)."""
        if not isinstance(channel_update_interval, int) or channel_update_interval <= 0:
            raise ValueError(f"Интервал радиоканала должен быть положительным int, получено: {channel_update_interval}")
        self.sim_config.channel_update_interval = channel_update_interval

    def set_lte_grid(self, grid: RES_GRID_LTE) -> None:
        """
        Установить готовую ресурсную сетку LTE.

        Args:
            grid (RES_GRID_LTE): Экземпляр ресурсной сетки.

        Raises:
            TypeError: Если передан объект неверного типа.
        """
        if not isinstance(grid, RES_GRID_LTE):
            raise TypeError(
                f"Ресурсная сетка должна быть типа RES_GRID_LTE. "
                f"Получено: {type(grid).__name__}"
            )
        self.lte_grid = grid

    def set_scheduler(
        self,
        scheduler: Optional[Union[SchedulerInterface, Any, str]] = None,
        *,
        algorithm: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Установить планировщик ресурсов.

        Принимает готовый объект SchedulerInterface.
        Для обратной совместимости поддерживается передача названия алгоритма строкой
        и параметров планировщика через kwargs.

        Args:
            scheduler: Готовый экземпляр планировщика или (legacy) название алгоритма.
            algorithm: (Legacy) Название алгоритма.
            **kwargs: Параметры планировщика для обратной совместимости.
        """
        target = scheduler if scheduler is not None else algorithm

        if target is None:
            raise ValueError("Не передан планировщик или название алгоритма.")

        if hasattr(target, "schedule") and callable(target.schedule):
            self.scheduler = target
            if self.lte_grid is None and hasattr(target, "lte_grid"):
                self.lte_grid = target.lte_grid
            if hasattr(target, "algorithm"):
                self.sched_config.algorithm = str(target.algorithm)
            elif hasattr(target, "__class__"):
                self.sched_config.algorithm = target.__class__.__name__
        elif isinstance(target, str):
            self.sched_config.algorithm = target
            for key, value in kwargs.items():
                if hasattr(self.sched_config, key):
                    setattr(self.sched_config, key, value)
        else:
            raise TypeError(
                f"Планировщик должен реализовывать метод schedule(). "
                f"Получено: {type(target).__name__}"
            )

    def set_stats_manager(
        self,
        stats_manager: Optional[StatsManager] = None,
        *,
        stats_mgr: Optional[StatsManager] = None,
        **kwargs,
    ) -> None:
        """
        Установить готовый менеджер статистики.

        Принимает готовый объект StatsManager.
        Для обратной совместимости поддерживается настройка через kwargs
        параметров StatsManagerConfig.

        Args:
            stats_manager (Optional[StatsManager]): Готовый объект StatsManager.
            stats_mgr (Optional[StatsManager]): Алиас для stats_manager.
            **kwargs: Параметры StatsManagerConfig (legacy).
        """
        target = stats_manager if stats_manager is not None else stats_mgr

        if isinstance(target, StatsManager):
            self.stats_manager = target
        elif target is None and kwargs:
            unknown = set(kwargs) - self.stats_config.__dataclass_fields__.keys()
            if unknown:
                raise ValueError(f"Неизвестные параметры статистики: {sorted(unknown)}")
            candidate = replace(self.stats_config, **kwargs)
            candidate.to_statistics_config()
            self.stats_config = candidate
        elif target is None:
            self.stats_manager = None
        else:
            raise TypeError(
                f"Менеджер статистики должен быть типа StatsManager. "
                f"Получено: {type(target).__name__}"
            )

    def register_collector(
        self,
        collector: IStatsCollector,
        level: Optional[Union[MetricLevel, int, str]] = None,
    ) -> None:
        """
        Зарегистрировать пользовательский коллектор телеметрии.
        Делегирует регистрацию в StatsManager.
        """
        if self.stats_manager is not None:
            self.stats_manager.register_collector(collector, level=level)
        else:
            lvl = MetricLevel.from_value(level) if level is not None else None
            self._pending_collectors.append((collector, lvl))

    def enable_verbose_log(self, to_file: bool = False) -> None:
        """Включить подробный вывод симуляции."""
        self.sim_config.verbose = True
        self._to_file = to_file

    def setup_traffic_profiles(self, traffic_profiles: List[Dict[str, Any]]) -> None:
        """
        Инициализирует и настраивает профили трафика (bearers) для списка пользователей.

        Args:
            traffic_profiles (List[Dict[str, Any]]): Список конфигураций для пользователей.
        """
        self.sim_config.use_legacy_traffic = False
        if self.traffic_gen is None:
            try:
                from TRAFFIC_MODEL import PacketManager
                self.traffic_gen = PacketManager()
            except ImportError:
                pass

        if self.traffic_gen is not None and hasattr(self.traffic_gen, "set_model"):
            for profile in traffic_profiles:
                ue_id = profile.get("ue_id")
                if ue_id is None:
                    continue
                for bearer in profile.get("bearers", []):
                    model_type = bearer.get("model_type")
                    qci = bearer.get("qci")
                    model_params = {k: v for k, v in bearer.items() if k not in ("model_type", "qci")}
                    self.traffic_gen.set_model(ue_id=ue_id, model_type=model_type, qci=qci, **model_params)

    def setup_ue_traffic(self, ue_id: int, model_type: str, **params) -> None:
        """Настройка генерации трафика для конкретного UE."""
        if self.base_station is not None and hasattr(self.base_station, "SET_TRAFFIC_MODEL"):
            try:
                import TRAFFIC_MODEL
                model_cls = getattr(TRAFFIC_MODEL, f"{model_type}Model", None) or getattr(TRAFFIC_MODEL, model_type, None)
                if model_cls is not None:
                    model_inst = model_cls(**params)
                    ue = self.ue_collection.GET_USER(ue_id) if self.ue_collection else None
                    if ue:
                        self.base_station.SET_TRAFFIC_MODEL(ue, model_inst)
            except Exception as e:
                warnings_msg = f"[SimulationManager] Ошибка настройки трафика: {e}"
                if self.sim_config.verbose:
                    print(warnings_msg)

    # -------------------------------------------------------------------------
    # Оркестрация цикла симуляции
    # -------------------------------------------------------------------------

    def _check_required_parameters(self) -> None:
        """
        Проверка наличия всех обязательных параметров симуляции.
        SimulationManager ожидает получение компонентов в готовом виде.
        """
        errors = []
        if self.sim_config.sim_duration is None:
            errors.append("Не задана длительность симуляции. Используйте set_sim_duration(*).")
        if self.ue_collection is None:
            errors.append("Не задана коллекция пользователей. Используйте set_ue_collection(*).")
        if self.base_station is None:
            errors.append("Не задана базовая станция. Используйте set_base_station(*).")
        if self.lte_grid is None:
            errors.append("Не задана ресурсная сетка LTE. Используйте set_lte_grid(*).")
        if self.scheduler is None:
            errors.append("Не задан планировщик ресурсов. Используйте set_scheduler(*).")

        if errors:
            msg = "Ошибка: не все обязательные компоненты симуляции переданы:\n" + "\n".join(errors)
            raise RuntimeError(msg)

    def start_simulation(self) -> None:
        """
        Запуск оркестрации цикла симуляции по миллисекундам (TTI).
        Не создает lte_grid, scheduler и stats_manager внутри себя.
        """
        self._check_required_parameters()

        # Связываем базовую станцию с ресурсной сеткой
        if hasattr(self.lte_grid, "SET_BS") and callable(self.lte_grid.SET_BS):
            self.lte_grid.SET_BS(self.base_station)

        # Регистрация пользователей в базовой станции
        for ue in self.ue_collection.GET_ALL_USERS():
            if hasattr(self.base_station, "REG_UE") and callable(self.base_station.REG_UE):
                self.base_station.REG_UE(ue)

        # Сброс буферов / инициализация очередей пользователей при наличии BufferManager
        if hasattr(self.base_station, "buffer_manager") and self.base_station.buffer_manager is not None:
            bm = self.base_station.buffer_manager
            for ue in self.ue_collection.GET_ALL_USERS():
                if not bm.ue_has_buffer(ue.UE_ID):
                    if self.sim_config.use_legacy_traffic or self.traffic_gen is None:
                        bm.create_ue_buffer(ue.UE_ID, self.base_station.per_ue_max)
                    else:
                        bearers_info = self.traffic_gen.get_bearers_info(ue.UE_ID)
                        bm.create_ue_buffer(ue.UE_ID, self.base_station.per_ue_max, bearers_info)

        # Регистрация отложенных коллекторов в StatsManager, если он был передан
        if self.stats_manager is not None:
            for col, lvl in self._pending_collectors:
                self.stats_manager.register_collector(col, level=lvl)
            self._pending_collectors.clear()

        # Настройка вывода в файл при verbose
        if self._to_file:
            self._log_file = open("output.txt", "w", encoding="utf-8")
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = self._log_file
            sys.stderr = self._log_file

        pbar = None
        sbar = None
        total_duration = self.sim_config.sim_duration
        progress_stream = sys.stderr if not self._to_file else self._log_file

        STATS_REFRESH_TTI = 100
        JFI_INTERVAL_TTI = 50

        try:
            pbar = tqdm(
                total=total_duration,
                desc="Simulation Progress",
                unit="TTI",
                dynamic_ncols=True,
                colour="cyan",
                file=progress_stream,
                position=0,
                leave=True,
            )
            sbar = tqdm(
                total=0,
                desc="Stats",
                bar_format="{desc}",
                file=progress_stream,
                position=1,
                leave=True,
            )

            # -----------------------------------------------------------------
            # Основной цикл TTI (чистая оркестрация)
            # -----------------------------------------------------------------
            for tti in range(total_duration):
                GLOBALS.CURRENT_TIME = tti

                sched_result = self.run_tti(current_time=tti)

                # Передача результатов планирования в менеджер статистики
                if self.stats_manager is not None:
                    self.stats_manager.record_schedule_result(tti, sched_result)

                    if tti % self.stats_manager.config.collect_interval == 0:
                        self.stats_manager.collect(tti)

                pbar.update(1)

                # Периодическое обновление метрик справедливости
                if self.stats_manager and (tti % JFI_INTERVAL_TTI == 0):
                    ue_avg_tps = {
                        ue.UE_ID: getattr(ue, "average_throughput", 0.0)
                        for ue in self.ue_collection.GET_ALL_USERS()
                    }
                    fairness = self.stats_manager.calculate_fairness(ue_avg_tps)
                    jfi = fairness.get("dl_fairness_jain_index")
                    jfi_str = f"{jfi:.3f}" if jfi is not None else "N/A"
                else:
                    jfi_str = "N/A"

                if (tti % STATS_REFRESH_TTI == 0) and self.scheduler:
                    # TECH DEBT (Stage E1): Прямое обращение к scheduler.get_stats() и
                    # scheduler.amc.get_stats() для Progress UI — это второй канал чтения
                    # статистики в обход StatsManager. В этапе E1 UI будет получать метрики
                    # через шину коллекторов / StatsManager.
                    s = self.scheduler.get_stats() if hasattr(self.scheduler, "get_stats") else {}
                    amc_mod = getattr(self.scheduler, "amc", None)
                    a = amc_mod.get_stats() if (amc_mod and hasattr(amc_mod, "get_stats")) else {}
                    tput = a.get("dl_throughput_sum_kbps", 0)

                    sbar.set_description_str(
                        f"Stats: Active UEs: {s.get('sch_active_ue_count', 0):2d} | "
                        f"Alloc RBs: {s.get('dl_rb_allocated_count', 0):2d} | "
                        f"Throughput: {tput:8.2f} kbps | JFI: {jfi_str}"
                    )

            # -----------------------------------------------------------------
            # Финализация и экспорт телеметрии
            # -----------------------------------------------------------------
            if self.stats_manager is not None:
                self.exported_stats_files = self.stats_manager.export_all(locale="ru")
                final_tps = {
                    ue.UE_ID: getattr(ue, "average_throughput", 0.0)
                    for ue in self.ue_collection.GET_ALL_USERS()
                }
                final_fairness = self.stats_manager.calculate_fairness(final_tps)
                jfi = final_fairness.get("dl_fairness_jain_index")
                jfi_str = f"{jfi:.4f}" if jfi is not None else "N/A"
                print(f"\n[SIMULATION] Завершено успешно! Итоговый Jain's Fairness Index: {jfi_str}")
                if self.exported_stats_files:
                    print(f"[SIMULATION] Экспортированы файлы телеметрии: {self.exported_stats_files}")

        finally:
            if pbar is not None:
                pbar.close()
            if sbar is not None:
                sbar.close()
            if self._to_file and self._log_file:
                sys.stdout = self._original_stdout
                sys.stderr = self._original_stderr
                self._log_file.close()

    def run_tti(self, current_time: int) -> dict:
        """
        Выполнение одного TTI симуляции (оркестрация шага времени).

        Args:
            current_time (int): Номер текущего TTI симуляции (в миллисекундах).

        Returns:
            dict: Результат работы планировщика ресурсов (allocation, statistics, bitmap).
        """
        # 1. Обновление физики и позиций UE
        if current_time % self.sim_config.update_interval == 0:
            if hasattr(self.ue_collection, "UPDATE_ALL_USERS"):
                update_fn = self.ue_collection.UPDATE_ALL_USERS
                try:
                    sig = inspect.signature(update_fn)
                    params = sig.parameters
                    kwargs = {}
                    if "time_ms" in params:
                        kwargs["time_ms"] = current_time
                    elif "current_time" in params:
                        kwargs["current_time"] = current_time

                    if "update_interval" in params:
                        kwargs["update_interval"] = self.sim_config.update_interval
                    if "bs_position" in params and self.base_station:
                        kwargs["bs_position"] = getattr(self.base_station, "position", (0.0, 0.0))
                    if "bs_height" in params and self.base_station:
                        kwargs["bs_height"] = getattr(self.base_station, "height", 25.0)
                    if "mobility_update_interval" in params:
                        kwargs["mobility_update_interval"] = self.sim_config.mobility_update_interval
                    if "channel_update_interval" in params:
                        kwargs["channel_update_interval"] = self.sim_config.channel_update_interval

                    update_fn(**kwargs)
                except Exception:
                    try:
                        update_fn(
                            time_ms=current_time,
                            update_interval=self.sim_config.update_interval,
                            bs_position=getattr(self.base_station, "position", (0.0, 0.0)),
                            bs_height=getattr(self.base_station, "height", 25.0),
                        )
                    except Exception:
                        pass

        # 2. Обновление буферов и генерация трафика
        bs = self.base_station
        if bs is not None:
            # Режим Layered/Simple BufferManager
            if hasattr(bs, "buffer_manager") and bs.buffer_manager is not None:
                bm = bs.buffer_manager
                if hasattr(bm, "upd_buffers_all"):
                    bm.upd_buffers_all()

                if self.traffic_gen is not None:
                    for ue in self.ue_collection.GET_ALL_USERS():
                        ue_id = ue.UE_ID
                        if hasattr(self.traffic_gen, "generate_packets"):
                            pkts = self.traffic_gen.generate_packets(
                                ue_id=ue_id,
                                current_time=current_time,
                                update_interval=self.sim_config.update_interval,
                            )
                            if pkts and hasattr(bm, "add_packet"):
                                for pkt in pkts:
                                    bm.add_packet(ue_id, pkt)

            # Режим прямого буфера базовой станции
            elif hasattr(bs, "UPD_GLOBAL_BUFFER") and callable(bs.UPD_GLOBAL_BUFFER):
                bs.UPD_GLOBAL_BUFFER(current_time)

            if self.sim_config.enable_traffic_gen and hasattr(bs, "GEN_TRFFC"):
                try:
                    bs.GEN_TRFFC(current_time, self.sim_config.update_interval)
                except Exception:
                    pass

        # 3. Вызов планировщика
        users = self.ue_collection.GET_USERS_FOR_SCHEDULER()
        sched_result = self.scheduler.schedule(current_time, users)

        return sched_result if isinstance(sched_result, dict) else {}


__all__ = [
    "SimulationConfig",
    "SchedulerConfig",
    "StatsManagerConfig",
    "StatisticsConfig",
    "MetricLevel",
    "LevelsConfig",
    "SimulationManager",
    "StatsManager",
    "IStatsCollector",
    "BaseStatsCollector",
]
