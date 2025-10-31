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
from typing import Optional

import GLOBALS
from UE_MODULE import UECollection
from BS_MODULE import BaseStation
from RES_GRID import RES_GRID_LTE
from SCHEDULER import SchedulerInterface

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
        
        self.ue_collection = None
        self.base_station = None
        
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
            4. Основной цикл по времени (TTI):
                - Периодическое обновление состояния пользователей.
                - Генерация трафика для пользователей при наличии модели.
                - Выполнение планирования ресурсов.
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
                        update_interval=self.sim_config.update_interval, 
                        bs_position=self.base_station.position, 
                        bs_height=self.base_station.height
                    )
                    
                    # Генерация трафика для UE, если задана модель
                    for ue in self.ue_collection.GET_ALL_USERS():
                        if ue.traffic_model is not None:
                            self.base_station.GEN_TRFFC(
                                current_time=tti, 
                                update_interval=self.sim_config.update_interval,
                                ue_id=ue.UE_ID
                            )
                            
                # Подготовка данных для планировщика  
                users = self.ue_collection.GET_USERS_FOR_SCHEDULER()
                
                # Планирование ресурсов
                sched_result = scheduler.schedule(tti, users)
                
                # Вывод статистики в CSV файл
                if self.sim_config.stats_log:
                    self._stats_logging(sched_result)

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
                tx_bits = int(ue.current_dl_throughput / 1000)
                buf_size = self.base_station.ue_buffers[ue_id].sizes[ue_id] * 8
                cqi = ue.cqi
                sinr = round(ue.SINR, 4)
                
                # Запись в файл
                row = [GLOBALS.CURRENT_TIME, ue_id, num_rbs, rbs, num_cce, 
                       tx_bits, buf_size, cqi, sinr]
                writer.writerow(row)
