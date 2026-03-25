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
from SCHEDULER import HARQManager, AdaptiveModulationAndCoding, SchedulerInterface, HARQState
import math
from collections import defaultdict
import matplotlib.pyplot as plt

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
    to_file: bool = False
    
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
    harq_enabled: bool = False

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

        self.harq_log = {
            "tti": [],
            "ue_id": [],
            "tx_count": [],
            "ack": [],
            "harq_enabled": []
        }
        
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
        self.sim_config.to_file = to_file
        
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
        if self.sim_config.to_file:
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
            
            amc_offset = defaultdict(float)

            TARGET_BLER = 0.10
            DELTA_ACK = 0.10
            DELTA_NACK = DELTA_ACK * (1.0 - TARGET_BLER) / TARGET_BLER  # для 10% => 0.9
            OFFSET_MIN, OFFSET_MAX = -5.0, 5.0

            def clamp(x, lo, hi):
                return lo if x < lo else hi if x > hi else x

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

                allocation = sched_result.get("allocation", {})

                if self.sim_config.verbose:
                    keys = list(allocation.keys())
                    print(f"[TTI={tti}] ALLOC keys sample={keys[:5]} types={[type(k).__name__ for k in keys[:5]]}")

                harq_enabled = getattr(self.sched_config, "harq_enabled", True)

                if harq_enabled:
                    # GUARD — строго ДО set_current_tti
                    if getattr(scheduler, "harq_manager", None) is None:
                        scheduler.harq_manager = HARQManager(num_processes=8, max_tx=4)

                    # ВАЖНО: 1 раз на TTI — обработка RTT очереди + таймаутов
                    scheduler.harq_manager.set_current_tti(tti)
                else:
                    # HARQ выключен — гарантируем, что UE не будет пытаться soft-combine
                    pass

                for u in users:
                    ue_id = int (u["UE_ID"])
                    rb_list = allocation.get(ue_id, [])
                    if not rb_list:
                        continue

                    ue_obj = u["ue"]
                    cqi = u["cqi"]

                    if harq_enabled:
                        # HARQ init guard (из-за UE_ID 1..N)
                        if ue_id not in scheduler.harq_manager.processes:
                            scheduler.harq_manager.init_ue(ue_id)

                        # чтобы UE мог soft-combining
                        ue_obj.harq_manager = scheduler.harq_manager
                    else:
                        # HARQ-OFF: UE не должен soft-combine
                        ue_obj.harq_manager = None

                    # 1) base_bler
                    sinr_avg = float(ue_obj.SINR)
                    base_bler = AdaptiveModulationAndCoding.lookup_bler(sinr_avg, cqi)

                    if not harq_enabled:
                        process_id = 0
                        is_retx = False

                        # HARQ-OFF: считаем AMC так же, как в HARQ-ON 
                        n_prb = len(rb_list)
                        mcs_base = scheduler.amc.cqi_to_mcs(cqi)
                        off = amc_offset.get(ue_id, 0.0)
                        mcs_eff = int(round(mcs_base + off))
                        mcs_eff = max(0, min(28, mcs_eff))
                        qm, itbs = scheduler.amc.mcs_to_qm_itbs(mcs_eff)

                        tbs_bits = GLOBALS.TB_SIZE_TABLE[itbs][n_prb]
                        tbs_bytes = int(math.ceil(tbs_bits / 8))

                        print(
                            f"[TTI={tti}] HARQ-OFF_TX UE={ue_id} "
                            f"CQI={cqi} MCS_base={mcs_base} MCS_eff={mcs_eff} off={off:+.2f} "
                            f"ITBS={itbs} Qm={qm} PRB={n_prb} TBS_bytes={tbs_bytes} "
                            f"SINR={sinr_avg:.2f} base_bler={base_bler:.4g}"
                        )

                        ack = ue_obj.receive_tb(
                            ue_id=ue_id,
                            rb_list=rb_list,
                            cqi=cqi,
                            base_bler=base_bler,
                            process_id=process_id,
                            is_retransmission=False,
                        )

                        # Логирование для графика
                        self.harq_log["tti"].append(tti)
                        self.harq_log["ue_id"].append(ue_id)
                        self.harq_log["tx_count"].append(proc.tx_count if harq_enabled else 1) # HARQ-OFF → TX всегда 1
                        self.harq_log["ack"].append(int(ack))  
                        self.harq_log["harq_enabled"].append(harq_enabled)
                        
                        print(f"[TTI={tti}] HARQ-OFF UE={ue_id} ACK={int(ack)} "
                            f"CQI={cqi} SINR={sinr_avg:.2f} base_bler={base_bler:.4g} "
                            f"PRB={len(rb_list)}")

                        # AMC offset update 
                        if ack:
                            amc_offset[ue_id] = clamp(amc_offset[ue_id] + DELTA_ACK, OFFSET_MIN, OFFSET_MAX)
                        else:
                            amc_offset[ue_id] = clamp(amc_offset[ue_id] - DELTA_NACK, OFFSET_MIN, OFFSET_MAX)

                        if self.sim_config.verbose:
                            print(f"[TTI={tti}] AMC_OFF UE={ue_id} ack={int(ack)} off={amc_offset[ue_id]:.2f}")

                        continue

                    # 2) выбрать процесс: ретрансмит или новый
                    harq_proc = scheduler.harq_manager.get_retransmission_process(ue_id)

                    if harq_proc is not None:
                        if harq_proc.tb_data is None or len(harq_proc.tb_data) == 0:
                            harq_proc.reset()
                            continue

                        # страховка
                        if harq_proc.k0 is None:
                            harq_proc.reset()
                            continue

                        process_id = harq_proc.process_id
                        is_retx = True

                        # ВАЖНО: после "отправки" ретрансмит тоже ждёт ACK
                        harq_proc.state = HARQState.WAITING_ACK
                        harq_proc.last_tx_tti = tti

                    else:
                        harq_proc = scheduler.harq_manager.get_idle_process(ue_id)
                        
                        if harq_proc is None:
                            continue

                        process_id = harq_proc.process_id
                        is_retx = False

                        #Новый TB: считаем размер TB по CQI и числу RB
                        n_prb = len(rb_list)
                        mcs_base = scheduler.amc.cqi_to_mcs(cqi)
                        off = amc_offset.get(ue_id, 0.0)
                        mcs_eff = int(round(mcs_base + off))
                        mcs_eff = max(0, min(28, mcs_eff))
                        qm, itbs = scheduler.amc.mcs_to_qm_itbs(mcs_eff)
                        mcs = mcs_eff  # дальше по коду используем уже эффективный MCS

                        tbs_bits = GLOBALS.TB_SIZE_TABLE[itbs][n_prb]
                        tbs_bytes = int(math.ceil(tbs_bits / 8))

                        tb_data = bytes(tbs_bytes)  # заглушка правильного размера

                        harq_proc.start_transmission(
                            tti=tti,
                            tb_data=tb_data,    
                            cqi=cqi,
                            rbs=rb_list,
                            soft_bits_received=None
                        )

                        # метаданные для AMC/HARQ логов 
                        harq_proc.mcs = mcs
                        harq_proc.itbs = itbs
                        harq_proc.qm = qm
                        harq_proc.n_prb = n_prb
                        harq_proc.tbs_bytes = tbs_bytes
                        harq_proc.sinr = sinr_avg
                        harq_proc.base_bler = base_bler
                        harq_proc.tbs_bits = tbs_bits

                        rv = harq_proc.get_current_rv()  # для новой передачи tx_count=1 => RV=0
                        print(f"[TTI={tti}] NEW_TB UE={ue_id} pid={process_id} "
                            f"CQI={cqi} MCS_base={mcs_base} MCS_eff={mcs_eff} offset={amc_offset.get(ue_id, 0.0):+.2f} ITBS={itbs} Qm={qm} "
                            f"PRB={n_prb} TBS_bytes={tbs_bytes} "
                            f"SINR={sinr_avg:.2f} base_bler={base_bler:.4g} RV={rv}")

                    # RX TB
                    ack = ue_obj.receive_tb(
                        ue_id=ue_id,
                        rb_list=rb_list,
                        cqi=cqi,
                        base_bler=base_bler,
                        process_id=process_id,
                        is_retransmission=is_retx,
                    )
                    
                    # Логирование HARQ для графика, для обеих веток
                    if harq_enabled:
                        proc = scheduler.harq_manager.processes[ue_id][process_id]
                        tx_count = proc.tx_count
                    else:
                        tx_count = 1

                    self.harq_log["tti"].append(tti)
                    self.harq_log["ue_id"].append(ue_id)
                    self.harq_log["tx_count"].append(tx_count)
                    self.harq_log["ack"].append(int(ack))
                    self.harq_log["harq_enabled"].append(harq_enabled)

                    # AMC offset update: только по первой передаче TB (не по ретрансмитам)
                    if not is_retx:
                        if ack:
                            amc_offset[ue_id] = clamp(amc_offset[ue_id] + DELTA_ACK, OFFSET_MIN, OFFSET_MAX)
                        else:
                            amc_offset[ue_id] = clamp(amc_offset[ue_id] - DELTA_NACK, OFFSET_MIN, OFFSET_MAX)

                        print(f"[TTI={tti}] AMC_OFF UE={ue_id} ack={int(ack)} off={amc_offset[ue_id]:.2f}")

                    # лог размера TB на ретрансмите
                    if is_retx:
                        print(f"[TTI={tti}] RETX_TB UE={ue_id} pid={process_id} "
                            f"TBS_bytes={getattr(harq_proc, 'tbs_bytes', len(harq_proc.tb_data))}")

                    # feedback (будет обработан через RTT=4 внутри HARQManager)
                    scheduler.harq_manager.handle_feedback(ue_id, process_id, ack)

                    proc = scheduler.harq_manager.processes[ue_id][process_id]

                    if proc.state == HARQState.FAIL:
                        print(f"[TTI={tti}] HARQ_FAIL UE={ue_id} pid={process_id} tx={proc.tx_count} k0={proc.k0}")

                    if (not ack) or is_retx:
                        rv = proc.get_current_rv() if hasattr(proc, "get_current_rv") else -1
                        print(
                            f"[TTI={tti}] RX UE={ue_id} pid={process_id} reTX={int(is_retx)} "
                            f"ACK={int(ack)} tx={proc.tx_count} k0={proc.k0} RV={rv} "
                            f"SINR={sinr_avg:.2f} base_bler={base_bler:.4g}")
                                        
                # Вывод статистики в CSV файл
                if self.sim_config.stats_log:
                    self._stats_logging(sched_result)

        finally:    
            # Возвращение консольного вывода
            if self.sim_config.to_file:
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
                                 "Tx_Bits", "Buffer_Size" ,"Wideband CQI", 
                                 "Subband CQI", "SINR"])
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
                subband_cqi = ue.cqi_subband
                sinr = round(ue.SINR, 4)
                
                # Запись в файл
                row = [GLOBALS.CURRENT_TIME, ue_id, num_rbs, rbs, num_cce, 
                       tx_bits, buf_size, cqi, subband_cqi, sinr]
                writer.writerow(row)

    def plot_harq_results(self, x_max=None, y_max=None, window=100):
        import matplotlib.pyplot as plt

        tti_list = self.harq_log["tti"]
        ack_list = self.harq_log["ack"]
        tx_list = self.harq_log["tx_count"]

        if not tti_list:
            print("HARQ log is empty")
            return

        if x_max is None:
            x_max = max(tti_list)

        full_tti = list(range(x_max + 1))

        # Сколько TB было в каждом TTI
        tb_count_per_tti = [0] * (x_max + 1)

        # Сколько успешных TB было в каждом TTI
        ack_per_tti = [0] * (x_max + 1)

        # Сумма tx_count по всем TB в данном TTI
        tx_sum_per_tti = [0] * (x_max + 1)

        for i, tti in enumerate(tti_list):
            if 0 <= tti <= x_max:
                tb_count_per_tti[tti] += 1
                ack_per_tti[tti] += int(ack_list[i])
                tx_sum_per_tti[tti] += tx_list[i]

        # Среднее число передач на TB в каждом TTI
        avg_tx_per_tti = []
        success_rate_per_tti = []

        for tti in range(x_max + 1):
            tb_cnt = tb_count_per_tti[tti]
            if tb_cnt > 0:
                avg_tx_per_tti.append(tx_sum_per_tti[tti] / tb_cnt)
                success_rate_per_tti.append(ack_per_tti[tti] / tb_cnt)
            else:
                avg_tx_per_tti.append(0.0)
                success_rate_per_tti.append(0.0)

        def moving_average(data, win):
            if win <= 1:
                return data[:]
            out = []
            s = 0.0
            q = []
            for x in data:
                q.append(x)
                s += x
                if len(q) > win:
                    s -= q.pop(0)
                out.append(s / len(q))
            return out

        avg_tx_smooth = moving_average(avg_tx_per_tti, window)
        success_rate_smooth = moving_average(success_rate_per_tti, window)

        if y_max is None:
            y_max = max(max(avg_tx_per_tti), max(avg_tx_smooth))
            if y_max < 1.0:
                y_max = 1.0

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Верхний график: среднее число передач
        axes[0].plot(full_tti, avg_tx_per_tti, alpha=0.25, label="Avg TX count per TTI")
        axes[0].plot(full_tti, avg_tx_smooth, linewidth=2, label=f"Moving average ({window} TTI)")
        axes[0].set_ylabel("Avg TX count")
        axes[0].set_title(f"HARQ {'ON' if self.sched_config.harq_enabled else 'OFF'}")
        axes[0].set_xlim(0, x_max)
        axes[0].set_ylim(0, y_max)
        axes[0].grid(True)
        axes[0].legend()

        # Нижний график: доля успешных TB
        axes[1].plot(full_tti, success_rate_per_tti, alpha=0.25, label="ACK rate per TTI")
        axes[1].plot(full_tti, success_rate_smooth, linewidth=2, label=f"Moving average ({window} TTI)")
        axes[1].set_xlabel("TTI")
        axes[1].set_ylabel("ACK rate")
        axes[1].set_xlim(0, x_max)
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def get_harq_plot_data(self, x_max=None, window=100):
        tti_list = self.harq_log["tti"]
        ack_list = self.harq_log["ack"]
        tx_list = self.harq_log["tx_count"]

        if not tti_list:
            return None

        if x_max is None:
            x_max = max(tti_list)

        full_tti = list(range(x_max + 1))

        tb_count_per_tti = [0] * (x_max + 1)
        ack_per_tti = [0] * (x_max + 1)
        tx_sum_per_tti = [0] * (x_max + 1)

        for i, tti in enumerate(tti_list):
            if 0 <= tti <= x_max:
                tb_count_per_tti[tti] += 1
                ack_per_tti[tti] += int(ack_list[i])
                tx_sum_per_tti[tti] += tx_list[i]

        avg_tx_per_tti = []
        success_rate_per_tti = []

        for tti in range(x_max + 1):
            tb_cnt = tb_count_per_tti[tti]
            if tb_cnt > 0:
                avg_tx_per_tti.append(tx_sum_per_tti[tti] / tb_cnt)
                success_rate_per_tti.append(ack_per_tti[tti] / tb_cnt)
            else:
                avg_tx_per_tti.append(0.0)
                success_rate_per_tti.append(0.0)

        def moving_average(data, win):
            if win <= 1:
                return data[:]
            out = []
            s = 0.0
            q = []
            for x in data:
                q.append(x)
                s += x
                if len(q) > win:
                    s -= q.pop(0)
                out.append(s / len(q))
            return out

        return {
            "tti": full_tti,
            "avg_tx_raw": avg_tx_per_tti,
            "avg_tx_smooth": moving_average(avg_tx_per_tti, window),
            "ack_rate_raw": success_rate_per_tti,
            "ack_rate_smooth": moving_average(success_rate_per_tti, window),
        }