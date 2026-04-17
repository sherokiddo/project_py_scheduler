"""
#------------------------------------------------------------------------------
# Модуль: SCHEDULER - Планировщик ресурсов для сети LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет классы и методы для распределения ресурсных блоков между
# пользовательскими устройствами (UE) в сети LTE. Реализует алгоритмы
# планирования Round Robin, Best CQI и Proportional Fair с поддержкой PDCCH.
#
# Версия: 2.0.0
# Дата последнего изменения: 2025-10-26
# Версия Python Kernel: 3.12.9
# Автор: Брагин Кирилл, Норицин Иван
#
# Зависимости:
# - BS_MODULE.py (модуль базовой станции)
# Опосредованные зависимости
# - UE_MODULE.py (модели пользовательского оборудования)
# - RES_GRID (модель ресурсной сетки LTE)
# - BS_MODULE.py (модуль базовой станции)
#
# Изменения v1.0.5:
# - пофикшен баг с выделением ресурсов пользователям в ситуациях, когда количество
# юзеров намного больше чем количество RB в слоте. Выявлена ошибка в расчете последнего
# обслуженного UE last_served_user метода schedule
# - RES_GRID (модель ресурсной сетки LTE)
#
# Изменения v1.0.6:
# - Обновлен планировщик RR в связи с изменением принципа работы буфера
# и изменения UE_MODULE.
# - Добавлен модуль AMC
# - Добавлены тесты планировщика
# - По итогам тестов оказалось, что АМС не работает полноценно. Это печально.
# - Для дальнейшей коррекции работы, необходимо ввести фрагментацию пакетов.
#
# Изменения v1.0.7:
# - Доделан планировщик RR.Дебаг работы планировщика при работе со слотами и буфером
# - Исправлено некорректное распределение блоков по времени.
# - Расширены тесты планировщика, добавлены доп. валидации для проверки работы АМС.
# - Отложить фрагментацию пакетов. Подумать над принципом работы с буфером, корректное
# извлечение пакетов и его уменьшение. А затем перейти к формированию концепции
# транспортных блоков.
# - АМС - пофикшено, работает правильно, статистику считает правильно. Успех!
# - На будущее: тесты выпилить в отдельный блок, чтобы больше тут не жили.
#
# Изменения v1.0.8:
# - Реализован класс PDCCHManager для управления ресурсами Physical Downlink Control
#   Channel (PDCCH) с расчетом CCE (Control Channel Elements) на основе bandwidth и PCFICH.
# - Интегрирован PDCCH во все планировщики с поддержкой Aggregation Level (1/2/4/8 CCE)
# - RoundRobinScheduler: использует Sequential PDCCH Pre-Allocation подход - PDCCH
#   выделяется до цикла RBG. Минимальный waste благодаря циклическому распределению.
# - BestCQIScheduler: использует Concurrent PDCCH allocation - PDCCH выделяется
#   итеративно в момент первого RBG для каждого UE. Предотвращает waste для жадного алгоритма.
# - Добавлен новый планировщик ProportionalFairScheduler с метрикой
#   PF = instant_rate / average_throughput. Использует
#   Concurrent PDCCH подход для минимизации waste CCE.
# - Добавлены параметры конфигурации PDCCH: pcfich (1/2/3), max_dl_cce_allowance,
#   verbose_pdcch для детального логирования.
# - Все планировщики теперь возвращают 'pdcch_stats' с детальной статистикой
#   использования CCE (total/used/available/utilization).
# - Исправлена опечатка в ProportionalFair: 'PF_metric' -> 'pf_metric' для соответствия
#   Python naming conventions (PEP8).
# - Добавлен метод release_cce() в PDCCHManager как заглушка для будущей разработки
#   (освобождение CCE при неудаче PDSCH allocation).
#
# Изменения v2.0.0:
# - Полностью новая, модульная архитектура. Старые методы сохранены и
#   временно перенесены в секцию *LEGACY*.
# - Новый API. Переход на Factory pattern. Теперь масштабировать и добавлять
#   новые алгоритмы проще. Вызов стал удобнее.
# - Введен базовый абстрактный класс SchedulerInterface. Унифицированы этапы
#   планирования и приближены к реально существующим системам.
# - PDCCHManager теперь полноценная часть пайплайна планирования
# - Механизм скользящего окна добавлен (не оптимальный, но рабочий)
# - Enhanced verbose. Теперь для всего модуля verbose обязателен и становится
#   базовым инструментом отладки и тестирования.
# - Расширенная статистика по разным этапам планирования.
# - Полный рефакторинг, naming (PEP8), docstrings, типы, обработка DRY кейсов.
# - Небольшие изменения в логике работы с буфером.
# - Новый подход при расчете метрики Proportional Fair.
# - Все новые функции протестированы. Но старые сохранены.
# - Подготовлено к внедрению HARQ.
# - Составлен список планирумых фич для следующих версий.
# - Полный лист изменений и документация на текущий момент составляется.
#
# Изменения v2.1.0:
# - Старые методы *LEGACY* полностью удалены из модуля.
# - Добавлена поддержка subband_cqi получаемых через модель TDL.
# - Введен датакласс CQIMap для хранения значений sb_cqi и wb_cqi per UE.
# - Весь модуль теперь поддерживает только функции работы с CQIMap -
#   'def _get_wb_cqi' и 'def _get_sb_cqi'. Теперь это единственный источник cqi.
# - Добавлена поддержка интервального обновления cqi (CQI_reports) через
#   функцию 'def _refresh_cqi', интервал настраиваемый в 'class CQIMap'.
# - 'class SchedulingGrant' начата подготовка к переносу работы модуля на
#   SchedulingGrant.
# - Функция 'def GET_BITS_PER_RB' улучшена калькуляция расчетов для учета
#   служебного overhead по ресурсам. Для дальнейшего улучшения необходима
#   дорботка модуля AMC.
# - Очередной рефакторинг, удаление мертвого кода, небольшая оптимизация
# - Мажорные изменения - разработаны и интегрированы планировщики с частотной
#   селективностью. Всего их три: FD_PF, FD_RR, FD_BCQI.
# - Все новые функции протестированы, тесты сохранены. Результат успешный.
# - Составлен список дальнейших фич для следующих версий.
# - Полный лист изменений и документация на текущий момент составляется.
#------------------------------------------------------------------------------
"""

import GLOBALS
import time
from dataclasses import dataclass
from BS_MODULE import BaseStation
from typing import Any, Dict, List, Optional

@dataclass(slots=True)
class SchedulingGrant:
    """
    Структура, описывающая результат планирования извлечения данных из
    буферов UE.

    Attributes:
        ue_id (int): Уникальный идентификатор UE.
        num_bytes (int): Размер данных, которые необходимо извечь из буфера (байты).
        lcid (Optional[int], optional): Идентификатор логического канала.
        ndi (bool): Флаг New Data Indicator.
        harq_process_id (int): Идентификатор HARQ-процесса.
        rv (int): Redundancy Version, определяет версию кодирования при
        HARQ-ретрансляции.

    """
    ue_id: int
    num_bytes: int
    lcid: Optional[int] = None
    ndi: bool = True
    harq_process_id: int = 0
    rv: int = 0

    def __post_init__(self):
        """
        Валидация после инициализации объекта.

        Raises:
            ValueError: Если размер извлекаемых данных отрицательный или значение
            redundancy version выходит за допустимый диапазон.

        """
        if self.num_bytes < 0:
            raise ValueError(
                f"The num bytes cannot be negative. "
                f"The obtained value: {self.num_bytes}"
            )

        if not (0 <= self.rv <= 3):
            raise ValueError(
                f"The redundancy version value must be between 0 and 3. "
                f"The obtained value: {self.rv}"
            )

    def to_dict(self) -> Dict:
        """
        Преобразование объекта в словарь.

        Returns:
            Dict: Словарь с параметрами гранта:
                ue_id int - Уникальный идентификатор UE.
                num_bytes: int - Размер данных, которые необходимо извечь из буфера (байты).
                lcid: Optional[int] - Идентификатор логического канала.
                ndi: bool - Флаг New Data Indicator.
                harq_process_id: int - Идентификатор HARQ-процесса.
                rv: int - Redundancy Version, определяет версию кодирования при
                HARQ-ретрансляции.

        """
        return {
            'ue_id': self.ue_id,
            'num_bytes': self.num_bytes,
            'lcid': self.lcid,
            'harq_process_id': self.harq_process_id,
            'ndi': self.ndi,
            'rv': self.rv
        }

#TODO: Интегрировать SchedulingGrant в пайплайн работы модуля

#==============================================================================
#                              РАЗДЕЛ DATACLASS
#==============================================================================

@dataclass(slots=True)
class CQIMap:
    """
    Датакласс со всей необходимой информацией CQI по каждому UE (ключ).
    На замену словарям и постоянным вызовам UE_MODULE.py
    Также добавляет элемент оптимизации по памяти и периодичность обновления.

    Предполагаемый жизненный цикл:
    - Created: Первый CQI Report - регистрация на BS
    - Updated: Периодический апдейт (по гибкому таймеру, апериодичный - позже)
    - Deleted: UE дерегистрация (истечение таймера - позже)
    """
    wb_cqi: int              # CQI значения [1-15]
    last_wb_update: int      # TTI посл. обновления (для счетчика)
    #wb_timer: int            # TTI таймер обновления

    sb_cqi: List[int]        # Per-RBG CQI [length = num_rbg]
    last_sb_update: int
    #sb_timer: int

    #TODO: вынести ручки для настройки интервалов в симуляцию

#==============================================================================
#                              ИНТЕРФЕЙС МОДУЛЯ
#==============================================================================

class SchedulerInterface:
    """
    Интерфейс для планировщиков ресурсов.
    Использует factory метод "create"
    для выбора конкретных планировщиков без лишнего импорта

    Args:
        lte_grid: Ресурсная сетка LTE
        bs: Базовая станция
        max_dl_ue_tti: Максимум UE за TTI (None = без ограничений)
        pcfich: PCFICH value
        max_dl_cce_allowance: Лимит CCE
        verbose_pdcch: Отладочный вывод для PDCCH
        enable_window: Включить скользящее окно
        window_size: Размер окна в TTI
        verbose: Отладочный вывод для всех этапов планирования (default: False)
    """

    @staticmethod
    def create(algorithm: str, lte_grid, bs, **kwargs):
        """
        Factory method для создания планировщиков по имени алгоритма.

        Позволяет создавать планировщики без знания конкретных классов.
        Удобно для конфигурирования через параметры или файлы.

        Args:
            algorithm: Имя алгоритма ('BestCQI', 'ProportionalFair', 'RoundRobin')
            lte_grid: Ресурсная сетка LTE
            bs: Базовая станция
            **kwargs: Параметры планировщика (max_dl_ue_tti, verbose, etc.)

        Returns:
            SchedulerInterface: Экземпляр конкретного планировщика

        Raises:
            ValueError: Если алгоритм неизвестен
        """
        # Реестр планировщиков (избегаем циклических зависимостей)
        schedulers = {
            'BestCQI':          BestCQIScheduler,
            'ProportionalFair': ProportionalFairScheduler,
            'RoundRobin':       RoundRobinScheduler,
            'FD_BCQI':          FDxBestCQIScheduler,
            'FD_FGS':           FDxFairGreedyScheduler,
            'FD_PF':            FDxProportionalFairScheduler,
            }

        if algorithm == 'DqnScheduler':
            from drl.dqn_scheduler import DqnScheduler

            schedulers['DqnScheduler'] = DqnScheduler

        if algorithm == 'PpoScheduler':
            from drl.ppo_scheduler import PpoScheduler

            schedulers['PpoScheduler'] = PpoScheduler

        if algorithm == 'PpoRankerScheduler':
            from drl.ppo_ranker_scheduler import PpoRankerScheduler

            schedulers['PpoRankerScheduler'] = PpoRankerScheduler

        if algorithm not in schedulers:
            valid = ', '.join(schedulers.keys())
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Valid algorithms: {valid}")

        scheduler_class = schedulers[algorithm]

        return scheduler_class(lte_grid, bs, **kwargs)

    @staticmethod
    def available_algorithms():
        """
        Получить список доступных алгоритмов приоритизации.

        Returns:
            List[str]: Список имен алгоритмов
        """
        return ['BestCQI',
                'ProportionalFair',
                'RoundRobin',
                'FD_BestCQI',
                'FD_FGS',
                'FD_FF',
                'DqnScheduler',
                'PpoScheduler',
                'PpoRankerScheduler',
                ]

    def __init__(self, lte_grid, bs,
                 max_dl_ue_tti=None,
                 pcfich=2,
                 max_dl_cce_allowance=None,
                 verbose_pdcch=False,
                 window_size=100,
                 enable_window=True,
                 verbose = False,
                 simulation_context: Optional[Dict[str, Any]] = None):

        self.lte_grid = lte_grid
        self.lte_grid.SET_BS(bs)
        self.amc = AdaptiveModulationAndCoding(self)
        self.pdcch_manager = PDCCHManager(bandwidth=lte_grid.bandwidth,
            pcfich=pcfich,
            max_dl_cce_allowance=max_dl_cce_allowance,
            verbose=verbose_pdcch)

        self.max_dl_ue_tti = max_dl_ue_tti
        self.enable_window = enable_window
        self.active_ue_window = []
        self.window_size = window_size
        self.simulation_context = dict(simulation_context or {})

        self._last_eligible_ue_count = 0
        self._last_allocated_rb_count = 0
        self._last_active_ue_count = 0
        self._last_tti = -1
        self._last_allocation = None
        self._last_users = None
        self._last_priority_list = []
        self._last_prioritized_users = []
        self._last_windowed_users = []

        self._last_sch_time_us = 0.0
        self._last_priority_calc_time_us = 0.0
        self._last_priority_sort_time_us = 0.0
        self._last_priority_list_size = 0
        self._last_priority_list_full = []
        self._last_avg_priority_value = 0.0
        self._last_pdcch_blocked_count = 0

        self.last_ue_transmitted_bits = {}

        self.cqi_map: Dict[int, CQIMap] = {}
        self.wb_cqi_upd_interval = 1
        self.sb_cqi_upd_interval = 1

        self.verbose = verbose

        self.harq_manager = None  # Заготовка для HARQ

        if self.verbose:
            print(f"[SCHEDULER] Initialized {self.__class__.__name__}")

    def schedule(self, tti: int, users: List[Dict]) -> Dict:
        """
        Основной оркестратор этапов планирования ресурсов
        Этапы:
        1. Фильтрация UE
        2. Active window update - обновление скользящего окна
        3. Priority calculation - расчет приоритетов
        4. PList formation - выбор Priority UE
        5. PDCCH allocation - выделение управляющих ресурсов
        6. PDSCH allocation - распределение RBG
        7. Buffer processing - обработка буферов
        8. Result formation - формирование результата и статс

        3, 6 реализуются в конкретных планировщиках
        """
        t_sch_start = time.perf_counter()
        self._last_tti = tti

        # ЭТАП 0: Preparation and CQI Map Check
        if tti % self.wb_cqi_upd_interval == 0:
            self._refresh_cqi(tti, users)

        # ЭТАП 1: Eligibility checks
        eligible_ues = self._filter_eligible_ues(tti, users)
        if not eligible_ues:
            return self._empty_result()

        # ЭТАП 2: Active window update
        self._update_active_window(eligible_ues)

        # ЭТАП 2.5: Window filtering
        windowed_ues = self.filter_by_window(eligible_ues)
        self._last_windowed_users = list(windowed_ues)
        if not windowed_ues:
            return self._empty_result()

        # ЭТАП 3: Priority calculation
        t_priority_start = time.perf_counter()

        prioritized_ues = self._calculate_priorities(windowed_ues, tti)
        self._last_prioritized_users = prioritized_ues
        t_priority_end = time.perf_counter()
        priority_calc_time_us = (t_priority_end - t_priority_start) * 1_000_000

        # ЭТАП 4: PList formation
        t_sort_start = time.perf_counter()

        priority_list = self._form_priority_list(prioritized_ues, tti)

        t_sort_end = time.perf_counter()
        priority_sort_time_us = (t_sort_end - t_sort_start) * 1_000_000

        self._last_priority_list_size = len(priority_list)

        if priority_list:
            self._last_avg_priority_value = sum(u.get('priority', 0) for u in priority_list) / len(priority_list)
        else:
            return self._empty_result()

        # Этап 4.5: PDSCH estimation
        priority_list_filtered = self._apply_pdsch_estimation(priority_list, tti)
        self._last_priority_list = priority_list_filtered
        self._last_priority_list_full = priority_list

        # ЭТАП 5: PDCCH allocation
        ues_with_pdcch = self._allocate_pdcch(priority_list_filtered)
        self._last_pdcch_blocked_count = len(priority_list_filtered) - len(ues_with_pdcch)
        if not ues_with_pdcch:
            return self._empty_result()

        # ЭТАП 6: PDSCH allocation
        # TODO: Оптимизировать allocator: кэшировать RBG->RB indices/width и CQI->bits_per_rb,
        # чтобы не пересчитывать FD/PF метрику и служебные lookup в каждом шаге по RBG.
        allocation = self._allocate_pdsch(tti, ues_with_pdcch, eligible_ues)

        # ЭТАП 7: Buffer processing
        self._process_buffers(tti, users, allocation)

        # ЭТАП 7.5: Stats processing
        allocated_rbs = sum(len(rbs) for rbs in allocation.values())
        active_ues = sum(1 for rbs in allocation.values() if len(rbs) > 0)
        self._update_stats(tti, len(eligible_ues), allocated_rbs, active_ues)
        self._last_allocation = allocation.copy()
        self._last_users = users

        t_sch_end = time.perf_counter()
        sch_time_us = (t_sch_end - t_sch_start) * 1_000_000
        self._save_timing_stats(sch_time_us,
                                priority_calc_time_us,
                                priority_sort_time_us,
                                )

        # ЭТАП 8: Result formation
        return self._build_result(allocation, users, eligible_ues, tti)

    def get_stats(self) -> Dict:
        """
        Получить статистику планировщика за последний TTI.

        Вызывается StatsManager'ом для сбора статистики.
        Возвращает счетчики, обновленные в schedule().

        Returns:
            Dict: {
                "tti": int,                      # Последний обработанный TTI
                "eligible_ue_count": int,        # Кол-во UE в очереди
                "allocated_rb_count": int,       # Всего выделено RB
                "active_ue_count": int,          # Кол-во UE с allocation > 0
                "allocation_efficiency": float   # Среднее RB на одного активного UE
            }
        """
        if self._last_active_ue_count == 0:
            rb_per_ue_avg = 0.0
        else:
            rb_per_ue_avg = self._last_allocated_rb_count / self._last_active_ue_count

        total_rbs = self.lte_grid.rb_per_slot
        prb_utilization_pct = (
            (self._last_allocated_rb_count / total_rbs * 100)
            if total_rbs > 0 else 0.0)

        total_buffer_bytes = 0
        ue_buffer_sizes = {}
        if hasattr(self, '_last_prioritized_users') and self._last_prioritized_users:
            for user in self._last_prioritized_users:
                ue_id = user.get('UE_ID')
                # bs_buffer_size добавляется в _filter_eligible_ues()
                buffer_size = user.get('bs_buffer_size', 0)
                total_buffer_bytes += buffer_size
                if ue_id is not None:
                    ue_buffer_sizes[ue_id] = buffer_size

        stats = {
            'tti': self._last_tti,
            'sch_eligible_ue_count': self._last_eligible_ue_count,
            'sch_active_ue_count': self._last_active_ue_count,
            'dl_rb_allocated_count': self._last_allocated_rb_count,
            'dl_rb_per_ue_avg': round(rb_per_ue_avg, 2),
            'buffer_size_sum_bytes': total_buffer_bytes,
            #TODO: временное решение. избавиться как сделаем buffer.get_stats()
            'dl_prb_utilization_pct': round(prb_utilization_pct, 2),
            'sch_total_time_us': round(self._last_sch_time_us, 2),
            'sch_priority_list': self._last_priority_list_full,
            'sch_priority_calc_time_us': round(self._last_priority_calc_time_us, 2),
            'sch_priority_sort_time_us': round(self._last_priority_sort_time_us, 2),
            'sch_priority_list_size': self._last_priority_list_size,
            'sch_window_ue_count': len(self._last_windowed_users),
            'sch_pdcch_blocked_count': self._last_pdcch_blocked_count,
            'sch_avg_priority_value': round(self._last_avg_priority_value, 4),
            "ue_buffer_sizes": ue_buffer_sizes,
            "ue_transmitted_bits": self.last_ue_transmitted_bits.copy()}

        if hasattr(self, '_last_priority_list') and self._last_priority_list:
            stats["ue_priorities"] = {
                u["UE_ID"]: u["priority"]
                for u in self._last_priority_list
            }
        else:
            stats["ue_priorities"] = {}

        return stats

    def _refresh_cqi(self, tti: int, users: List[Dict]) -> None:
        """
        Обновление CQI map.
        Обновляет wideband CQI всегда, subband CQI только если enable_fd=True.
        Для новых UE создаёт CQIMap entry, для существующих - mutation.
        Вызывается в начале schedule() для синхронизации CQI data.

        Args:
            users: Список всех UE (не filtered!)
            tti: Текущий TTI

        Примечание:
            - WB CQI: берётся из user['cqi'] (индекс 1-15)
            - SB CQI: берётся из user['ue'].cqi_subband (List[int])
            - Invalid CQI (вне [1-15]) игнорируются
        """
        for user in users:
            ueid = user.get('UE_ID')
            if ueid is None:
                continue

            cqi = user.get('cqi', 0)

            if 1 <= cqi <= 15:
                if ueid not in self.cqi_map:
                    self.cqi_map[ueid] = CQIMap(
                        wb_cqi=cqi,
                        last_wb_update=tti,
                        sb_cqi=[],
                        last_sb_update=0
                    )
                    if self.verbose:
                        print(f"[CQI_MAP TTI {tti}] UE {ueid}: Created entry (WB CQI={cqi})")
                else:
                    self.cqi_map[ueid].wb_cqi = cqi
                    self.cqi_map[ueid].last_wb_update = tti
                    # if self.verbose:
                    #     print(f"[CQI_MAP TTI {tti}] UE {ueid}: Updated WB CQI={cqi}")

            # === Subband CQI update (только если FD enabled) ===
            # TODO: Заменить на реальную проверку FD flag когда будет реализовано
            # Сейчас проверяем наличие ue.cqi_subband как индикатор FD
            sb_cqi_list = user.get('sbb_cqi')

            if sb_cqi_list and isinstance(sb_cqi_list, list) and len(sb_cqi_list) > 0:
                if ueid in self.cqi_map:
                    self.cqi_map[ueid].sb_cqi = sb_cqi_list.copy()
                    self.cqi_map[ueid].last_sb_update = tti
                    # if self.verbose:
                    #     print(f"[CQI_MAP TTI {tti}] UE {ueid}: Updated SB CQI "
                    #           f"({len(sb_cqi_list)} RBG)")

        if self.verbose and self.cqi_map:
            print(f"[CQI_MAP TTI {tti}] Active entries: {len(self.cqi_map)} UE")

    def _filter_eligible_ues(self, tti: int, users: List[Dict]) -> List[Dict]:
        """
        Фильтрация доступных (допустимых) юзеров.
        Подготовительный этап планирования

        Текущие проверки:
        1. BS buffer существует
        2. Buffer size > 0 (есть данные)
        3. Valid CQI (1-15)
        """
        # Подготовка: Сброс throughput
        for user in users:
            user['ue'].current_dl_throughput = 0

        eligible = []
        buffer_manager = self.lte_grid.bs.buffer_manager

        for user in users:
            ue_id = user['UE_ID']

            # CHECK 1: BS buffer существует?
            if not buffer_manager.ue_has_buffer(ue_id):
                continue # Пропускаем UE без буфера

            # CHECK 2: Получение Buffer status
            buffer_status_list = buffer_manager.get_buffer_status(ue_id)

            buffer_size = 0
            for buffer_status in buffer_status_list:
                buffer_size += buffer_status.buffer_size
            user_cqi = self._get_wb_cqi(ue_id)

            # CHECK 3: Buffer size > 0 и Valid CQI (1-15)
            if buffer_size > 0 and 1 <= user_cqi <= 15:
                user['bs_buffer_size'] = buffer_size
                eligible.append(user)

            # TODO: Проверка HARQ процессов
            # if harq_manager.all_processes_busy(ue_id):  # Все процессы заняты
            #   continue  # В режиме ретрансляции HARQ не планируем

            # TODO: Проверка DRX state
            # if user['ue'].drx_state == 'SLEEP':  # UE в режиме сна
            #   continue  # В режиме сна не планируем

        # verbose
        if self.verbose:
            print(f"[SCHEDULER TTI {tti}] Eligibility: {len(users)} total -> {len(eligible)} eligible")

        return eligible

    def _update_active_window(self, eligible_ues: List[Dict]) -> None:
        """
        Обновление скользящего окна активных UE.
        Окно размером window_size TTI используется для отслеживания активности UE.
        Скользящее окно очень пригодится, когда количество UE в симуляции намного
        больше, чем десятки. А также для QoS. Ограничивает использование CPU
        Методов скольящего окна существует множество и это проприетарное решение.
        #TODO: Сейчас реализована FIFO-Queue. Можно (и полезно для оптимизации):
            - циклический метод (token ring/cyclic);
            - временной метод (time-based window);
            - экспоненциальный метод (exponential decay);
            - разряженный метод (sparse);
            - и др.

        Args:
            eligible_ues: Список eligible UE
        """
        if not self.enable_window:
            self.active_ue_window = eligible_ues
            return

        current_ue_ids = {ue['UE_ID'] for ue in eligible_ues}

        if not hasattr(self, '_ue_queue') or not self._ue_queue:
            self._ue_queue = list(current_ue_ids)
            if self.verbose:
                print(f"SCHEDULER [Window] Initialized queue with {len(self._ue_queue)} UE")
            return

        existing_ue_set = set(self._ue_queue)
        new_ues = current_ue_ids - existing_ue_set
        removed_ues = existing_ue_set - current_ue_ids

        self._ue_queue.extend(new_ues)

        if removed_ues:
            self._ue_queue = [ue_id for ue_id in self._ue_queue if ue_id not in removed_ues]

        # verbose
        if self.verbose and (new_ues or removed_ues):
            print(f"SCHEDULER [Window] Queue updated: +{len(new_ues)} new, -{len(removed_ues)} removed, "
                  f"total={len(self._ue_queue)} UE")

    def filter_by_window(self, eligible_ues: List[Dict]) -> List[Dict]:
        """
        Фильтрация eligible UE по активному окну.
        Только UE, которые присутствуют в active_ue_window за последние
        windowsize TTI, могут быть рассмотрены для планирования.

        Args:
            eligible_ues: Все eligible UE

        Returns:
            List[Dict]: UE, которые есть в окне
        """
        if not self.enable_window or not hasattr(self, '_ue_queue') or not self._ue_queue:
            return eligible_ues

        window_ue_ids = self._ue_queue[:self.window_size]
        current_ue_ids = {ue['UE_ID']: ue for ue in eligible_ues}
        windowed_ues = []
        for ue_id in window_ue_ids:
            if ue_id in current_ue_ids:
                windowed_ues.append(current_ue_ids[ue_id])
        processed_count = min(self.window_size, len(self._ue_queue))
        self._ue_queue = self._ue_queue[processed_count:] + self._ue_queue[:processed_count]

        if self.verbose:
            print(f"SCHEDULER [Window] TTI window: {len(windowed_ues)}/{len(eligible_ues)} UE selected, "
                  f"order: {[ue['UE_ID'] for ue in windowed_ues[:3]]}..., "
                  f"rotated {processed_count} UE to end of queue")

        return windowed_ues

    def _form_priority_list(self, prioritized_ues: List[Dict], tti: int) -> List[Dict]:
        """
        Сортировка по priority (descending) и выбор top-N UE.

        Args:
            prioritized_ues: UE с рассчитанными приоритетами
            tti: Текущий TTI

        Returns:
            List[Dict]: Top-N UE для планирования
        """
        sorted_ues = sorted(prioritized_ues,
                           key=lambda u: u['priority'],
                           reverse=True)

        if self.max_dl_ue_tti:
            priority_list = sorted_ues[:self.max_dl_ue_tti]
        else:
            priority_list = sorted_ues

        if self.verbose:
            print(f"[SCHEDULER TTI {tti}] PriorityList: Selected {len(priority_list)} UE")
            if priority_list:
                top3 = priority_list[:3]
                print(f"[SCHEDULER TTI {tti}] Top-3 UE: {[(u['UE_ID'], u['priority']) for u in top3]}")

        return priority_list

    def _apply_pdsch_estimation(self, priority_list: List[Dict], tti: int) -> List[Dict]:
        """
        Concurrent PDSCH estimation
        Фильтрует priority_list учитывая PDSCH capacity constraints:
        - Оценивает RB потребность для каждого UE (на основе wideband CQI)
        - Прекращает отбор когда estimated PDSCH достигает 95% от доступных RB
        - Предотвращает CCE overhead (не выделяем PDCCH тем кто не получит PDSCH)

        Args:
            priority_list: Sorted UE list (from _form_priority_list)
            tti: Текущий TTI

        Returns:
            List[Dict]: Filtered UE list для PDCCH allocation (после estimation)
        """
        total_rb     = self.lte_grid.rb_per_slot
        estimated_rb = 0.0
        selected_ues = []

        # Threshold для early stopping (95% от total RB)
        # Оставляем 5% запас для overhead и динамики канала
        pdsch_threshold = total_rb * 0.95

        for idx, user in enumerate(priority_list):
            ue_id       = user['UE_ID']
            cqi         = self._get_wb_cqi(ue_id)
            bits_per_rb = self.amc.GET_BITS_PER_RB(cqi)
            buffer_bits = user['bs_buffer_size'] * 8
            rb_needed   = min(buffer_bits // bits_per_rb, total_rb)

            if idx == 0:
                selected_ues.append(user)
                estimated_rb += rb_needed
                continue

            if estimated_rb + rb_needed >= pdsch_threshold:
                if self.verbose:
                    remaining = len(priority_list) - len(selected_ues)
                    print(f"[SCHEDULER TTI {tti}] PDSCH estimation threshold reached "
                          f"({estimated_rb:.0f}/{total_rb} RB = {estimated_rb/total_rb*100:.1f}%), "
                          f"{remaining} UE excluded")
                break

            # Добавляем UE
            selected_ues.append(user)
            estimated_rb += rb_needed

        # Verbose
        if self.verbose and selected_ues:
            excluded = len(priority_list) - len(selected_ues)
            print(f"[SCHEDULER TTI {tti}] After estimation: {len(selected_ues)} UE selected, "
                  f"{excluded} UE excluded")
            print(f"[SCHEDULER TTI {tti}] Estimated PDSCH: "
                  f"{estimated_rb:.0f}/{total_rb} RB ({estimated_rb/total_rb*100:.1f}%)")

        return selected_ues

    def _allocate_pdcch(self, priority_list: List[Dict]) -> List[Dict]:
        """
        PDCCH allocation.
        Выделение PDCCH (CCE) для UE из Priority_List.
        Используется для BestCQI и ProportionalFair планировщиков
        (concurrent PDCCH/PDSCH allocation).

        RoundRobin переопределяет этот метод как pass-through,
        т.к. выделяет PDCCH внутри _allocate_pdsch() (sequential allocation).

        Args:
            priority_list: Список UE для планирования

        Returns:
            List[Dict]: UE с успешно выделенным PDCCH
        """
        self.pdcch_manager.reset_tti()

        ues_with_pdcch = []

        for user in priority_list:
            ue_id   = user['UE_ID']
            cqi     = self._get_wb_cqi(ue_id)

            required_cce = self.pdcch_manager.get_aggregation_level(cqi)

            if self.pdcch_manager.allocate_cce(ue_id, required_cce):
                user['allocated_cce'] = required_cce
                ues_with_pdcch.append(user)

        # verbose
        if self.verbose:
            blocked_count = len(priority_list) - len(ues_with_pdcch)
            print(f"[SCHEDULER] PDCCH allocation: {len(priority_list)} requested -> {len(ues_with_pdcch)} allocated")

            if blocked_count > 0:
                allocated_ids = {u['UE_ID'] for u in ues_with_pdcch}
                blocked_ue_ids = [u['UE_ID'] for u in priority_list if u['UE_ID'] not in allocated_ids]
                print(f"[SCHEDULER] PDCCH blocked: {blocked_count} UE (no CCE available) - UE IDs: {blocked_ue_ids}")

        return ues_with_pdcch

    def _logical_channel_multiplexing(self, tb_size: int, buffer_status_list: List) -> List[SchedulingGrant]:
        """
        Мультиплексирование логических каналов  в пределах одного транспортного 
        блока. В режиме Simple Buffer весь размер транспортного блока выделяется 
        единственному буферу UE. В режиме Layered Buffer предполагается 
        распределение TB между несколькими логическими каналами.
        
        Args:
            tb_size (int): Размер транспортного блока (байты).
            buffer_status_list (List): Список состояний буферов UE.

        Raises:
            ValueError: Если количество полученных BufferStatus'ов в режиме
                Simple Buffer не равно 1, список полученных BufferStatus'ов в режиме
                Layered Buffer пуст или значения UE ID у полученных BufferStatus'ов 
                в режиме Layered Buffer отличаются.

        Returns:
            List[SchedulingGrant]: Список грантов, определяющих количество байт,
                для каждого логического канала UE.

        """
        # Simple buffer mode
        if self.lte_grid.bs.use_simple_buffer:
            if len(buffer_status_list) != 1:
                raise ValueError(
                    "The size of the buffer status list for Simple Buffer "
                    "must be 1"
                )

            buffer_status = buffer_status_list[0]
            grant = SchedulingGrant(
                ue_id=buffer_status.ue_id,
                num_bytes=tb_size,
            )

            return [grant]

        # Layered buffer mode
        else:
            if not buffer_status_list:
                raise ValueError(
                    "The buffer status list must not be empty."
                )
        
            if not all(status.ue_id == buffer_status_list[0].ue_id for status in buffer_status_list):
                raise ValueError(
                    "The UE ID in all buffer statuses must be the same"
                )
            
            # Просто делим транспортный блок на равные части между всеми активными LC.
            # Несправедливая стратегия, т.к. LC с малым количеством данных получает такой же
            # объём транспортного блока, что и LC с большим количеством данных.
            active_lcs = sum(1 for status in buffer_status_list if status.buffer_size > 0)
            num_bytes_per_lc = tb_size // active_lcs

            grants = []
            for buffer_status in buffer_status_list:
                if buffer_status.buffer_size > 0:
                    grant = SchedulingGrant(
                        ue_id=buffer_status.ue_id,
                        num_bytes=num_bytes_per_lc,
                        lcid=buffer_status.lcid,
                    )

                    grants.append(grant)

            return grants
                
    def _process_buffers(self, tti: int, users: List[Dict], allocation: Dict) -> None:
        """
        Обработка буферов и обновление throughput статистики.
        Извлечение пакетов из BS буферов для всех UE на основе allocation.

        Для каждого UE:
        1. Рассчитывает сколько RB было выделено
        2. Определяет максимальное количество байт для передачи (на основе CQI и RB)
        3. Извлекает пакеты из BS буфера
        4. Обновляет DL throughput статистику UE

        Args:
            tti: Текущий TTI
            users: Список всех UE (не только запланированных)
            allocation: Allocation map (UE_ID -> List[freq_idx])
        """
        total_bits_transmitted = 0
        ues_transmitted = 0
        time_interval_ms = 1
        self.last_ue_transmitted_bits = {}
        buffer_manager = self.lte_grid.bs.buffer_manager

        for user in users:
            ue   = user.get('ue')
            ueid = user.get('UE_ID')

            if ue is None or ueid is None:
                continue

            allocated_rbs = len(allocation.get(ueid, []))

            if allocated_rbs == 0:
                ue.UPD_DL_THROUGHPUT_BPS(0, time_interval_ms)
                self.last_ue_transmitted_bits[ueid] = 0

            if not buffer_manager.ue_has_buffer(ueid):
                self.last_ue_transmitted_bits[ueid] = 0
            # UE нет в буферах BS — обновить на 0
                ue.UPD_DL_THROUGHPUT_BPS(0, time_interval_ms)
                continue

            cqi = self._get_wb_cqi(ueid)
            bits_per_rb = self.amc.GET_BITS_PER_RB(cqi)

            # @IvanNoritsin: Тут по хорошему должен расчитываться размер транспортного
            # блока (на основе allocated_rbs и MCS), но пока что у нас этого нет
            max_bits  = allocated_rbs * bits_per_rb
            max_bytes = max_bits // GLOBALS.BITS_PER_BYTE
            #ВНИМАНИЕ! Временный костыль.
            remainder_bits = max_bits % GLOBALS.BITS_PER_BYTE

            if max_bytes <= 0:
                ue.UPD_DL_THROUGHPUT_BPS(0, time_interval_ms)
                self.last_ue_transmitted_bits[ueid] = 0
                continue

            buffer_status_list = buffer_manager.get_buffer_status(ueid)

            # @IvanNoritsin: Мультиплексер принимает на вход размер транспорного
            # блока, но т.к. у нас нет этой системы просто пердаём вместимость
            # выделенных ресурсных блоков
            grants = self._logical_channel_multiplexing(max_bytes, buffer_status_list)

            try:
                packets, total_bytes = buffer_manager.get_packets(grants)

            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] TTI {tti} UE {ueid}: Buffer extraction failed - {e}")
                ue.UPD_DL_THROUGHPUT_BPS(0, time_interval_ms)
                self.last_ue_transmitted_bits[ueid] = 0
                continue

            transmitted_bits = GLOBALS.bytes_to_bits(total_bytes) + remainder_bits
            self.last_ue_transmitted_bits[ueid] = transmitted_bits
            ue.UPD_DL_THROUGHPUT_BPS(transmitted_bits, time_interval_ms)

            total_bits_transmitted += transmitted_bits
            if total_bytes > 0:
                ues_transmitted += 1

            if self.verbose:
                throughput_kbps = (transmitted_bits * 1000) / (time_interval_ms * 1000)  # Кбит/с
                print(f"[BUFFER] TTI {tti} UE {ueid}: "
                      f"RB={allocated_rbs}, CQI={cqi}, bits/RB={bits_per_rb}, "
                      f"MaxBytes={max_bytes}, Transmitted={total_bytes}B ({transmitted_bits}bits, {throughput_kbps:.1f}Kbps)")

        if self.verbose and (total_bits_transmitted > 0 or ues_transmitted > 0):
            print(f"[SCHEDULER] TTI {tti}: Buffer processing - "
                  f"Total {total_bits_transmitted} bits to {ues_transmitted} UE")


    def _build_result(self, allocation: Dict, users: List[Dict],
                     eligible_ues: List[Dict], tti: int) -> Dict:
        """
        Формирование финального результата планирования.

        Результат содержит:
        - allocation: Карта распределения ресурсов (UE_ID -> List[freq_idx])
        - statistics: Throughput статистика для всех UE
        - bitmap: Визуализация ресурсной сетки для eligible UE
        - pdcch_stats: Статистика использования PDCCH

        Args:
            allocation: Allocation map (UE_ID -> List[freq_idx])
            users: Список всех UE
            eligible_ues: Список eligible UE (для bitmap generation)
            tti: Текущий TTI

        Returns:
            Dict с ключами: allocation, statistics, bitmap, pdcch_stats
        """
        bitmap = {
            user['UE_ID']: self.lte_grid.GENERATE_BITMAP(tti, user['UE_ID'])
            for user in eligible_ues}

        pdcch_stats = self.pdcch_manager.get_stats()

        # Verbose
        if self.verbose:
            num_scheduled = sum(1 for rbs in allocation.values() if len(rbs) > 0)
            total_rb_allocated = sum(len(rbs) for rbs in allocation.values())
            print(f"[SCHEDULER TTI {tti}] Result: {num_scheduled} UE scheduled, {total_rb_allocated} RB allocated")

        return {
            'allocation': allocation,
            'statistics': {},
            'bitmap': bitmap,
            'pdcch_stats': pdcch_stats}

    #TODO: метод будет преобразован или полностью убран,
    # когда в планировщиках останется только формирование bitmap

    def _empty_result(self) -> Dict:
        """
        Вдруг пустой результат?

        Используется когда нет UE для планирования:
        - Нет eligible UE (все буферы пустые или неверный CQI)
        - Пустой priority_list
        - Нет UE с выделенным PDCCH (все заблокированы)

        Returns:
            Dict с пустыми результатами и PDCCH статистикой
        """
        if self.verbose:
            print("[SCHEDULER] Empty result: No UE to schedule")
        #TODO: можно добавить *reason чтобы возвращал в консоль причину

        self._last_priority_list_full = []
        self._update_stats(self._last_tti, 0, 0, 0)
        self._last_allocation = {}
        self._last_users = []
        self._last_windowed_users = []

        return {
            'allocation': {},
            'statistics': {},
            'bitmap': {},
            'pdcch_stats': self.pdcch_manager.get_stats()}

    def _get_wb_cqi(self, ueid: int) -> int:
        """
        Получить wideband CQI из map.
        Args:
            ueid: UE ID
        Returns:
            int: CQI value [1-15] или 0 если UE не в map
        Примечание:
            Возврат 0 означает:
            - UE ещё не отправил первый CQI report, ИЛИ
            - UE отключился но ещё в users list
            Планировщик автоматически отфильтрует UE с CQI=0
        """
        if ueid in self.cqi_map:
            return self.cqi_map[ueid].wb_cqi
        else:
            # if self.verbose:
            #     print(f"[CQI_MAP] WARNING: UE {ueid} not in map, returning CQI=0")
            return 0

    def _get_sb_cqi(self, ueid: int) -> List[int]:
        """
        Получить subband CQI из map.
        Args:
            ueid: UE ID
        Returns:
            List[int]: Subband CQI per RBG или пустой список если:
            - UE не в map
            - UE не использует FD (sb_cqi пустой)
        Примечание:
            Пустой список - нормальное состояние для non-FD.
            Caller должен проверить len() перед использованием.
        """
        if ueid in self.cqi_map:
            return self.cqi_map[ueid].sb_cqi
        else:
            if self.verbose:
                print(f"[CQI_MAP] WARNING: UE {ueid} not in map, returning empty SB CQI")
            return []

    def _update_stats(self, tti: int, eligible_count: int, allocated_rbs: int, active_ues: int) -> None:
        """
        Обновить счетчики статистики.

        Args:
            tti (int): Номер текущего TTI
            eligible_count (int): Кол-во eligible UE
            allocated_rbs (int): Всего выделено RB
            active_ues (int): Кол-во UE с allocation > 0
        """
        self._last_eligible_ue_count    = eligible_count
        self._last_allocated_rb_count   = allocated_rbs
        self._last_active_ue_count      = active_ues
        self._last_tti                  = tti

    def _get_active_ue_stats(self) -> Dict:
        """
        Статистика скользящего окна.

        Вспомогательный метод для отладки и анализа активности UE.
        Возвращает информацию о скользящем окне:
        - Включено ли окно
        - Текущее количество активных UE
        - Уникальные UE за весь период окна

        Returns:
            Dict со статистикой окна или None если окно отключено
        """
        if not self.enable_window:
            return {
                'window_enabled': False,
                'message': 'Sliding window is disabled'}

        if not self.active_ue_window:
            return {
                'window_enabled': True,
                'window_size': self.window_size,
                'current_active_count': 0,
                'unique_ues_in_window': 0}

        unique_ues = len(set.union(*self.active_ue_window))

        return {
            "window_enabled": True,
            "window_size_tti": self.window_size,
            "current_depth_tti": len(self.active_ue_window),
            "unique_ues_in_window": unique_ues,
            "avg_ues_per_tti": sum(len(s) for s in self.active_ue_window) / len(self.active_ue_window)
        }

    def _save_timing_stats(self, sch_time_us: float,
                       priority_calc_time_us: float,
                       priority_sort_time_us: float,
                       ) -> None:
        """
        Сохранить timing статистику для get_stats().

        Args:
            total_time_us: Полное время schedule() (микросекунды)
            priority_calc_time_us: Время _calculate_priorities() (микросекунды)
            priority_sort_time_us: Время _form_priority_list() (микросекунды)
        """
        self._last_sch_time_us = sch_time_us
        self._last_priority_calc_time_us = priority_calc_time_us
        self._last_priority_sort_time_us = priority_sort_time_us

#===============АБСТРАКТНЫЕ МЕТОДЫ ДЛЯ АЛГОРИТМОВ ПЛАНИРОВАНИЯ=================

    def _calculate_priorities(self, windowed_ues: List[Dict], tti: int) -> List[Dict]:
        """
        Расчет приоритетов для UE на основе алгоритма планирования.
        Подклассы ОБЯЗАНЫ реализовать этот метод.

        !Контракт: добавить поле user['priority'] для каждого UE.

        Примеры реализации:
        - BestCQI: priority = cqi
        - ProportionalFair: priority = instant_rate / avg_throughput
        - RoundRobin: priority = 1.0 (все равны)

        Args:
            windowed_ues: Отфильтрованные UE (прошли eligibility checks)
            tti: Текущий TTI

        Returns:
            List[Dict]: UE с добавленным полем 'priority'

        Raises:
            NotImplementedError: Если подкласс не реализовал метод
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} должен реализовать метод _calculate_priorities()")

    def _allocate_pdsch(self, tti: int, ues_with_pdcch: List[Dict],
                       eligible_ues: List[Dict]) -> Dict[int, List[int]]:
        """
        Распределение PDSCH ресурсов (RBG) между UE.
        Подклассы ОБЯЗАНЫ реализовать этот метод.

        !Контракт:
        - Инициализировать allocation dict для всех eligible UE
        - Распределить RBG между ues_with_pdcch
        - Учитывать remaining_buffer (уменьшать в цикле RBG!)
        - Возвращать Dict[UE_ID, List[freq_idx]]

        Примечание для RoundRobin:
        RR использует sequential (последовательно) PDCCH/PDSCH allocation,
        поэтому:
        1. Переопределяет _allocate_pdcch() как pass-through
        2. Выделяет PDCCH внутри _allocate_pdsch() (в цикле RBG)

        Args:
            tti: Текущий TTI
            ues_with_pdcch: UE с успешно выделенным PDCCH
            eligible_ues: Все eligible UE (для инициализации allocation)

        Returns:
            Dict[int, List[int]]: Allocation map (UE_ID -> список freq_idx)

        Raises:
            NotImplementedError: Если подкласс не реализовал метод
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} должен реализовать метод _allocate_pdsch()")

# =================================HARQ========================================

    def _get_harq_retransmissions(self, tti: int) -> List[Dict]:
        """
        Когда HARQ Manager будет реализован, этот метод должен возвращать
        список UE, требующих retransmission на данном TTI.

        Retransmissions имеют приоритет над new transmissions и должны
        обрабатываться до основного планирования (до ЭТАПА 3).

        #TODO: Интеграция HARQ Manager
        - Добавить self.harq_manager в __init__
        - Вызывать этот метод в schedule() перед ЭТАПОМ 3
        - Обработать retransmissions в _allocate_pdsch() с высоким приоритетом
        - Retransmissions используют тот же HARQ process ID

        Args:
            tti: Текущий TTI

        Returns:
            List[Dict]: UE с pending retransmissions (пока пустой список)
        """
        if self.harq_manager is not None:
            return self.harq_manager.get_retransmissions(tti)
        return []  # Пока HARQ не реализован

#TODO: Задачка оптимизаторам. С секретом. Посмотрите как сортируется
# priority_list. Интересно, почему же он не ограничен по количеству элементов?

#==============================================================================
#                              ЛОГИКА PDCCH
#==============================================================================

class PDCCHManager:
    def __init__(self, bandwidth: int,
                 pcfich: int = 2,
                 max_dl_cce_allowance: Optional[int] = None,
                 verbose: bool = False):
        """
        Args:
            bandwidth: Ширина полосы, влияет на кол-во CCE
            pcfich: Число OFDM символов для PDCCH (1, 2, или 3)
            max_dl_cce_allowance: Максимум CCE для DL UE-specifiс
                                  None = использовать весь Total_CCE
            verbose: режим детального логирования для дебага
        Raises:
            ValueError: при некорректных параметрах
        """

        self._validate_parameters(bandwidth, pcfich)

        self.bandwidth = bandwidth
        self.pcfich    = pcfich
        self.verbose   = verbose

        self.total_cce = self._calculate_total_cce()

        # определение лимита для DL по PDCCH
        if max_dl_cce_allowance is None:
            self.max_dl_cce = self.total_cce
        else:
            if max_dl_cce_allowance > self.total_cce:
                raise ValueError(
                    f"max_dl_cce_allowance ({max_dl_cce_allowance})"
                    f"Cannot exceed total_cce! ({self.total_cce})")
            self.max_dl_cce = max_dl_cce_allowance

        # определение текущего tti
        self.num_assigned_cce = 0
        self.cce_allocations  = {}    # {ue_id: cce_count}
        self.history          = []

        # для дебага
        if self.verbose:
            print(
                f"[PDCCH] Initialized: Bandwidth={bandwidth} MHz, PCFICH={pcfich}, "
                f"Total CCE={self.total_cce}, Max DL CCE={self.max_dl_cce}")

    def _validate_parameters(self, bandwidth: float, pcfich: int):
        """
        Валидация входных параметров.

        Args:
            bandwidth: Ширина полосы в МГц
            pcfich: Число OFDM символов

        Raises:
            ValueError: При некорректных значениях
        """
        # Проверка bandwidth
        valid_bandwidths = [1.4, 3, 5, 10, 15, 20]
        if bandwidth not in valid_bandwidths:
            raise ValueError(
                f"Unsupported bandwidth: {bandwidth} MHz. "
                f"Valid values: {valid_bandwidths}"
            )

        # Проверка pcfich в зависимости от bandwidth
        if bandwidth == 1.4:
            valid_pcfich = [2, 3, 4]
        else:  # >= 3 MHz
            valid_pcfich = [1, 2, 3]

        if pcfich not in valid_pcfich:
            raise ValueError(
                f"Invalid pcfich: {pcfich} for bandwidth {bandwidth} MHz. "
                f"Valid values: {valid_pcfich}"
            )

        # Предупреждение для граничных случаев
        if bandwidth == 5 and pcfich == 1:
            import warnings
            warnings.warn(
                "PCFICH=1 with 5MHz bandwidth provides only 8 CCE, "
                "which may be insufficient for Common/Broadcast messages. "
                "Consider using PCFICH=2 (12 CCE) or PCFICH=3 (20 CCE)."
            )

    def _calculate_total_cce(self) -> int:
        """
        Расчет общего числа CCE (Control Channel Elements) на основе
        ширины полосы пропускания и PCFICH.

        Таблица взята из LTE Release 15 Scheduler Design Document, Table 8.

        Returns:
            int: Общее число CCE, доступных в одном subframe

        Raises:
            ValueError: Если комбинация (bandwidth, pcfich) не поддерживается

        Note:
            - Для 1.4 MHz: PCFICH может быть 2, 3 или 4
            - Для >= 3 MHz: PCFICH может быть 1, 2 или 3
            - PCFICH=2 - наиболее распространенное значение (баланс между PDCCH и PDSCH)
        """
        # Таблица CCE (bandwidth_MHz, pcfich) -> total_cce
        # для полос 1.4 и 3 МГц очень спорно, нет информации в стандартах
        CCE_TABLE = {
            # 1.4 MHz (6 RB)
            (1.4, 1): 2, # НЕ используется, слишком маленькое!
            (1.4, 2): 4, # используется редко, для служебн. инф. или VoLTE
            (1.4, 3): 6,

            # 3 MHz (15 RB)
            (3, 1): 2,
            (3, 2): 7,
            (3, 3): 12,

            # 5 MHz (25 RB)
            (5, 1): 3,
            (5, 2): 12,
            (5, 3): 20,

            # 10 MHz (50 RB)
            (10, 1): 8,
            (10, 2): 25,
            (10, 3): 41,

            # 15 MHz (75 RB)
            (15, 1): 12,
            (15, 2): 37,
            (15, 3): 62,

            # 20 MHz (100 RB)
            (20, 1): 17,
            (20, 2): 50,
            (20, 3): 84,
        }

        key = (self.bandwidth, self.pcfich)

        if key not in CCE_TABLE:
            raise ValueError(
                f"Unsupported combination: bandwidth={self.bandwidth} MHz, pcfich={self.pcfich}. "
                f"Please check LTE standard specifications (TS 36.211, TS 36.213)."
            )

        total_cce = CCE_TABLE[key]

        if self.verbose:
            print(f"[PDCCH] Calculated Total CCE: {total_cce} "
                  f"(Bandwidth={self.bandwidth}MHz, PCFICH={self.pcfich})")

        return total_cce
    #TODO: вынести таблицы в GLOBALS при следующем апдейте

    def get_aggregation_level(self, cqi: int) -> int:
        """
        Определение Aggregation Level (уровня агрегации CCE) на основе CQI.

        Aggregation Level показывает, сколько CCE требуется для передачи PDCCH
        одному пользователю. Зависит от качества канала (CQI): чем хуже канал,
        тем больше CCE нужно для надежной передачи управляющей информации.

        Args:
            cqi: Channel Quality Indicator (1-15)
                15 = отличное качество канала
                1 = очень плохое качество канала

        Returns:
            int: Aggregation Level - число CCE (1, 2, 4 или 8)

        Raises:
            ValueError: Если CQI вне диапазона [1-15]

        Mapping (упрощенная модель):
            CQI 13-15: 1 CCE  (отличное качество, минимальные ресурсы)
            CQI 10-12: 2 CCE  (хорошее качество)
            CQI 7-9:   4 CCE  (среднее качество)
            CQI 1-6:   8 CCE  (плохое качество, максимальная защита)

        Note:
            В реальных системах используются более сложные алгоритмы с учетом
            SINR, interference, mobility, и истории HARQ NACK. Текущая упрощенная
            модель нам подходит для симуляции на уровне планировщика.
        """
        if not isinstance(cqi, int) or cqi < 1 or cqi > 15:
            raise ValueError(
                f"Invalid CQI: {cqi}. CQI must be an integer in range [1-15]."
            )

        if cqi >= 13:
            aggregation_level = 1
        elif cqi >= 10:
            aggregation_level = 2
        elif cqi >= 7:
            aggregation_level = 4
        else:  # cqi <= 6
            aggregation_level = 8

        if self.verbose:
            print(f"[PDCCH] CQI={cqi} -> Aggregation Level={aggregation_level} CCE")

        return aggregation_level

    def check_cce_availability(self, required_cce: int) -> bool:
        """
        Проверка доступности CCE для выделения PDCCH.

        Метод проверяет, достаточно ли свободных CCE для выделения PDCCH
        с заданным Aggregation Level, не превышая лимит max_dl_cce.

        Args:
            required_cce: Требуемое количество CCE (обычно 1, 2, 4 или 8)

        Returns:
            bool: True если CCE доступны, False если недостаточно

        Note:
            Метод НЕ изменяет состояние PDCCHManager, только проверяет.
            Для фактического выделения используйте allocate_cce().
        """
        available_cce = self.max_dl_cce - self.num_assigned_cce

        is_available  = required_cce <= available_cce

        if self.verbose:
            if is_available:
                print(f"[PDCCH] Check: {required_cce} CCE requested, "
                      f"{available_cce} available -> ✅ ALLOCATED")
            else:
                print(f"[PDCCH] Check: {required_cce} CCE requested, "
                      f"{available_cce} available -> ❌ INSUFFICIENT")

        return is_available

    def allocate_cce(self, ue_id: int, cce_count: int) -> bool:
        """
        Выделение CCE для PDCCH конкретного пользователя.

        Метод выполняет:
        1. Проверку доступности CCE
        2. Увеличение счетчика выделенных CCE
        3. Сохранение информации о выделении

        Args:
            ue_id: Идентификатор пользователя (UE ID)
            cce_count: Количество CCE для выделения (обычно 1, 2, 4 или 8)

        Returns:
            bool: True если выделение успешно, False если:
                  - Недостаточно CCE
                  - Пользователь уже получил PDCCH в этом TTI (warning)

        Note:
            - Метод изменяет состояние PDCCHManager (num_assigned_cce, cce_allocations)
            - Перед вызовом рекомендуется использовать check_cce_availability()
            - При повторном выделении одному UE выдается warning и возвращается False
        """
        # Проверка получил ли уже этот UE CCE в этом TTI
        if ue_id in self.cce_allocations:
            if self.verbose:
                print(f"[PDCCH] WARNING: UE {ue_id} already has PDCCH allocated "
                      f"({self.cce_allocations[ue_id]} CCE). Ignoring second allocation.")
            return False

        # Проверка доступности CCE
        if not self.check_cce_availability(cce_count):
            if self.verbose:
                available = self.max_dl_cce - self.num_assigned_cce
                print(f"[PDCCH] BLOCKED: UE {ue_id} cannot allocate {cce_count} CCE. "
                      f"Only {available} CCE available.")
            return False

        # Выделение CCE
        self.num_assigned_cce      += cce_count
        self.cce_allocations[ue_id] = cce_count

        if self.verbose:
            utilization = (self.num_assigned_cce / self.max_dl_cce) * 100
            print(f"[PDCCH] ALLOCATED: UE {ue_id} → {cce_count} CCE. "
                  f"Total used: {self.num_assigned_cce}/{self.max_dl_cce} ({utilization:.1f}%)")

        return True

    def reset_tti(self) -> None:
        """
        Сброс состояния CCE для нового TTI.

        Метод вызывается в начале каждого TTI перед планированием PDCCH
        для обнуления счетчиков и освобождения информации о выделениях
        предыдущего TTI.

        Выполняет:
            1. Обнуление счетчика выделенных CCE (num_assigned_cce = 0)
            2. Очистку словаря выделений (cce_allocations = {})

        Returns:
            None
        """
        self.num_assigned_cce = 0

        self.cce_allocations.clear()

        if self.verbose:
            print(f"[PDCCH] TTI reset: CCE counters cleared. "
                  f"Available CCE: {self.max_dl_cce}/{self.max_dl_cce}")

    def release_cce(self, ue_id: int) -> bool:
        """
        Освобождение CCE, выделенных для пользователя (ЗАГЛУШКА для дальнейшей разработки).

        СЦЕНАРИЙ ПРИМЕНЕНИЯ:
        1. Планировщик успешно выделил PDCCH для пользователя (allocate_cce())
        2. Планировщик пытается выделить PDSCH ресурсы (RBG)
        3. PDSCH ресурсы исчерпаны (lte_grid.ALLOCATE_RBG() вернул False)
        4. Необходимо освободить выделенный PDCCH → вызов release_cce()
        5. CCE возвращаются в пул доступных для других пользователей

        ПЛАНИРУЕМАЯ ЛОГИКА (v2):

        Шаг 1: Проверка существования выделения
        Шаг 2: Получение количества выделенных CCE
        Шаг 3: Уменьшение счетчика
        Шаг 4: Удаление записи из словаря
        Шаг 5: Verbose логирование
        Шаг 6: Возврат успеха
        ПРИМЕР ИНТЕГРАЦИИ В ПЛАНИРОВЩИК (v2):

        EDGE CASES (граничные случаи):
        1. Попытка освободить несуществующее выделение:
           release_cce(ue_id=999) где UE 999 не имеет PDCCH
           → return False, warning в verbose mode
        2. Освобождение после reset_tti():
           Если reset_tti() уже вызван, cce_allocations пуст
           → return False (ничего не освободить)
        3. Множественные освобождения:
           release_cce(1), затем release_cce(1) снова
           → Первый успешен, второй возвращает False

        Почему отложил до будущего:
        Сложно интегрировать, требуется вновь переписать логику планировщика
        Это пока не стоит того, PDSCH обычно имеет больше ресурсов чем PDCCH
        Поэтому, сначала реализуем базовую блокировку PDCCH
        А потом уже нужны дополнительные тесты для edge cases

        Args:
            ue_id: Идентификатор пользователя

        Returns:
            bool: True если освобождение успешно, False если UE не найден
                  В v1 всегда возвращает False (не реализовано)
        """
        # TODO: Реализовать PDCCH Release
        # См. детальное описание планируемой логики в docstring выше

        if self.verbose:
            print(f"[PDCCH] WARNING: release_cce() called for UE {ue_id} but not implemented in v1. "
                  f"CCE will be freed automatically in next TTI via reset_tti().")

        return False  # Заглушка

#=============================GETTER-МЕТОДЫ====================================

    def get_total_cce(self) -> int:
        """
        Получить общее число CCE в системе.

        Returns:
            int: Total CCE
        """
        return self.total_cce

    def get_max_dl_cce(self) -> int:
        """
        Получить максимум CCE для DL UE-specific передач.

        Returns:
            int: Max DL CCE (DBdlNumCceAllowance или total_cce)
        """
        return self.max_dl_cce

    def get_used_cce(self) -> int:
        """
        Получить число выделенных CCE в текущем TTI.

        Returns:
            int: Количество выделенных CCE (0 до max_dl_cce)
        """
        return self.num_assigned_cce

    def get_available_cce(self) -> int:
        """
        Получить число доступных CCE для выделения.

        Returns:
            int: Количество доступных CCE (max_dl_cce - used_cce)
        """
        return self.max_dl_cce - self.num_assigned_cce

    def get_stats(self) -> Dict:
        """
        Получение статистики использования CCE в текущем TTI.

        Возвращает детальную информацию о состоянии PDCCH, включая:
        - Общие параметры конфигурации
        - Текущее использование CCE
        - Коэффициент утилизации
        - Детальную информацию о выделениях по пользователям

        Returns:
            dict: Словарь со статистикой, содержащий:
                - 'total_cce': int - Общее число CCE в системе
                - 'max_dl_cce': int - Максимум CCE для DL UE-specific
                - 'used_cce': int - Количество выделенных CCE
                - 'available_cce': int - Количество доступных CCE
                - 'utilization': float - Коэффициент утилизации (0.0-1.0)
                - 'utilization_percent': float - Утилизация в процентах
                - 'num_scheduled_users': int - Число пользователей с PDCCH
                - 'allocations': dict - Словарь {ue_id: cce_count}
        """
        # available_cce = self.max_dl_cce - self.num_assigned_cce

        if self.max_dl_cce > 0:
            cce_utilization = self.num_assigned_cce / self.max_dl_cce
        else:
            cce_utilization = 0.0

        stats = {
            'pdcch_cce_total_count': self.total_cce,
            #'max_dl_cce': self.max_dl_cce,
            'pdcch_cce_allocated_count': self.num_assigned_cce,
            #'pdcch_cce_available_count': available_cce,
            'pdcch_cce_utilization_pct': round(cce_utilization*100, 2),
            #'num_scheduled_users': len(self.cce_allocations),
            'pdcch_ue_cce_allocations': self.cce_allocations.copy()} #Копия, не оригинал

        return stats

#==============================================================================
#                               ЛОГИКА AMC
#==============================================================================

class AdaptiveModulationAndCoding:
    """
    Класс для преобразования CQI в MCS и расчета бит на ресурсный блок.
    """

    # Таблица соответствия CQI → (Modulation Order, Code Rate)
    CQI_TO_MCS = {
        1: (2, 0.152),   # QPSK
        2: (2, 0.234),   # QPSK
        3: (2, 0.377),   # QPSK
        4: (2, 0.601),   # QPSK
        5: (4, 0.369),   # 16QAM
        6: (4, 0.479),   # 16QAM
        7: (4, 0.601),   # 16QAM
        8: (6, 0.455),   # 64QAM
        9: (6, 0.554),   # 64QAM
        10: (6, 0.650),  # 64QAM
        11: (6, 0.754),  # 64QAM
        12: (6, 0.852),  # 64QAM
        13: (6, 0.926),  # 64QAM
        14: (6, 0.953),  # 64QAM
        15: (6, 0.978)   # 64QAM
    }

    def __init__(self, scheduler=None):
        self.scheduler = scheduler

    def GET_BITS_PER_RB(self, cqi: int, n_ports: int = 1) -> int:
        """
        Рассчитать количество бит на ресурсный блок (RB) для заданного CQI.
        """
        if cqi not in self.CQI_TO_MCS:
            raise ValueError(f"Invalid CQI {cqi}. Must be 1-15.")

        pcfich = self.scheduler.pdcch_manager.pcfich if self.scheduler else 2

        modulation, code_rate = self.CQI_TO_MCS[cqi]

        rs_per_slot           = n_ports * 4          # CRS: 4 RE/slot при 1 порте
        re_slot0              = (7 - pcfich) * 12 - rs_per_slot
        re_slot1              = 7 * 12 - rs_per_slot
        re_per_rb_tti         = re_slot0 + re_slot1

        return int(re_per_rb_tti * modulation * code_rate)

    # @sherokiddo: "Добавить зависимость от CP"
    # TODO: [AMC] Заменить формульный расчёт RE на lookup-таблицу TBS
    # по TS 36.213 Table 7.1.7.2.1 (I_TBS + N_PRB → TBS).
    # Текущая формула даёт отклонение ~8-10% от реального TBS.
    # Точный метод: CQI → I_MCS → I_TBS → TBS_TABLE[I_TBS][N_PRB]
    # Ref: 3GPP TS 36.213, Section 7.1.7.2 [enhancement]

    def get_stats(self) -> Dict:
        """
        Получить статистику AMC за последний TTI из кэша scheduler.
        Рассчитывает throughput на основе _last_allocation и _last_users.
        Вызывается StatsManager.

        Returns:
            Dict: {
                "total_throughput": float,     # Общий throughput в bps
                "avg_bits_per_rb": float,      # Средние биты на RB
                "ue_throughputs": Dict         # UEID -> throughput (bps)
            }
        """
        if not self.scheduler or not hasattr(self.scheduler, '_last_users'):
            return {
                "dl_capacity_bits_sum_tti": 0,
                "dl_transmitted_bits_sum_tti": 0,
                "dl_throughput_sum_kbps": 0.0,
                "dl_bits_per_rb_avg": 0.0,
                "dl_cqi_wb_avg_idx": 0.0,
                "dl_sinr_avg": 0.0,
                "dl_ue_throughputs": {},
                "ue_cqi": {},
                "ue_cqi_sbb": {},
                "ue_sinr": {},
                "ue_rb_allocated": {},
            }

        users = self.scheduler._last_users
        allocation = self.scheduler._last_allocation

        total_throughput_bps    = 0.0
        total_capacity_bits     = 0
        total_transmitted_bits  = 0
        total_allocated_rbs     = 0

        ue_throughputs = {user.get('UE_ID'): 0 for user in users}

        ue_cqi = {}
        ue_cqi_sbb = {}
        ue_sinr = {}
        ue_rb_allocated = {}
        cqi_values = []
        sinr_values = []

        for user in users:
            ue = user.get('ue')
            ue_id = user.get('UE_ID')

            if not ue or not ue_id:
                continue

            if ue_id in self.scheduler.cqi_map:
                cqi_entry = self.scheduler.cqi_map[ue_id]

                cqi = cqi_entry.wb_cqi
                ue_cqi[ue_id] = cqi

                if cqi_entry.sb_cqi and len(cqi_entry.sb_cqi) > 0:
                    ue_cqi_sbb[ue_id] = cqi_entry.sb_cqi.copy()
            else:
                # Fallback
                cqi = 0

            #TODO: вынести cqi из цикла

            throughput_bps        = ue.current_dl_throughput
            ue_throughputs[ue_id] = throughput_bps
            total_throughput_bps += throughput_bps

            allocated_rbs = len(allocation.get(ue_id, []))
            ue_rb_allocated[ue_id] = allocated_rbs
            
            if allocated_rbs > 0 and 1 <= cqi <= 15:
                bits_per_rb     = self.GET_BITS_PER_RB(cqi)
                capacity_bits   = bits_per_rb * allocated_rbs
                
                total_capacity_bits += capacity_bits

            transmitted_bits        = int(ue.last_transmitted_bits)
            total_transmitted_bits += transmitted_bits
            total_allocated_rbs    += allocated_rbs

            cqi_values.append(cqi)

            if hasattr(ue, 'SINR'):
                sinr_values.append(ue.SINR)
                ue_sinr[ue_id] = ue.SINR

        # Avg bits per RB (на основе actual)
        if total_allocated_rbs > 0:
            avg_bits_per_rb = total_transmitted_bits / (total_allocated_rbs)
        else:
            avg_bits_per_rb = 0.0

        if cqi_values:
            cqi_avg = sum(cqi_values) / len(cqi_values)
        else: cqi_avg = 0

        if sinr_values:
            sinr_avg = sum(sinr_values) / len(sinr_values)
        else: sinr_avg = 0

        stats = {
            "dl_capacity_bits_sum_tti": total_capacity_bits,
            "dl_transmitted_bits_sum_tti": total_transmitted_bits,
            "dl_throughput_sum_kbps": round(total_throughput_bps/1000, 2),
            "dl_bits_per_rb_avg": round(avg_bits_per_rb, 2),
            "dl_cqi_wb_avg_idx": round(cqi_avg, 2),
            "dl_sinr_avg": round(sinr_avg, 2),
            "dl_ue_throughputs": ue_throughputs,
            "ue_cqi": ue_cqi,
            "ue_cqi_sbb": ue_cqi_sbb,
            "ue_sinr": ue_sinr,
            "ue_rb_allocated": ue_rb_allocated,}

        return stats

#==============================================================================
#                              ЛОГИКА SCHEDULER_NEW
#==============================================================================

class BestCQIScheduler(SchedulerInterface):
    """
    Best CQI Scheduler.
    Жадный алгоритм: выбирает UE с лучшим CQI (максимальный мгновенный throughput).
    Фокус на максимизацию системного throughput, но может приводить к голоданию
    для UE с плохими условиями канала.

    Алгоритм:
    1. Priority = CQI (чем выше CQI, тем выше приоритет)
    2. Сортировка UE по priority (descending)
    3. Жадное распределение RBG лучшему UE пока есть данные

    Args:
        lte_grid: Ресурсная сетка LTE
        bs: Базовая станция
        max_dl_ue_tti: Максимум UE за TTI (default: None = без ограничений)
        pcfich: PCFICH value (default: 2)
        max_dl_cce_allowance: Лимит CCE (default: None)
        verbose_pdcch: Отладочный вывод для PDCCH (default: False)
        enable_window: Включить скользящее окно (default: True)
        window_size: Размер окна в TTI (default: 100)
        verbose: Отладочный вывод для планировщика (default: False)
    """

    def __init__(self, lte_grid, bs, **kwargs):
        """Инициализация BestCQI scheduler."""
        super().__init__(
            lte_grid=lte_grid,
            bs=bs,
            **kwargs)

    def _calculate_priorities(self, eligible_ues: List[Dict], tti: int) -> List[Dict]:
        """
        Args:
            eligible_ues: Отфильтрованные UE
            tti: Текущий TTI

        Returns:
            List[Dict]: UE с добавленным полем 'priority'
        """
        for user in eligible_ues:
            ue_id = user['UE_ID']
            user['priority'] = self._get_wb_cqi(ue_id)  # Priority = CQI

        if self.verbose:
            avg_priority = sum(u['priority'] for u in eligible_ues) / len(eligible_ues)
            print(f"[SCHEDULER.BestCQI TTI {tti}] Priority calculation: {len(eligible_ues)} UE, avg priority={avg_priority:.2f}")

        return eligible_ues

    #TODO: Уважаемые оптимизаторы. Вашему вниманию представляется 100%
    # неоптимизированная логика. Ваша задача изучить, как лучше всего
    # формировать приоритеты для BCQI

    def _allocate_pdsch(self, tti: int, ues_with_pdcch: List[Dict],
                       eligible_ues: List[Dict]) -> Dict[int, List[int]]:
        """
        Жадное распределение RBG: отдаем все RBG лучшему UE,
        пока у него есть данные, потом следующему лучшему.
        """
        allocation = {user['UE_ID']: [] for user in eligible_ues}

        if not ues_with_pdcch:
            return allocation

        remaining_buffer = {
            user['UE_ID']: user['bs_buffer_size'] * 8
            for user in ues_with_pdcch}

        rbg_size  = self.lte_grid.GET_RBG_SIZE()
        total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size

        for rbg_idx in range(total_rbg):
            if all(buf <= 0 for buf in remaining_buffer.values()):
                break

            best_user = None
            for user in ues_with_pdcch:
                if remaining_buffer[user['UE_ID']] > 0:
                    best_user = user
                    break

            if best_user is None:
                break

            ue_id = best_user['UE_ID']

            # Выделение RBG
            if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
                allocation[ue_id].extend(rb_indices)

                # Уменьшаем remaining_buffer
                cqi             = self._get_wb_cqi(ue_id)
                bits_per_rb     = self.amc.GET_BITS_PER_RB(cqi)
                rbg_capacity    = len(rb_indices) * bits_per_rb

                remaining_buffer[ue_id] -= min(remaining_buffer[ue_id], rbg_capacity)

        if self.verbose:
            allocated_ues = [ue_id for ue_id, rbs in allocation.items() if len(rbs) > 0]
            for ue_id in allocated_ues:
                rb_list = allocation[ue_id]
                print(f"[PDSCH] UE {ue_id}: {len(rb_list)} RB allocated "
                      f"(RB {rb_list[0]}-{rb_list[-1]})")

        return allocation

class RoundRobinScheduler(SchedulerInterface):
    """
    Round Robin Scheduler - циклическое распределение с ротацией.
    Использует счетчик для справедливой ротации UE между TTI.
    """

    def __init__(self, lte_grid, bs, **kwargs):
        super().__init__(lte_grid, bs, **kwargs)
        self.rr_rbg_offset = 0 #можно сделать механизм без ротации RBG
        self.rr_ue_offset = 0

    def _apply_pdsch_estimation(self, priority_list: List[Dict], tti: int) -> List[Dict]:
        """
        RR: Не применяем PDSCH estimation. Костыль. Очень костыльный.
        Round Robin распределяет ресурсы поровну (по 1 RBG каждому),
        поэтому estimation (предназначенная для жадных алгоритмов) не подходит.

        Args:
            priority_list: UE для планирования
            tti: Текущий TTI

        Returns:
            List[Dict]: Тот же priority_list без изменений
        """
        if self.verbose:
            print(f"[SCHEDULER TTI {tti}] PDSCH estimation: SKIPPED (Round Robin distributes evenly)")
            print(f"[SCHEDULER TTI {tti}] After estimation: {len(priority_list)} UE selected, 0 UE excluded")

        return priority_list

    def _calculate_priorities(self, windowed_ues: List[Dict], tti: int) -> List[Dict]:
        """
        Round Robin принцип распределения ресурсов
        """
        num_ues = len(windowed_ues)
        if num_ues == 0:
            return windowed_ues

        if self.max_dl_ue_tti and self.max_dl_ue_tti < num_ues:
            ues_to_plan = self.max_dl_ue_tti
        else:
            ues_to_plan = num_ues

        start_idx   = self.rr_ue_offset % num_ues
        rotated_ues = windowed_ues[start_idx:] + windowed_ues[:start_idx]

        for idx, user in enumerate(rotated_ues):
            user['priority'] = ues_to_plan - idx

        if self.verbose:
            top_ue = rotated_ues[0]['UE_ID']
            print(f"[SCHEDULER.RoundRobin TTI {tti}] Priority: Rotated (start UE {top_ue}, offset={self.rr_ue_offset})")

        return rotated_ues

    def _allocate_pdsch(self, tti: int, ues_with_pdcch: List[Dict],
                        eligible_ues: List[Dict]) -> Dict[int, List[int]]:
        """
        Round Robin выделение ресурса PDSCH allocation
        Каждому UE выделяется по одному RBG циклически.

        Args:
            tti: Текущий TTI
            ues_with_pdcch: UE получившие PDCCH
            eligible_ues: Все eligible UE (не используется в RR)

        Returns:
            Dict[int, List[int]]: Allocation map {ue_id: [rb_indices]}
        """
        allocation = {ue['UE_ID']: [] for ue in ues_with_pdcch}
        rbg_size   = self.lte_grid.GET_RBG_SIZE()
        total_rbg  = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size

        num_ues = len(ues_with_pdcch)
        if num_ues == 0:
            return allocation

        eligible_ids = [ue['UE_ID'] for ue in eligible_ues]
        ue_index     = self.rr_rbg_offset
        last_served_idx = -1

        for rbg_idx in range(total_rbg):
            ue    = ues_with_pdcch[ue_index % num_ues]
            ue_id = ue['UE_ID']

            if ue['bs_buffer_size'] > 0:
                if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                    rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
                    allocation[ue_id].extend(rb_indices)

                    try:
                        current_idx = eligible_ids.index(ue_id)
                        if current_idx > last_served_idx:
                            last_served_idx = current_idx
                    except ValueError:
                        pass

                    cqi                 = self._get_wb_cqi(ue_id)
                    bits_per_rb         = self.amc.GET_BITS_PER_RB(cqi)
                    transmitted_bits    = len(rb_indices) * bits_per_rb
                    transmitted_bytes   = transmitted_bits // 8
                    ue['bs_buffer_size'] = max(0, ue['bs_buffer_size'] - transmitted_bytes)

            ue_index += 1

        self.rr_rbg_offset = ue_index % num_ues

        if last_served_idx >= 0 and len(eligible_ues) > 0:
            self.rr_ue_offset = (last_served_idx + 1) % len(eligible_ues)

        if self.verbose:
            allocated_ues = sum(1 for rbs in allocation.values() if len(rbs) > 0)
            total_rb = sum(len(rbs) for rbs in allocation.values())
            print(f"[SCHEDULER.RoundRobin TTI {tti}] PDSCH: {allocated_ues} UE, {total_rb} RB total "
                  f"(next_offset={self.rr_rbg_offset})")

        return allocation

class ProportionalFairScheduler(SchedulerInterface):
    """
    Proportional Fair Scheduler - баланс между throughput и fairness.

    Priority: instant_rate / avg_throughput
    - instant_rate: Потенциальный throughput в текущем TTI (зависит от CQI)
    - avg_throughput: Средний throughput UE (из history)

    UE с низким avg_throughput получают высший приоритет (fairness).
    UE с хорошим CQI тоже получают бонус (efficiency).
    """

    def __init__(self, lte_grid, bs, **kwargs):
        super().__init__(lte_grid, bs, **kwargs)

    def _calculate_priorities(self, eligible_ues: List[Dict], tti: int) -> List[Dict]:
        """
        Args:
            eligible_ues: UE прошедшие eligibility проверку
            tti: Текущий TTI

        Returns:
            List[Dict]: UE с рассчитанными PF-приоритетами
        """
        for user in eligible_ues:
            ue_id       = user['UE_ID']
            cqi         = self._get_wb_cqi(ue_id)
            bits_per_rb = self.amc.GET_BITS_PER_RB(cqi)
            rb_per_slot = self.lte_grid.rb_per_slot
            instant_rate = rb_per_slot * bits_per_rb
            avg_throughput = user['ue'].average_throughput

            if avg_throughput <= 0:
                avg_throughput = 1e-6
                pf_metric = instant_rate / 1.0
            else:
                avg_throughput_per_tti = avg_throughput / 1000
                pf_metric = instant_rate / avg_throughput_per_tti

            user['priority']        = pf_metric
            user['instant_rate']    = instant_rate  # Для debug

            # avg_throughput = user['ue'].average_throughput
            # if avg_throughput <= 0:
            #     avg_throughput = 1e-6

            # pf_metric = instant_throughput / avg_throughput

            # return pf_metric

        if self.verbose:
            avg_priority = sum(u['priority'] for u in eligible_ues) / len(eligible_ues) if eligible_ues else 0
            print(f"[SCHEDULER.ProportionalFair TTI {tti}] Priority calculation: "
                  f"{len(eligible_ues)} UE, avg PF metric={avg_priority:.2f}")

        return eligible_ues

    def _allocate_pdsch(self, tti: int, ues_with_pdcch: List[Dict],
                        eligible_ues: List[Dict]) -> Dict[int, List[int]]:
        """
        Выделяем RBG по приоритету (highest priority first).

        Args:
            tti: Текущий TTI
            ues_with_pdcch: UE получившие PDCCH (отсортированы по priority)
            eligible_ues: Все eligible UE

        Returns:
            Dict[int, List[int]]: Allocation map {ue_id: [rb_indices]}
        """
        allocation = {ue['UE_ID']: [] for ue in eligible_ues}

        if not ues_with_pdcch:
            return allocation

        remaining_buffer = {ue['UE_ID']: ue['bs_buffer_size'] * 8 for ue in ues_with_pdcch}

        rbg_size = self.lte_grid.GET_RBG_SIZE()
        total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size

        for rbg_idx in range(total_rbg):
            if all(buf <= 0 for buf in remaining_buffer.values()):
                break

            best_user = None
            for user in ues_with_pdcch:
                if remaining_buffer[user['UE_ID']] > 0:
                    best_user = user
                    break

            if best_user is None:
                break

            ue_id = best_user['UE_ID']

            if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
                allocation[ue_id].extend(rb_indices)

                cqi             = self._get_wb_cqi(ue_id)
                bits_per_rb     = self.amc.GET_BITS_PER_RB(cqi)
                rbg_capacity    = len(rb_indices) * bits_per_rb      # bits

                remaining_buffer[ue_id] -= min(remaining_buffer[ue_id], rbg_capacity)

        if self.verbose:
            allocated_ues = sum(1 for rbs in allocation.values() if len(rbs) > 0)
            total_rb = sum(len(rbs) for rbs in allocation.values())
            print(f"[SCHEDULER.ProportionalFair TTI {tti}] PDSCH: {allocated_ues} UE, {total_rb} RB total")

        return allocation

class FDxBestCQIScheduler(SchedulerInterface):
    """
    Frequency Domain BestCQI Scheduler

    Алгоритм:
    1. Priority = wb_cqi (чем выше CQI, тем выше приоритет)
    2. Сортировка UE по priority (descending)
    3. Жадное распределение per-RBG лучшему UE опираясь на sb_cqi (Map)

    Args:
        lte_grid: Ресурсная сетка LTE
        bs: Базовая станция
        max_dl_ue_tti: Максимум UE за TTI (default: None = без ограничений)
        pcfich: PCFICH value (default: 2)
        max_dl_cce_allowance: Лимит CCE (default: None)
        verbose_pdcch: Отладочный вывод для PDCCH (default: False)
        enable_window: Включить скользящее окно (default: True)
        window_size: Размер окна в TTI (default: 100)
        verbose: Отладочный вывод для планировщика (default: False)
    """

    def __init__(self, lte_grid, bs, **kwargs):
        """Инициализация FD_BestCQI scheduler."""
        super().__init__(
            lte_grid=lte_grid,
            bs=bs,
            **kwargs)

    def _apply_pdsch_estimation(self, priority_list: List[Dict], tti: int) -> List[Dict]:
        """FD: estimation не нужен. Per-RBG выбор сам регулирует распределение."""
        if self.verbose:
            print(f"[SCHEDULER TTI {tti}] PDSCH estimation: SKIPPED (FD per-RBG selection)")
            print(f"[SCHEDULER TTI {tti}] After estimation: {len(priority_list)} UE selected, 0 UE excluded")
        return priority_list

    def _calculate_priorities(self, eligible_ues: List[Dict], tti: int) -> List[Dict]:
        """
        Args:
            eligible_ues: Отфильтрованные UE
            tti: Текущий TTI

        Returns:
            List[Dict]: UE с добавленным полем 'priority'
        """
        for user in eligible_ues:
            ue_id = user['UE_ID']
            user['priority'] = self._get_wb_cqi(ue_id)  # Priority = CQI

        if self.verbose:
            avg_priority = sum(u['priority'] for u in eligible_ues) / len(eligible_ues)
            print(f"[SCHEDULER.FD_BestCQI TTI {tti}] Priority calculation: {len(eligible_ues)} UE, avg priority={avg_priority:.2f}")

        return eligible_ues

    def _allocate_pdsch(self, tti: int, ues_with_pdcch: List[Dict],
                       eligible_ues: List[Dict]) -> Dict[int, List[int]]:
        """
        Greedy per-RBG:
        для каждого RBG выбираем UE с максимальным SB CQI на этом RBG
        (если SB CQI отсутствует — fallback на WB CQI).
        Tie-break: оставляем первого в списке (он уже отсортирован по priority)
        """
        allocation = {user['UE_ID']: [] for user in eligible_ues}

        remaining_buffer = {
            user['UE_ID']: user.get('bs_buffer_size', 0) * 8
            for user in ues_with_pdcch
        }

        rbg_size = self.lte_grid.GET_RBG_SIZE()
        total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size

        # Greedy per-RBG allocation
        for rbg_idx in range(total_rbg):
            if all(buf <= 0 for buf in remaining_buffer.values()):
                break

            rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
            rbg_width  = len(rb_indices)

            best_user   = None
            best_cqi    = -1

            for user in ues_with_pdcch:
                ue_id = user['UE_ID']
                if remaining_buffer.get(ue_id, 0) <= 0:
                    continue

                # Получаем SB CQI для этого RBG (с fallback на WB)
                sb_cqi_list = self._get_sb_cqi(ue_id)

                if sb_cqi_list and rbg_idx < len(sb_cqi_list):
                    cqi = sb_cqi_list[rbg_idx] if sb_cqi_list[rbg_idx] > 0 else self._get_wb_cqi(ue_id)
                else:
                    cqi = self._get_wb_cqi(ue_id)

                if cqi <= 0:
                    continue

                if cqi > best_cqi:
                    best_cqi    = cqi
                    best_user   = user

            if best_user is None:
                continue

            ue_id = best_user['UE_ID']

            if not self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                continue
            allocation[ue_id].extend(rb_indices)
            bits_per_rb = self.amc.GET_BITS_PER_RB(best_cqi)

            rbg_capacity = rbg_width * bits_per_rb
            remaining_buffer[ue_id] = max(0, remaining_buffer[ue_id] - rbg_capacity)

        if self.verbose:
            allocated_ues = sum(1 for rbs in allocation.values() if len(rbs) > 0)
            total_rb = sum(len(rbs) for rbs in allocation.values())
            print(f"SCHEDULER.FD_BestCQI [TTI {tti}] PDSCH: "
                  f"{allocated_ues} UE, {total_rb} RB")

        return allocation

class FDxFairGreedyScheduler(SchedulerInterface):
    """
    Channel-Aware Per-round Greedy FD Scheduler
    Объединяет долгосрочную справедливость с частотной селективностью
    Формирует очередь в TD из обслуживаемых UE, затем последовательно
    выделяет RBG UE с лучшим CQI для текущего RBG, убирая его из очереди
    и двигаясь дальше по RBG.
    Очередь сохраняется во времени.

    Алгоритм:
      TD-фаза (calculate_priorities):
        Ротация через rr_ue_offset — определяет порядок очереди и tie-break.

      FD-фаза (_allocate_pdsch):
        Для каждого RBG:
          1. Из текущего раунда выбираем UE с лучшим SB CQI на этом RBG
          2. Tie-break: при равном CQI побеждает тот, кто раньше в RR-очереди
          3. Победитель удаляется из текущего раунда
          4. Раунд опустел → перезапуск из всех UE с ненулевым буфером
        После TTI: rr_ue_offset += 1 → в следующем TTI другой UE первый в очереди

    Args:
        lte_grid: Ресурсная сетка LTE
        bs: Базовая станция
        max_dl_ue_tti: Максимум UE за TTI (default: None = без ограничений)
        pcfich: PCFICH value (default: 2)
        max_dl_cce_allowance: Лимит CCE (default: None)
        verbose_pdcch: Отладочный вывод для PDCCH (default: False)
        enable_window: Включить скользящее окно (default: True)
        window_size: Размер окна в TTI (default: 100)
        verbose: Отладочный вывод для планировщика (default: False)
    """

    def __init__(self, lte_grid, bs, **kwargs):
        super().__init__(lte_grid, bs, **kwargs)
        self.rr_rbg_offset = 0 #можно сделать механизм без ротации RBG
        self.rr_ue_offset = 0

    def _apply_pdsch_estimation(self, priority_list: List[Dict], tti: int) -> List[Dict]:
        """
        RR: Не применяем PDSCH estimation. FD-фаза автоматически управляет
        распределением по RBG без предварительного отсева UE.
        Args:
            priority_list: UE для планирования
            tti: Текущий TTI

        Returns:
            List[Dict]: Тот же priority_list без изменений
        """
        if self.verbose:
            print(f"[SCHEDULER TTI {tti}] PDSCH estimation: SKIPPED (FD-FGS per-RBG selection)")
            print(f"[SCHEDULER TTI {tti}] After estimation: {len(priority_list)} UE selected, 0 UE excluded")

        return priority_list

    def _calculate_priorities(self, windowed_ues: List[Dict], tti: int) -> List[Dict]:
        """
        Round-ротация: назначаем нисходящие приоритеты начиная с rr_ue_offset.

        Приоритет используется в form_priority_list для сортировки, что даёт
        base_queue в _allocate_pdsch уже в правильном RR-порядке.
        Tie-break в FD-фазе работает автоматически через этот порядок.
        """
        num_ues = len(windowed_ues)
        if num_ues == 0:
            return windowed_ues

        if self.max_dl_ue_tti and self.max_dl_ue_tti < num_ues:
            ues_to_plan = self.max_dl_ue_tti
        else:
            ues_to_plan = num_ues

        start_idx   = self.rr_ue_offset % num_ues
        rotated_ues = windowed_ues[start_idx:] + windowed_ues[:start_idx]

        for idx, user in enumerate(rotated_ues):
            user['priority'] = ues_to_plan - idx

        if self.verbose:
            top_ue = rotated_ues[0]['UE_ID']
            print(f"[SCHEDULER.FDxFGS TTI {tti}] Priority: Rotated (start UE {top_ue}, offset={self.rr_ue_offset})")

        return rotated_ues

    def _allocate_pdsch(self, tti: int, ues_with_pdcch: List[Dict],
                       eligible_ues: List[Dict]) -> Dict[int, List[int]]:
        """
        Greedy per-RBG:
        для каждого RBG выбираем UE с максимальным SB CQI на этом RBG
        (если SB CQI отсутствует — fallback на WB CQI).
        Tie-break: оставляем первого в списке (он уже отсортирован по priority)
        """
        allocation  = {user['UE_ID']: [] for user in eligible_ues}
        rbg_size    = self.lte_grid.GET_RBG_SIZE()
        total_rbg   = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size

        num_ues = len(ues_with_pdcch)
        if num_ues == 0:
            return allocation

        eligible_id_map = {ue['UE_ID']: idx for idx, ue in enumerate(eligible_ues)}
        ue_index        = self.rr_rbg_offset
        last_served_idx = -1

        remaining_buffer = {
            user['UE_ID']: user.get('bs_buffer_size', 0) * 8
            for user in ues_with_pdcch
        }

        served_in_round = set()

        for rbg_idx in range(total_rbg):

            if all(buf <= 0 for buf in remaining_buffer.values()):
                break

            rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
            rbg_width  = len(rb_indices)

            for _ in range(2):
                best_user = None
                best_cqi  = -1

                for k in range(num_ues):
                    ue    = ues_with_pdcch[(ue_index + k) % num_ues]
                    ue_id = ue['UE_ID']

                    if remaining_buffer[ue_id] <= 0:
                        continue
                    if ue_id in served_in_round:
                        continue

                    sb = self._get_sb_cqi(ue_id)
                    if sb and rbg_idx < len(sb):
                        cqi = sb[rbg_idx] if sb[rbg_idx] > 0 else self._get_wb_cqi(ue_id)
                    else:
                        cqi = self._get_wb_cqi(ue_id)

                    if cqi <= 0:
                        continue

                    if cqi > best_cqi:
                        best_cqi  = cqi
                        best_user = ue

                if best_user is not None:
                    break

                if served_in_round:
                    served_in_round.clear()
                else:
                    break

            if best_user is None:
                continue

            ue_id = best_user['UE_ID']

            if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                allocation[ue_id].extend(rb_indices)

                bits_per_rb     = self.amc.GET_BITS_PER_RB(best_cqi)
                rbg_capacity    = rbg_width * bits_per_rb

                remaining_buffer[ue_id] = max(0, remaining_buffer[ue_id] - rbg_capacity)

                served_in_round.add(ue_id)
                ue_index += 1

                # RR bookkeeping как в RR: last_served_idx для rr_ue_offset
                current_idx = eligible_id_map.get(ue_id, -1)
                last_served_idx = current_idx

        self.rr_rbg_offset = ue_index % num_ues
        if last_served_idx >= 0 and len(eligible_ues) > 0:
            self.rr_ue_offset = (last_served_idx + 1) % len(eligible_ues)

        if self.verbose:
            allocated_ues = sum(1 for rbs in allocation.values() if len(rbs) > 0)
            total_rb      = sum(len(rbs) for rbs in allocation.values())
            print(f"[SCHEDULER.FDхFGS TTI {tti}] PDSCH: {allocated_ues} UE, {total_rb} RB total "
                  f"(next_offset={self.rr_rbg_offset})")

        return allocation

class FDxProportionalFairScheduler(SchedulerInterface):
    """
    Frequency Domain Proportional Fair (FD-PF) Scheduler.

    Реализует классическую PF-метрику в частотной плоскости (FD):
    победителем каждого RBG становится UE с максимальным отношением
    мгновенной достижимой скорости к исторической средней пропускной способности.

    Алгоритм (три этапа в рамках одного TTI):
        1. _calculate_priorities:
               Вычисляет WB PF-метрику для упорядочивания PDCCH-очереди.
               priority_j = R_j(WB, t) / T_j(t)
               Используется только в form_priority_list и allocate_pdcch —
               определяет порядок выдачи CCE при их нехватке.

        2. _apply_pdsch_estimation:
               Пропускается. FD per-RBG цикл самостоятельно контролирует
               использование RB — WB-оценка ёмкости некорректна для FD.

        3. _allocate_pdsch:
               Per-RBG FD-PF выбор победителя по SB CQI.
               Для каждого RBG k:
                   j* = argmax_j [ R_j(k, t) / T_j(t) ]
               R_j(k, t) вычисляется по sb_cqi[k] через AMC,
               fallback на WB CQI при отсутствии SB-отчёта.
               T_j(t) фиксирована на весь TTI — пересчёт только между TTI.

    Args:
        ltegrid           : Экземпляр LTE Grid.
        bs                : Экземпляр Base Station.
        maxdluetti        : Максимум UE на один TTI. По умолчанию None.
        pcfich            : Значение PCFICH (1, 2 или 3). По умолчанию 2.
        maxdlcceallowance : Ограничение CCE для DL UE-specific PDCCH.
                            По умолчанию None (= totalcce).
        verbosepdcch      : Verbose-логирование PDCCHManager. По умолчанию False.
        enablewindow      : Включить sliding window. По умолчанию True.
        windowsize        : Размер окна планирования в TTI. По умолчанию 100.
        verbose           : Verbose-логирование планировщика. По умолчанию False.
    """

    def __init__(self, lte_grid, bs, **kwargs):
        """Инициализация FD_BestCQI scheduler."""
        super().__init__(
            lte_grid=lte_grid,
            bs=bs,
            **kwargs)

    def _apply_pdsch_estimation(self, priority_list: List[Dict], tti: int) -> List[Dict]:
        """
        RR: Не применяем PDSCH estimation. FD-фаза автоматически управляет
        распределением по RBG без предварительного отсева UE.
        Args:
            priority_list: UE для планирования
            tti: Текущий TTI

        Returns:
            List[Dict]: Тот же priority_list без изменений
        """
        if self.verbose:
            print(f"[SCHEDULER TTI {tti}] PDSCH estimation: SKIPPED (FD-PF per-RBG selection)")
            print(f"[SCHEDULER TTI {tti}] After estimation: {len(priority_list)} UE selected, 0 UE excluded")

        return priority_list

    def _calculate_priorities(self, windowed_ues: List[Dict], tti: int) -> List[Dict]:
        """
        Args:
            windowed_ues : UE после filter_by_window (window_size уже применён
                           базовым классом до вызова этого метода).
            tti          : Текущий TTI.

        Returns:
            List[Dict]: windowed_ues с заполненными 'priority' и 'instant_rate'.
        """
        for user in windowed_ues:
            ue_id        = user['UE_ID']
            cqi          = self._get_wb_cqi(ue_id)
            bits_per_rb  = self.amc.GET_BITS_PER_RB(cqi)
            instant_rate = self.lte_grid.rb_per_slot * bits_per_rb

            avg_throughput = user['ue'].average_throughput

            if avg_throughput <= 0:
                avg_throughput = 1e-6
                pf_metric = instant_rate / 1.0
            else:
                avg_throughput_per_tti = avg_throughput / 1000
                pf_metric = instant_rate / avg_throughput_per_tti

            user['priority']     = pf_metric
            user['instant_rate'] = instant_rate

            if self.verbose:
                print(f"SCHEDULER.FDxProportionalFair TTI {tti}: "
                      f"UE {ue_id} | CQI={cqi} | "
                      f"InstRate={instant_rate} bits/TTI | "
                      f"AvgTput={avg_throughput:.1f} bps | "
                      f"PF={pf_metric:.4f}")

        if self.verbose:
            avg_pf = (sum(u['priority'] for u in windowed_ues) / len(windowed_ues)
                      if windowed_ues else 0.0)
            print(f"SCHEDULER.FDxProportionalFair TTI {tti}: "
                  f"Priority calc {len(windowed_ues)} UE, avg PF={avg_pf:.4f}")

        return windowed_ues

    def _allocate_pdsch(self, tti: int,
                        ues_with_pdcch: List[Dict],
                        eligible_ues: List[Dict]):

        allocation = {u['UE_ID']: [] for u in eligible_ues}

        if not ues_with_pdcch:
            return allocation

        remaining_buffer = {u['UE_ID']: u.get('bs_buffer_size', 0) * 8 for u in ues_with_pdcch}

        rbg_size  = self.lte_grid.GET_RBG_SIZE()
        total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size

        for rbg_idx in range(total_rbg):
            if all(buf <= 0 for buf in remaining_buffer.values()):
                break

            rb_indices      = self.lte_grid.GET_RBG_INDICES(rbg_idx)
            rbg_width       = len(rb_indices)
            best_user       = None
            best_metric     = -1.0
            best_rbg_bits   = 0
            best_rb_indices: List[int] = []

            for user in ues_with_pdcch:
                ue_id = user['UE_ID']
                if remaining_buffer[ue_id] <= 0:
                    continue

                sb_cqi_list = self._get_sb_cqi(ue_id)
                if (sb_cqi_list and rbg_idx < len(sb_cqi_list)
                    and sb_cqi_list[rbg_idx] > 0
                ):
                    cqi = sb_cqi_list[rbg_idx]
                else:
                    cqi = self._get_wb_cqi(ue_id)

                if cqi <= 0:
                    continue

                bits_per_rb = self.amc.GET_BITS_PER_RB(cqi)
                r_j_k       = rbg_width * bits_per_rb

                avg_tput = user['ue'].average_throughput
                denom    = 1.0 if avg_tput <= 0 else (avg_tput / 1000.0)
                metric   = r_j_k / denom

                if metric > best_metric:
                    best_metric     = metric
                    best_user       = user
                    best_rbg_bits   = r_j_k
                    best_rb_indices = rb_indices

            if best_user is None:
                continue

            best_ueid = best_user['UE_ID']
            ok = self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, best_ueid)
            if not ok:
                continue

            allocation[best_ueid].extend(best_rb_indices)
            remaining_buffer[best_ueid] = max(0, remaining_buffer[best_ueid] - best_rbg_bits)

        if self.verbose:
            allocated_ues = sum(1 for rbs in allocation.values() if len(rbs) > 0)
            total_rb = sum(len(rbs) for rbs in allocation.values())
            print(f"SCHEDULER.FDxProportionalFair TTI {tti} PDSCH "
                  f"{allocated_ues} UE, {total_rb} RB total")

        return allocation

#TODO: Финальный аккорд модуля. Новая архитектура готова. Теперь можно подумать
# о развитии. Хочу отметить, что требуется еще много доработок. Вот список:
#   1) Разобраться с buffer_processing. Выпилить его из планировщика. ЗАЧЕМ?
#   - прежде всего, оптимизация. Обрабатывать буфер это не задача scheduler;
#   - неизбежно при работе с HARQ;
#   - потому что так правильно.
#   РЕШЕНИЕ: BufferManager
#   2) Избавиться от связи с RES_GRID.
#   - это кушает ресурсы планировщика;
#   - потому что так правильно.
#   РЕШЕНИЕ: реализовать DCI и AllocationTracker
#   3) Каноничный результат работы планировщика (output) и DCI
#   - нет нормального сбора статистики;
#   - нет нормального DCI;
#   - нет важных выводов что есть в реальном планировщике.
#   РЕШЕНИЕ: будет.
#   4) QoS-aware планировщики
#   5) Исследование механизмов tie-brake когда метрики приоритетов совпадают
#   6) Исследование механизма распределения RBG, венгерский, жадный и др.
#   7) Исследование механизма оценки средней пропускной способности для PF
#   8) Исследование механизмов защиты голодающих UE
