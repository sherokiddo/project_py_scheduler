"""
#------------------------------------------------------------------------------
# Модуль: SCHEDULER - Планировщик ресурсов для сети LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет классы и методы для распределения ресурсных блоков между
# пользовательскими устройствами (UE) в сети LTE. Реализует алгоритмы
# планирования Round Robin, Best CQI и Proportional Fair с поддержкой PDCCH.
#
# Версия: 1.0.8
# Дата последнего изменения: 2025-10-16
# Версия Python Kernel: 3.12.9
# Автор: Брагин Кирилл, Норицин Иван
#
# Зависимости:
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
# - Реализация основана на LTE Release 15 Scheduler Design Document (Nokia, Section 3.5).
# - Добавлен метод release_cce() в PDCCHManager как заглушка для будущей разработки
#   (освобождение CCE при неудаче PDSCH allocation).
#------------------------------------------------------------------------------
"""

from typing import Dict, List, Optional, Union, Tuple
from RES_GRID import RES_GRID_LTE, SchedulerInterface
from BS_MODULE import BaseStation

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
            max_dl_cce_allowance: Максимум CCE для DL UE-specific (dlNumCceAllowance)
                                  None = использовать весь Total_CCE
            verbose: режим детального логирования для дебага
        Raises:
            ValueError: при некорректных параметрах
        """
        
        self._validate_parameters(bandwidth, pcfich)
        
        self.bandwidth = bandwidth
        self.pcfich = pcfich
        self.verbose = verbose
        
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
        self.cce_allocations = {}    # {ue_id: cce_count}
        self.history = []
        
        # для дебага
        if self.verbose:
            print(
                f"[PDCCH] Initialized: Bandwidth={bandwidth}RB, PCFICH={pcfich}, "
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
            (1.4, 1): 2,
            (1.4, 2): 4,
            (1.4, 3): 6,
            
            # 3 MHz (15 RB)
            (3, 1): 5,
            (3, 2): 7,
            (3, 3): 12,
            
            # 5 MHz (25 RB)
            (5, 1): 8,
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
            print(f"[PDCCH] CQI={cqi} → Aggregation Level={aggregation_level} CCE")
        
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
        
        is_available = required_cce <= available_cce
        
        if self.verbose:
            if is_available:
                print(f"[PDCCH] Check: {required_cce} CCE requested, "
                      f"{available_cce} available → ✅ ALLOCATED")
            else:
                print(f"[PDCCH] Check: {required_cce} CCE requested, "
                      f"{available_cce} available → ❌ INSUFFICIENT")
        
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
        self.num_assigned_cce += cce_count
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
        available_cce = self.max_dl_cce - self.num_assigned_cce
        
        if self.max_dl_cce > 0:
            utilization = self.num_assigned_cce / self.max_dl_cce
        else:
            utilization = 0.0
        
        stats = {
            'total_cce': self.total_cce,
            'max_dl_cce': self.max_dl_cce,
            'used_cce': self.num_assigned_cce,
            'available_cce': available_cce,
            'utilization': utilization,
            'utilization_percent': utilization * 100,
            'num_scheduled_users': len(self.cce_allocations),
            'allocations': self.cce_allocations.copy()  # Копия, не оригинал
        }
        
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

    def GET_BITS_PER_RB(self, cqi: int) -> int:
        """
        Рассчитать количество бит на ресурсный блок (RB) для заданного CQI.
        """
        if cqi not in self.CQI_TO_MCS:
            raise ValueError(f"Invalid CQI: {cqi}. Must be 1-15.")
        
        modulation, code_rate = self.CQI_TO_MCS[cqi]
        symbols_per_rb = 12 * 7  # 84 символа в RB (с учетом слотов)
        return int(symbols_per_rb * modulation * code_rate)
	#@sherokiddo: "Добавить зависимость от CP"
    
    def calculate_throughput(self, allocation: Dict, users: List[Dict], tti: int, bs: BaseStation) -> Dict:
        """
        Рассчитывает фактическую пропускную способность с учетом:
        - Реальных данных из буфера
        - Адаптивной модуляции и кодирования (AMC)
    
        Args:
            allocation: Словарь распределения RB {UE_ID: список freq_indices}
            users: Список активных пользователей с параметрами CQI
    
        Returns:
            Словарь с метриками производительности:
            - total_allocated_rbs: Общее количество выделенных RB
            - user_throughput: Пропускная способность на пользователя (бит/с)
            - average_throughput: Средняя пропускная способность (бит/с)
            - total_effective_bits: Фактически переданные биты
        """
        all_users = {u['UE_ID']: u for u in users}
        stats = {
            'total_allocated_rbs': 0,
            'user_throughput': {ue_id: 0 for ue_id in all_users}, 
            'user_effective_throughput': {ue_id: 0 for ue_id in all_users},
            'user_max_throughput': {ue_id: 0 for ue_id in all_users},
            'average_dl_throughput': 0.0,
            'total_effective_bits': 0
        }
        
        total_effective_bits = 0
    
        for ue_id in all_users:
            user = all_users[ue_id]
            rb_count = len(allocation.get(ue_id, [])) * 2
            
            if rb_count == 0:
                # Обновление истории для неактивных пользователей
                user['ue'].UPD_DL_THROUGHPUT(0, 1)
                continue
    
            # 1. Расчет максимальной ёмкости RB для данного CQI
            bits_per_rb = self.GET_BITS_PER_RB(user['cqi'])

            # 2. Расчет эффективно использованных бит
            max_bits = rb_count * bits_per_rb
            
            # 3. Реальные переданные биты из буфера
            real_bits = user['ue'].current_dl_throughput
            effective_bits = min(real_bits, max_bits)
            
            # 4. Обновление статистики
            stats['user_throughput'][ue_id] = effective_bits
            total_effective_bits += effective_bits
            stats['total_allocated_rbs'] += rb_count
            stats['user_effective_throughput'][ue_id] = effective_bits
            stats['user_max_throughput'][ue_id] = max_bits
    
        # 5. Расчет средней пропускной способности
        active_users = [u for u in all_users.values() if stats['user_throughput'][u['UE_ID']] > 0]
        if active_users:
            stats['average_dl_throughput'] = total_effective_bits / len(active_users)
        
        stats['total_effective_bits'] = total_effective_bits
        return stats
    
#==============================================================================
#                              ЛОГИКА SCHEDULER
#==============================================================================

class RoundRobinScheduler(SchedulerInterface):
    
    def __init__(self, lte_grid: RES_GRID_LTE, bs: BaseStation, 
                 max_dl_ue_tti: Optional[int] = None,
                 pcfich: int = 2,
                 max_dl_cce_allowance: Optional[int] = None,
                 verbose_pdcch: bool = False):
        super().__init__(lte_grid, max_dl_ue_tti)
        self.lte_grid = lte_grid
        self.lte_grid.SET_BS(bs)
        self.last_served_ue_id = None 
        #теперь планировщик знает предыдущего обслуженного в tti прользователя
        #именно через этот метод
        self.amc = AdaptiveModulationAndCoding()
        
        # добавлено. инициализация PDCCH Manager
        self.pdcch_manager = PDCCHManager(
            bandwidth = self.lte_grid.bandwidth,
            pcfich = pcfich,
            max_dl_cce_allowance = max_dl_cce_allowance,
            verbose = verbose_pdcch)
        
    def schedule(self, tti: int, users: List[Dict]) -> Dict:
            """
            Планирование ресурсов с учётом данных в буфере и CQI.
            
            Args:
                tti: Индекс TTI
                users: Список пользователей с параметрами:
                    - 'UE_ID': Идентификатор
                    - 'buffer_size': Размер буфера в байтах
                    - 'cqi': Индекс качества канала
                    - 'ue': Объект UserEquipment
            
            Returns:
                Dict: Результаты распределения ресурсов
            """
            for user in users:
                user['ue'].current_dl_throughput = 0
            
            # 1. Фильтрация активных пользователей по буферу BS
            active_users = []
            for user in users:
                ue_id = user['UE_ID']
                bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
                if not bs_buffer:
                    continue
                    
                buffer_status = bs_buffer.GET_UE_STATUS(tti)['per_ue'].get(ue_id, {})
                buffer_size = buffer_status.get('size', 0)
                
                if buffer_size > 0 and 1 <= user['cqi'] <= 15:
                    active_users.append(user)
                    user['bs_buffer_size'] = buffer_size
            
            if not active_users:
                return {'allocation': {}, 
                        'statistics': {}, 
                        'bitmap': {}, 
                        'pdcch_stats': self.pdcch_manager.get_stats()}
        
        
            # 2. Определение стартового индекса планирования
            if self.last_served_ue_id is None:
                # Первый запуск, начинаем с 0
                start_idx = 0
            else:
                # Ищем последнего обслуженного пользователя в новом списке активных
                found_idx = -1
                for i, user in enumerate(active_users):
                    if user['UE_ID'] == self.last_served_ue_id:
                        found_idx = i
                        break
                        
                # Начинаем со следующего пользователя после последнего обслуженного
                if found_idx == -1:
                    # Если не нашли пользователя, начинаем с 0
                    start_idx = 0
                else:
                    # Если нашли, берем следующего
                    start_idx = (found_idx + 1) % len(active_users)

            # 3. Создание списка пользователей для планирования с учетом лимитов
            if self.max_dl_ue_tti is not None:
                scheduled_count = min(self.max_dl_ue_tti, len(active_users))
                scheduled_users = []
                idx = start_idx
                for _ in range(scheduled_count):
                    scheduled_users.append(active_users[idx])
                    idx = (idx + 1) % len(active_users)
            else:
                scheduled_users = active_users

            # 4. Обработка выделения ресурсов PDCCH.
            #TODO: Вынести логику планирования PDCCH в отедльный метод
            
            self.pdcch_manager.reset_tti()
            scheduled_users_with_pdcch = []
            
            for user in scheduled_users:
                cqi = user['cqi']
                ue_id = user['UE_ID']
                
                required_cce = self.pdcch_manager.get_aggregation_level(cqi)
                
                if self.pdcch_manager.allocate_cce(ue_id, required_cce):
                    scheduled_users_with_pdcch.append(user)
                    user['allocated_cce'] = required_cce
                    # успешное выделение
                else:
                    continue
                
            scheduled_users = scheduled_users_with_pdcch
            
            if not scheduled_users:
                return {'allocation': {}, 
                        'statistics': {}, 
                        'bitmap': {}, 
                        'pdcch_stats': self.pdcch_manager.get_stats()}
            
            # 5. Расчет параметров планирования и инициализация структур
            rbg_size = self.lte_grid.GET_RBG_SIZE()
            total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size            

            allocation = {user['UE_ID']: [] for user in active_users}
            remaining_buffer = {user['UE_ID']: user['bs_buffer_size'] * 8 for user in scheduled_users} 
            last_allocated_ue_id = None
            current_idx = 0

            # 6. Основной цикл распределения по RBG
            for rbg_idx in range(total_rbg):
                if all(v <= 0 for v in remaining_buffer.values()):
                    break
        
                # Поиск следующего пользователя с данными
                initial_idx = current_idx
                while remaining_buffer[scheduled_users[current_idx]['UE_ID']] <= 0:
                    current_idx = (current_idx + 1) % len(scheduled_users)
                    if current_idx == initial_idx:
                        break
                
                # Получаем текущего пользователя и выделяем RBG
                user = scheduled_users[current_idx]
                ue_id = user['UE_ID']
                
                # Цикл распределения RBG только если в буфере еще есть данные
                if remaining_buffer[ue_id] > 0:
                    if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                        # Получаем индексы ресурсных блоков в группе
                        rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
                        allocation[ue_id].extend(rb_indices)
                        last_allocated_ue_id = ue_id
                    
                # Уменьшает размер буфера в соотв. с емкостью RBG
                        bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
                        rbg_capacity = len(rb_indices) * bits_per_rb * 2
                        remaining_buffer[ue_id] -= min(remaining_buffer[ue_id], rbg_capacity)

                # 7. Переход к следующему пользователю
                current_idx = (current_idx + 1) % len(scheduled_users)
                
            # 8. Обновление индекса последнего обслуженного UE
            if last_allocated_ue_id is not None:
                self.last_served_ue_id = last_allocated_ue_id
                
            # 9. Обработка буфера и статистики
            for user in users:
                ue = user['ue']
                ue_id = user['UE_ID'] #да, эта часть кода странная, но только после этого все заработало
                bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)                
                if not bs_buffer:
                    continue
                
                allocated_rb = len(allocation.get(user['UE_ID'], [])) * 2
                bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
                max_bytes = (allocated_rb * bits_per_rb) // 8  #rbg_capacity // 8??

                #Если пользователь неактивен, передаём 0 байт
                packets, total = bs_buffer.GET_PACKETS(
                    ue_id=ue_id,
                    max_bytes=max_bytes,
                    bits_per_rb=bits_per_rb,
                    current_time=tti
                )               
            # Обновление метрик DL
                ue.UPD_DL_THROUGHPUT(total * 8, 1)                
    
        # 10. Формирование bitmap
            bitmap = {user['UE_ID']: self.lte_grid.GENERATE_BITMAP(tti, user['UE_ID']) for user in active_users}
            pdcch_stats = self.pdcch_manager.get_stats()
            
            return {
                'allocation': allocation,
                'statistics': self.amc.calculate_throughput(allocation, users, tti, self.lte_grid.bs),
                'bitmap': bitmap,
                'pdcch_stats': pdcch_stats #,
                #'dl_throughput': {user['UE_ID']: user['ue'].current_dl_throughput for user in users}
            }
        
    #@sherokiddo в рамках оптимизации можно разбить весь планировщик на несколько методов
    #для удобного логирования и подсчета времени. Например - подготовка данных один метод
    #затем идет непосредственно все планирование, и метод формирования статистики

class BestCQIScheduler(SchedulerInterface):
    
    def __init__(self, lte_grid: RES_GRID_LTE, bs: BaseStation, 
                 max_dl_ue_tti: Optional[int] = None,
                 pcfich: int = 2,
                 max_dl_cce_allowance: Optional[int] = None,
                 verbose_pdcch: bool = False):

        super().__init__(lte_grid, max_dl_ue_tti)
        self.lte_grid = lte_grid
        self.lte_grid.SET_BS(bs)
        self.amc = AdaptiveModulationAndCoding() 

        self.pdcch_manager = PDCCHManager(
            bandwidth=self.lte_grid.bandwidth,
            pcfich=pcfich,
            max_dl_cce_allowance=max_dl_cce_allowance,
            verbose=verbose_pdcch
        )
    
    def schedule(self, tti: int, users: List[Dict]) -> Dict:
        """
        Планирование ресурсов с учётом данных в буфере и CQI.
        
        Args:
            tti: Индекс TTI
            users: Список пользователей с параметрами:
                - 'UE_ID': Идентификатор
                - 'buffer_size': Размер буфера в байтах
                - 'cqi': Индекс качества канала
                - 'ue': Объект UserEquipment
        
        Returns:
            Dict: Результаты распределения ресурсов
        """
        for user in users:
            user['ue'].current_dl_throughput = 0
        
        # 1. Фильтрация активных пользователей
        active_users = []
        for user in users:
            ue_id = user['UE_ID']
            bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
            if not bs_buffer:
                continue
            
            buffer_status = bs_buffer.GET_UE_STATUS(tti)['per_ue'].get(ue_id, {})
            buffer_size = buffer_status.get('size', 0)
            
            if buffer_size > 0 and 1 <= user['cqi'] <= 15:
                active_users.append(user)
                user['bs_buffer_size'] = buffer_size

        if not active_users:
            return {'allocation': {},
                    'statistics': {},
                    'bitmap': {},
                    'pdcch_stats': self.pdcch_manager.get_stats()}

        # 2. Сортируем список по CQI
        active_users.sort(key=lambda u: u['cqi'], reverse=True)

        # 3. Создание списка пользователей для планирования с учетом лимитов
        if self.max_dl_ue_tti is not None:
            scheduled_count = min(self.max_dl_ue_tti, len(active_users))
            scheduled_users = active_users[:scheduled_count]
        else:
            scheduled_users = active_users

        # 4. Расчет параметров планирования и инициализация структур
        self.pdcch_manager.reset_tti()
        rbg_size = self.lte_grid.GET_RBG_SIZE()
        total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size

        allocation = {user['UE_ID']: [] for user in active_users}
        remaining_buffer = {user['UE_ID']: user['bs_buffer_size'] * 8 for user in scheduled_users}
        
        pdcch_allocated_ues = set()

        # 5. Основной цикл распределения RBG
        for rbg_idx in range(total_rbg):
            if all(v <= 0 for v in remaining_buffer.values()):
                break
            
            # фильтрация пользователей с данными в буфере (удалена лишняя сортировка по CQI)
            best_user = None
            for user in scheduled_users:
                if remaining_buffer[user['UE_ID']] > 0:
                    best_user = user
                    break
            
            if best_user is None:
                break

            ue_id = best_user['UE_ID']
            
            if ue_id not in pdcch_allocated_ues:
                required_cce = self.pdcch_manager.get_aggregation_level(best_user['cqi'])
                
                # Попытка выделить CCE
                if not self.pdcch_manager.allocate_cce(ue_id, required_cce):
                    # PDCCH недоступен → блокируем этого UE
                    # Обнуляем буфер чтобы не выбирать его снова
                    remaining_buffer[ue_id] = 0
                    continue
                
                # PDCCH успешно выделен → добавляем в трекинг
                pdcch_allocated_ues.add(ue_id)
                best_user['allocated_cce'] = required_cce  # Для статистики            
            
            # выделение RBG
            if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
                allocation[ue_id].extend(rb_indices)
                
                # Уменьшаем размер буфера в соотв. с емкостью RBG
                bits_per_rb = self.amc.GET_BITS_PER_RB(best_user['cqi'])
                rbg_capacity = len(rb_indices) * bits_per_rb * 2
                remaining_buffer[ue_id] -= min(remaining_buffer[ue_id], rbg_capacity)
   
        # 6. Обработка буфера и статистики
        for user in users:
            ue = user['ue']
            ue_id = user['UE_ID']
            bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
            if not bs_buffer:
                continue
            
            allocated_rb = len(allocation.get(user['UE_ID'], [])) * 2
            bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
            max_bytes = (allocated_rb * bits_per_rb) // 8
            
            packets, total = bs_buffer.GET_PACKETS(
                ue_id=ue_id,
                max_bytes=max_bytes,
                bits_per_rb=bits_per_rb,
                current_time=tti
            )
            
            # Обновление метрик DL
            ue.UPD_DL_THROUGHPUT(total * 8, 1)
   
        # 7. Формирование bitmap
        bitmap = {user['UE_ID']: self.lte_grid.GENERATE_BITMAP(tti, user['UE_ID']) for user in active_users}
        pdcch_stats = self.pdcch_manager.get_stats()
        
        return {
            'allocation': allocation,
            'statistics': self.amc.calculate_throughput(allocation, users, tti, self.lte_grid.bs),
            'bitmap': bitmap,
            'pdcch_stats': pdcch_stats#,
            #'dl_throughput': {user['UE_ID']: user['ue'].current_dl_throughput for user in users}
        }
    
    #@sherokiddo в рамках оптимизации можно разбить весь планировщик на несколько методов
    #для удобного логирования и подсчета времени. Например - подготовка данных один метод
    #затем идет непосредственно все планирование, и метод формирования статистики

class ProportionalFairScheduler(SchedulerInterface):
    
    def __init__(self, lte_grid: RES_GRID_LTE, bs: BaseStation, 
                 max_dl_ue_tti: Optional[int] = None,
                 pcfich: int = 2,
                 max_dl_cce_allowance: Optional[int] = None,
                 verbose_pdcch: bool = False):
        super().__init__(lte_grid, max_dl_ue_tti)
        self.lte_grid = lte_grid
        self.lte_grid.SET_BS(bs)
        self.amc = AdaptiveModulationAndCoding()
        
        self.pdcch_manager = PDCCHManager(
            bandwidth=self.lte_grid.bandwidth,
            pcfich=pcfich,
            max_dl_cce_allowance=max_dl_cce_allowance,
            verbose=verbose_pdcch)
        
    def calculate_pf_metric(self, user: List[Dict]):
        """
        Расчёт PF-метрики для каждого пользователя.
        
        Args:
            user: словарь с параметрами пользователя
        Returns:
            pf_metric: просто значение PF-метрики
        """
        bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
        rb_per_slot = self.lte_grid.rb_per_slot
        instant_throughput = rb_per_slot * bits_per_rb * 2 * 1000

        avg_throughput = user['ue'].average_throughput
        if avg_throughput <= 0:
            avg_throughput = 1e-6
            
        pf_metric = instant_throughput / avg_throughput

        return pf_metric
            
    def schedule(self, tti: int, users: List[Dict]) -> Dict:
        """
        Планирование ресурсов с учётом данных в буфере и CQI.
        
        Args:
            tti: Индекс TTI
            users: Список пользователей с параметрами:
                - 'UE_ID': Идентификатор
                - 'buffer_size': Размер буфера в байтах
                - 'cqi': Индекс качества канала
                - 'ue': Объект UserEquipment
        
        Returns:
            Dict: Результаты распределения ресурсов
        """
        for user in users:
            user['ue'].current_dl_throughput = 0
            
        # 1. Фильтрация активных пользователей по буферу
        active_users = []
        for user in users:
            ue_id = user['UE_ID']
            bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
            if not bs_buffer:
                continue
            buffer_status = bs_buffer.GET_UE_STATUS(tti)['per_ue'].get(ue_id, {})
            buffer_size = buffer_status.get('size', 0)
            if buffer_size > 0 and 1 <= user['cqi'] <= 15:
                active_users.append(user)
                user['bs_buffer_size'] = buffer_size

        if not active_users:
            return {'allocation': {},
                    'statistics': {},
                    'bitmap': {},
                    'pdcch_stats': self.pdcch_manager.get_stats()}
        
        # 2. Расчёт и сортировка UE согласно PF-метрики
        for user in active_users:
            user['pf_metric'] = self.calculate_pf_metric(user)
        active_users.sort(key=lambda u: u['pf_metric'], reverse=True)
        
        if self.max_dl_ue_tti is not None:
            scheduled_users =  active_users[:self.max_dl_ue_tti]
        else:
            scheduled_users = active_users
        
        # 3. Логика выделения PDCCH
        # Логика сделана так. PDCCH и PDSCH выделяются как
        # В продвинутых планировщиках - принцип "concurrently"
        self.pdcch_manager.reset_tti()

        rbg_size = self.lte_grid.GET_RBG_SIZE()
        total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size
        
        allocation = {user['UE_ID']: [] for user in active_users}
        remaining_buffer = {user['UE_ID']: user['bs_buffer_size'] * 8 for user in scheduled_users}
        
        pdcch_allocated_ues = set()
        
        # 4. Основной цикл распределения RBG
        for rbg_idx in range(total_rbg):
            if all(v <= 0 for v in remaining_buffer.values()):
                break
            
            # фильтрация пользователей с данными в буфере (по PF-метрике)
            best_user = None
            for user in scheduled_users:
                if remaining_buffer[user['UE_ID']] > 0:
                    best_user = user
                    break
            
            if best_user is None:
                break

            ue_id = best_user['UE_ID']
            
            if ue_id not in pdcch_allocated_ues:
                required_cce = self.pdcch_manager.get_aggregation_level(best_user['cqi'])
                
                if not self.pdcch_manager.allocate_cce(ue_id, required_cce):
                    remaining_buffer[ue_id] = 0
                    continue
            pdcch_allocated_ues.add(ue_id)
            best_user['allocated_cce'] = required_cce
            
            # 5. Выделение RBG
            if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
                allocation[ue_id].extend(rb_indices)
                
                # 6. Обновление буфера
                bits_per_rb = self.amc.GET_BITS_PER_RB(best_user['cqi'])
                rbg_capacity = len(rb_indices) * bits_per_rb * 2
                remaining_buffer[ue_id] -= min(remaining_buffer[ue_id], rbg_capacity)
        
        # 7. Обработка буфера и статистики
        for user in users:
            ue = user['ue']
            ue_id = user['UE_ID']
            bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
            if not bs_buffer:
                continue
            
            allocated_rb = len(allocation.get(user['UE_ID'], [])) * 2
            bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
            max_bytes = (allocated_rb * bits_per_rb) // 8
            packets, total = bs_buffer.GET_PACKETS(
                ue_id=ue_id,
                max_bytes=max_bytes,
                bits_per_rb=bits_per_rb,
                current_time=tti
            )
            
            # Обновление метрик DL
            ue.UPD_DL_THROUGHPUT(total * 8, 1)

            average_throughput_past = ue.average_throughput
            ue.average_throughput = (1 - 0.2) * average_throughput_past + 0.2 * ue.current_dl_throughput 

        # 11. Формирование bitmap по Resource Allocation 0
        bitmap = {user['UE_ID']: self.lte_grid.GENERATE_BITMAP(tti, user['UE_ID']) for user in active_users}
        pdcch_stats = self.pdcch_manager.get_stats()
        
        return {
            'allocation': allocation,
            'statistics': self.amc.calculate_throughput(allocation, users, tti, self.lte_grid.bs),
            'bitmap': bitmap,
            'pdcch_stats': pdcch_stats#,
            #'dl_throughput': {user['UE_ID']: user['ue'].current_dl_throughput for user in users}
        }    