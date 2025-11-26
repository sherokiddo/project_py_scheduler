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
# - Подготовлено к внедрнеию HARQ.
# - Составлен список планирумых фич для следующих версий.
# - Полный лист изменений и документация на текущий момент составляется.
#------------------------------------------------------------------------------
"""

from typing import Dict, List, Optional, Union, Tuple
from BS_MODULE import BaseStation
from enum import Enum
from CHANNEL_MODEL import ChannelModel
import numpy as np

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
            'BestCQI': BestCQIScheduler,
            'ProportionalFair': ProportionalFairScheduler,
            'RoundRobin': RoundRobinScheduler}
        
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
        return ['BestCQI', 'ProportionalFair', 'RoundRobin']
    
    def __init__(self, lte_grid, bs, 
                 max_dl_ue_tti=None, 
                 pcfich=2, 
                 max_dl_cce_allowance=None, 
                 verbose_pdcch=False, 
                 window_size=100,
                 enable_window = True,
                 verbose = False):
        
        self.lte_grid = lte_grid
        self.lte_grid.SET_BS(bs)
        self.amc = AdaptiveModulationAndCoding()
        self.pdcch_manager = PDCCHManager(bandwidth=lte_grid.bandwidth,
            pcfich=pcfich,
            max_dl_cce_allowance=max_dl_cce_allowance,
            verbose=verbose_pdcch)
        
        self.max_dl_ue_tti = max_dl_ue_tti
        self.enable_window = enable_window
        self.active_ue_window = []
        self.window_size = window_size
        
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
        
        # ЭТАП 1: Eligibility checks
        eligible_ues = self._filter_eligible_ues(tti, users)
        if not eligible_ues:
            return self._empty_result()
        
        # ЭТАП 2: Active window update
        self._update_active_window(eligible_ues)
        
        # ЭТАП 3: Priority calculation
        prioritized_ues = self._calculate_priorities(eligible_ues, tti)
        
        # ЭТАП 4: PList formation
        priority_list = self._form_priority_list(prioritized_ues, tti)
        if not priority_list:
            return self._empty_result()
        
        # Этап 4.5: PDSCH estimation
        priority_list = self._apply_pdsch_estimation(priority_list, tti)
        
        # ЭТАП 5: PDCCH allocation
        ues_with_pdcch = self._allocate_pdcch(priority_list)
        if not ues_with_pdcch:
            return self._empty_result()
        
        # ЭТАП 6: PDSCH allocation
        allocation = self._allocate_pdsch(tti, ues_with_pdcch, eligible_ues)
        
        # ЭТАП 7: Buffer processing
        self._process_buffers(tti, users, allocation)
        
        # ЭТАП 8: Result formation
        return self._build_result(allocation, users, eligible_ues, tti)
    
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
        
        for user in users:
            ue_id = user['UE_ID']
            
            # CHECK 1: BS buffer существует?
            bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
            if not bs_buffer:
                continue  # Пропускаем UE без буфера
            
            # CHECK 2: Получение Buffer status (формальный BSR)
            buffer_status = bs_buffer.GET_UE_STATUS(tti)['per_ue'].get(ue_id, {})
            buffer_size = buffer_status.get('size', 0)
            
            # CHECK 3: Buffer size > 0 и Valid CQI (1-15)
            if buffer_size > 0 and 1 <= user['cqi'] <= 15:
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
            print(f"[SCHEDULER TTI {tti}] Eligibility: {len(users)} total → {len(eligible)} eligible")
                
        return eligible

    def _update_active_window(self, eligible_ues: List[Dict]) -> None:
        """
        Обновление скользящего окна активных UE.
        Окно размером window_size TTI используется для отслеживания активности UE.
        Скользящее окно очень пригодится, когда количество UE в симуляции намного
        больше, чем десятки. А также для QoS.
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
            return
        
        current_ue_ids = {ue['UE_ID'] for ue in eligible_ues}
        
        self.active_ue_window.append(current_ue_ids)
        
        if len(self.active_ue_window) > self.window_size:
            self.active_ue_window.pop(0)

        # verbose
        if self.verbose and self.enable_window:
            unique = len(set.union(*self.active_ue_window)) if self.active_ue_window else 0
            print(f"[SCHEDULER] Active window: {len(current_ue_ids)} current, {unique} unique in window")

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
        total_rb = self.lte_grid.rb_per_slot
        estimated_rb = 0.0
        selected_ues = []
        
        # Threshold для early stopping (95% от total RB)
        # Оставляем 5% запас для overhead и динамики канала
        pdsch_threshold = total_rb * 0.95
        
        for user in priority_list:
            bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
            buffer_bits = user['bs_buffer_size'] * 8
            rb_needed = min(buffer_bits / (bits_per_rb * 2), total_rb)
        
            selected_ues.append(user)
            estimated_rb += rb_needed
        
            if estimated_rb >= pdsch_threshold:
                if self.verbose:
                    remaining = len(priority_list) - len(selected_ues)
                    print(f"[SCHEDULER TTI {tti}] PDSCH estimation: threshold reached "
                          f"({estimated_rb:.0f}/{total_rb} RB ≈ {estimated_rb/total_rb*100:.1f}%), "
                          f"{remaining} UE excluded")
                break  # Прекращаем для СЛЕДУЮЩИХ UE
            
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
            cqi = user['cqi']
            ue_id = user['UE_ID']
            
            required_cce = self.pdcch_manager.get_aggregation_level(cqi)
            
            if self.pdcch_manager.allocate_cce(ue_id, required_cce):
                user['allocated_cce'] = required_cce
                ues_with_pdcch.append(user)
        
        # verbose
        if self.verbose:
            blocked_count = len(priority_list) - len(ues_with_pdcch)
            print(f"[SCHEDULER] PDCCH allocation: {len(priority_list)} requested → {len(ues_with_pdcch)} allocated")
            
            if blocked_count > 0:
                allocated_ids = {u['UE_ID'] for u in ues_with_pdcch}
                blocked_ue_ids = [u['UE_ID'] for u in priority_list if u['UE_ID'] not in allocated_ids]
                print(f"[SCHEDULER] PDCCH blocked: {blocked_count} UE (no CCE available) - UE IDs: {blocked_ue_ids}")
        
        return ues_with_pdcch

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
        
        for user in users:
            ue = user['ue']
            ue_id = user['UE_ID']
            bs_buffer = self.lte_grid.bs.ue_buffers.get(ue_id)
            
            if not bs_buffer:
                continue
            
            allocated_rb = len(allocation.get(ue_id, [])) * 2  # 2 слота (0.5ms каждый)
        
            if allocated_rb == 0:
                continue  # UE не получил ресурсов - пропускаем
            
            bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
            max_bytes = (allocated_rb * bits_per_rb) // 8
            
            # Извлечение пакетов из буфера
            packets, total_bytes = bs_buffer.GET_PACKETS(
                ue_id=ue_id,
                max_bytes=max_bytes,
                bits_per_rb=bits_per_rb,
                current_time=tti)
            
            # Обновление throughput статистики
            transmitted_bits = total_bytes * 8
            ue.UPD_DL_THROUGHPUT(transmitted_bits, 1)
            
            total_bits_transmitted += transmitted_bits
            if transmitted_bits > 0:
                ues_transmitted += 1
        
        # Verbose 
        if self.verbose:
            print(f"[SCHEDULER TTI {tti}] Buffer processing: {total_bits_transmitted} bits transmitted to {ues_transmitted} UE")

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
        
        statistics = self.amc.calculate_throughput(
            allocation, users, tti, self.lte_grid.bs)
        
        pdcch_stats = self.pdcch_manager.get_stats()
        
        # Verbose
        if self.verbose:
            num_scheduled = sum(1 for rbs in allocation.values() if len(rbs) > 0)
            total_rb_allocated = sum(len(rbs) for rbs in allocation.values())
            print(f"[SCHEDULER TTI {tti}] Result: {num_scheduled} UE scheduled, {total_rb_allocated} RB allocated")
        
        return {
            'allocation': allocation,
            'statistics': statistics,
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
        
        return {
            'allocation': {},
            'statistics': {},
            'bitmap': {},
            'pdcch_stats': self.pdcch_manager.get_stats()}

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
        
        return {
            'window_enabled': True,
            'window_size': self.window_size,
            'current_active_count': len(self.active_ue_window[-1]),
            'unique_ues_in_window': len(set.union(*self.active_ue_window))}

#===============АБСТРАКТНЫЕ МЕТОДЫ ДЛЯ АЛГОРИТМОВ ПЛАНИРОВАНИЯ=================

    def _calculate_priorities(self, eligible_ues: List[Dict], tti: int) -> List[Dict]:
        """ 
        Расчет приоритетов для UE на основе алгоритма планирования.
        Подклассы ОБЯЗАНЫ реализовать этот метод.
    
        !Контракт: добавить поле user['priority'] для каждого UE.
        
        Примеры реализации:
        - BestCQI: priority = cqi
        - ProportionalFair: priority = instant_rate / avg_throughput
        - RoundRobin: priority = 1.0 (все равны)
        
        Args:
            eligible_ues: Отфильтрованные UE (прошли eligibility checks)
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
            'allocations': self.cce_allocations.copy()} #Копия, не оригинал
        
        return stats
    
#==============================================================================
#                               ЛОГИКА HARQ
#==============================================================================

class HARQState(Enum):
    """
    Состояния HARQ-процесса.
    """
    IDLE = 0           # Процесс свободен
    WAITING_ACK = 1    # Ожидание ACK/NACK
    RETRANSMIT = 2     # Требуется повторная передача
    
class HARQProcess:
    """
    Класс для управления одним HARQ-процессом.
    
    Attributes:
        process_id: Идентификатор процесса (0-7 для LTE)
        state: Текущее состояние процесса
        tx_count: Число попыток передачи (включая начальную)
        max_tx: Максимальное число передач
        tb_data: Буфер транспортного блока
        rv_sequence: Последовательность Redundancy Version
        cqi: Текущий CQI для передачи
        allocated_rbs: Список выделенных RB
        self.soft_bits: История soft bits всех передач
        self.soft_bits_combined: Результат combining
        self.llr_accumulator: Накопитель LLR значений
    """
    
    def __init__(self, process_id: int, max_tx: int = 4):
        """
        Инициализация HARQ-процесса.
        
        Args:
            process_id: Идентификатор процесса (0-7)
            max_tx: Максимальное число передач (по умолчанию 4)
            state: Текущее состояние процесса HARQ (IDLE, WAITING_ACK, RETRANSMIT)
            tx_count: Счетчик текущей попытки передачи
            tb_data: Сохраняемые транспортные блоки для передачи/повтора
            rv_sequence: Стандартная циклическая последовательность Redundancy Versions
            combined_rvs: Список RV, использованных для данного TB
            cqi: Channel Quality Indicator для UE в данной передаче
            allocated_rbs: Список выделенных ресурсных блоков
            soft_bits: Список массивов soft bits (LLR) каждой попытки (для soft combining)
            soft_bits_combined: Итоговый объединённый массив soft bits
            llr_accumulator: Накопитель LLR для soft combining
        """
        self.process_id = process_id
        self.state = HARQState.IDLE
        self.tx_count = 0
        self.max_tx = max_tx
        self.tb_data = None
        self.rv_sequence = [0, 2, 3, 1]  # Стандартная последовательность RV для LTE
        self.cqi = None
        self.allocated_rbs = []
        self.soft_bits = []
        self.soft_bits_combined = None
        self.llr_accumulator = None
        self.combined_rvs = []
        
    def start_transmission(self, tb_data: bytes, cqi: int, rbs: List[int],
                            soft_bits_received=None ):
        """
        Начало новой передачи TB.
        
        Args:
            tb_data: Данные транспортного блока
            cqi: Channel Quality Indicator
            rbs: Список выделенных resource blocks
            soft_bits_received: soft bits из канала (numpy array LLR значений)
                                если None - создаётся нулевой accumulator
            combined_rvs: Список RV, использованных для данного TB
            
        """
        self.state = HARQState.WAITING_ACK
        self.tx_count = 1
        self.tb_data = tb_data
        self.cqi = cqi
        self.allocated_rbs = rbs
        # инициализируем список RV для IR combining (первый RV)
        try:
            self.combined_rvs = [self.get_current_rv()]
        except Exception:
            self.combined_rvs = []

        # Сохранить soft bits первой передачи
        if soft_bits_received is not None:
            self.soft_bits.append(np.array(soft_bits_received).copy())
            # Инициализировать accumulator с первыми soft bits
            self.llr_accumulator = np.array(soft_bits_received).copy()
        else:
            # Если soft bits не переданы, создать нулевой accumulator
            self.llr_accumulator = None
        
    def handle_ack(self):
        """
        Обработка положительного подтверждения (ACK).
        """
        # При положительном подтверждении очищаем комбинированные RV и освобождаем процесс
        self.combined_rvs = []
        self.reset()
        self.tx_count = 0
        self.tb_data = None # Процесс освобождается
        
    def handle_nack(self, soft_bits_received=None) -> bool:
        """
        Обработка отрицательного подтверждения (NACK).
        
        Returns:
            True если возможна повторная передача, False если достигнут лимит
        """
        if self.tx_count < self.max_tx:
            self.tx_count += 1
            self.state = HARQState.RETRANSMIT

            # SOFT COMBINING: ГЛАВНАЯ ЛОГИКА
            if soft_bits_received is not None:
                soft_bits_received = np.array(soft_bits_received)
                self.soft_bits.append(soft_bits_received.copy())
                
                # Суммировать LLR значения
                if self.llr_accumulator is not None:
                    # Проверить совпадение размеров
                    if len(self.llr_accumulator) == len(soft_bits_received):
                        # LLR_новый = LLR_старый + LLR_новый
                        self.llr_accumulator = self.llr_accumulator + soft_bits_received
                        # Сохранить результат combining
                        self.soft_bits_combined = self.llr_accumulator.copy()
                    else:
                        # Если размеры не совпадают, использовать новые bits
                        self.llr_accumulator = soft_bits_received.copy()
                else:
                    # Если accumulator ещё не инициализирован
                    self.llr_accumulator = soft_bits_received.copy()
                    self.soft_bits_combined = soft_bits_received.copy()

            return True
        else:
            # Достигнут максимум передач, сбрасываем процесс
            self.reset()
            return False
            
    def get_current_rv(self) -> int:
        """
        Получение текущего Redundancy Version.
        
        Returns:
            Индекс RV для текущей передачи
        """
        return self.rv_sequence[(self.tx_count - 1) % len(self.rv_sequence)]
        
    def reset(self):
        """
        Сброс процесса в начальное состояние.
        """
        self.state = HARQState.IDLE
        self.tx_count = 0
        self.tb_data = None
        self.cqi = None
        self.allocated_rbs = []
        self.combined_rvs = []
        # Очистить soft bits при сбросе
        self.soft_bits = []
        self.soft_bits_combined = None
        self.llr_accumulator = None
        
    def is_idle(self) -> bool:
        """
        Проверка, свободен ли процесс.
        """
        return self.state == HARQState.IDLE
        
    def needs_retransmission(self) -> bool:
        """
        Проверка, требуется ли повторная передача.
        """
        return self.state == HARQState.RETRANSMIT

    def get_combined_soft_bits(self):
        """
        Получить объединённые soft bits после combining.
        
        Returns:
            numpy array с суммированными LLR значениями (или None)
        """
        return self.soft_bits_combined
    
    def get_soft_bits_history(self):
        """
        Получить историю всех soft bits для отладки.
        
        Returns:
            List[numpy array] - soft bits для каждой попытки передачи
        """
        return self.soft_bits
    
    def get_llr_accumulator(self):
        """
        Получить текущий LLR accumulator.
        """
        return self.llr_accumulator

class HARQManager:
    """
    Менеджер HARQ-процессов для всех UE в системе.
    
    Attributes:
        num_processes: Число HARQ-процессов на UE (обычно 8 для LTE)
        max_tx: Максимальное число передач на процесс
        processes: Словарь {ue_id: [HARQProcess, ...]}
    """
    
    def __init__(self, num_processes: int = 8, max_tx: int = 4):
        """
        Инициализация менеджера HARQ.
        
        Args:
            num_processes: Число HARQ-процессов на UE (по умолчанию 8)
            max_tx: Максимальное число передач (по умолчанию 4)
        """
        self.num_processes = num_processes
        self.max_tx = max_tx
        self.processes: Dict[int, List[HARQProcess]] = {}
        # Текущий TTI
        self.current_tti: Optional[int] = None
        # Очередь обратной связи: элементы (handle_tti, ue_id, process_id, ack, rv)
        self.feedback_queue: List[Tuple[int, int, int, bool, int]] = []
        # Статистика комбинирования IR
        self.combining_stats = {'combined_success': 0, 'combined_fail': 0}
        
    def init_ue(self, ue_id: int):
        """
        Инициализация HARQ-процессов для нового UE.
        
        Args:
            ue_id: Идентификатор UE
        """
        if ue_id not in self.processes:
            self.processes[ue_id] = [
                HARQProcess(pid, self.max_tx) 
                for pid in range(self.num_processes)
            ]
            
    def get_idle_process(self, ue_id: int) -> Optional[HARQProcess]:
        """
        Получение свободного HARQ-процесса для UE.
        
        Args:
            ue_id: Идентификатор UE
            
        Returns:
            Свободный HARQProcess или None если все заняты
        """
        if ue_id not in self.processes:
            self.init_ue(ue_id)
            
        for process in self.processes[ue_id]:
            if process.is_idle():
                return process
        return None
        
    def get_retransmission_process(self, ue_id: int) -> Optional[HARQProcess]:
        """
        Получение процесса, требующего повторной передачи.
        
        Args:
            ue_id: Идентификатор UE
            
        Returns:
            HARQProcess требующий ретрансмиссии или None
        """
        if ue_id not in self.processes:
            return None
            
        for process in self.processes[ue_id]:
            if process.needs_retransmission():
                return process
        return None
        
    def handle_feedback(self, ue_id: int, process_id: int, ack: bool,
    soft_bits_received=None):
        """
        Обработка ACK/NACK от UE.
        
        Args:
            ue_id: Идентификатор UE
            process_id: Идентификатор HARQ-процесса
            ack: True для ACK, False для NACK
            soft_bits_received: soft bits из канала (numpy array) для soft combining

        Если self.current_tti не задан — обратная связь обрабатывается немедленно.
        Если self.current_tti задан — обратная связь помещается в очередь
        и обработается через 4 TTI (HARQ RTT).
        """
        if ue_id not in self.processes:
            return
            
        proc = self.processes[ue_id][process_id]
        try:
            rv = proc.get_current_rv()
        except Exception:
            rv = -1

        if self.current_tti is None:
            # Немедленная обработка обратной связи (без RTT)
            if ack:
                proc.handle_ack()
                self.combining_stats['combined_success'] += 1
            else:
                # Передать soft bits в handle_nack
                success = proc.handle_nack(soft_bits_received=soft_bits_received)
                if success:
                    proc.combined_rvs.append(rv)
                else:
                    self.combining_stats['combined_fail'] += 1
            return

        # Работаем c симуляцией задержки HARQ RTT (DL LTE обычно 4 TTI)
        handle_tti = self.current_tti + 4
        self.feedback_queue.append((handle_tti, ue_id, process_id, ack, rv, soft_bits_received))

    def set_current_tti(self, tti: int):
        """
        Устанавливает текущий TTI и обрабатывает очередь обратной связи, срок которой наступил.
        """
        self.current_tti = tti
        self.process_feedback_queue()

    def process_feedback_queue(self):
        """
        Обрабатываем элементы очереди обратной связи, чей handle_tti <= current_tti.
        """
        if self.current_tti is None:
            return
        remaining = []
        for item in self.feedback_queue:
            handle_tti, ue_id, process_id, ack, rv, soft_bits_received = item
            if handle_tti <= self.current_tti:
                if ue_id not in self.processes:
                    continue
                proc = self.processes[ue_id][process_id]
                if ack:
                    if proc.combined_rvs:
                        self.combining_stats['combined_success'] += 1
                    proc.handle_ack()
                else:
                    s = proc.handle_nack(soft_bits_received)
                    if s:
                        proc.combined_rvs.append(rv)
                    else:
                        self.combining_stats['combined_fail'] += 1
            else:
                remaining.append(item)
        self.feedback_queue = remaining

    def get_retransmissions(self, tti: int):
        """
        Возвращает список UE/process, требующих ретрансмиссии (для указанного TTI).
        """
        ret=[]
        for ue_id, procs in self.processes.items():
            for p in procs:
                if p.needs_retransmission():
                    ret.append({'UE_ID': ue_id, 'process_id': p.process_id, 'rv_sequence': p.combined_rvs})
        return ret
                
    def get_statistics(self, ue_id: int) -> Dict[str, float]:
        """
        Получение статистики HARQ для UE.
        
        Args:
            ue_id: Идентификатор UE
            
        Returns:
            Словарь со статистикой {metric: value}
        """
        if ue_id not in self.processes:
            return {}
            
        total_tx = sum(p.tx_count for p in self.processes[ue_id] if not p.is_idle())
        active_processes = sum(1 for p in self.processes[ue_id] if not p.is_idle())
        
        return {
            'active_processes': active_processes,
            'avg_transmissions': total_tx / max(active_processes, 1),
            'idle_processes': self.num_processes - active_processes
        }

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
    
    def calculate_throughput(self, allocation: Dict, 
                             users: List[Dict], 
                             tti: int, bs: BaseStation) -> Dict:
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
            'total_effective_bits': 0}
        
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
            user['priority'] = user['cqi']  # Priority = CQI
        
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
            
            # Выделение RBG
            if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
                allocation[ue_id].extend(rb_indices)
                
                # Уменьшаем remaining_buffer
                bits_per_rb = self.amc.GET_BITS_PER_RB(best_user['cqi'])
                rbg_capacity = len(rb_indices) * bits_per_rb * 2
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
    
    def _calculate_priorities(self, eligible_ues: List[Dict], tti: int) -> List[Dict]:
        """
        Round Robin принцип распределения ресурсов
        """
        num_ues = len(eligible_ues)
        
        if num_ues == 0:
            return eligible_ues
        
        start_idx = self.rr_ue_offset % num_ues
        rotated_ues = eligible_ues[start_idx:] + eligible_ues[:start_idx]
        
        for idx, user in enumerate(rotated_ues):
            user['priority'] = num_ues - idx
        
        if self.max_dl_ue_tti:
            self.rr_ue_offset += self.max_dl_ue_tti
        else:
            self.rr_ue_offset += num_ues
        
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
        rbg_size = self.lte_grid.GET_RBG_SIZE()
        total_rbg = (self.lte_grid.rb_per_slot + rbg_size - 1) // rbg_size
        
        num_ues = len(ues_with_pdcch)
        if num_ues == 0:
            return allocation
        
        ue_index = self.rr_rbg_offset
        
        for rbg_idx in range(total_rbg):
            ue = ues_with_pdcch[ue_index % num_ues]
            ue_id = ue['UE_ID']
            
            if ue['bs_buffer_size'] > 0:
                if self.lte_grid.ALLOCATE_RBG(tti, rbg_idx, ue_id):
                    rb_indices = self.lte_grid.GET_RBG_INDICES(rbg_idx)
                    allocation[ue_id].extend(rb_indices)
                    
                    bits_per_rb = self.amc.GET_BITS_PER_RB(ue['cqi'])
                    transmitted_bits = len(rb_indices) * bits_per_rb * 2
                    transmitted_bytes = transmitted_bits // 8
                    ue['bs_buffer_size'] = max(0, ue['bs_buffer_size'] - transmitted_bytes)
            
            ue_index += 1
        
        self.rr_rbg_offset = ue_index % num_ues
        
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
            bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
            rb_per_slot = self.lte_grid.rb_per_slot
            instant_rate = rb_per_slot * bits_per_rb * 2 / 1000
            
            avg_throughput = user['ue'].current_dl_throughput
            
            if avg_throughput > 0:
                pf_metric = instant_rate / avg_throughput
            else:
                pf_metric = instant_rate
                # оптимизировал этот момент. можно изучить как было.
            
            user['priority'] = pf_metric
            user['instant_rate'] = instant_rate  # Для debug
        
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
                
                bits_per_rb = self.amc.GET_BITS_PER_RB(best_user['cqi'])
                rbg_capacity = len(rb_indices) * bits_per_rb * 2  # bits
                remaining_buffer[ue_id] -= min(remaining_buffer[ue_id], rbg_capacity)
        
        if self.verbose:
            allocated_ues = sum(1 for rbs in allocation.values() if len(rbs) > 0)
            total_rb = sum(len(rbs) for rbs in allocation.values())
            print(f"[SCHEDULER.ProportionalFair TTI {tti}] PDSCH: {allocated_ues} UE, {total_rb} RB total")
        
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
#   4) FD-Scheduling. Страшно. Очень страшно.
    
#==============================================================================
#                          ЛОГИКА SCHEDULER LEGACY
#==============================================================================

class RoundRobinScheduler_OLD:
    
    def __init__(self, lte_grid, bs: BaseStation, 
                 max_dl_ue_tti: Optional[int] = None,
                 pcfich: int = 2,
                 max_dl_cce_allowance: Optional[int] = None,
                 verbose_pdcch: bool = False):

        self.lte_grid = lte_grid
        self.lte_grid.SET_BS(bs)
        self.max_dl_ue_tti = max_dl_ue_tti
        self.last_served_ue_id = None 
        #теперь планировщик знает предыдущего обслуженного в tti прользователя
        #именно через этот метод
        self.amc = AdaptiveModulationAndCoding()
        self.harq_manager = HARQManager(num_processes=8, max_tx=4)
        for user_id in range(self.lte_grid.bs.num_ues):
            self.harq_manager.init_ue(user_id)
    
        
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

                        # --- HARQ: передача TB через канал ---
                        # пытаемся найти процесс для ретрансмиссии
                        harq_proc = self.harq_manager.get_retransmission_process(ue_id)
                        if harq_proc:
                            proc_id = harq_proc.process_id
                            tb_data = harq_proc.tb_data
                            is_retx = True
                        else:
                            # новый процесс
                            harq_proc = self.harq_manager.get_idle_process(ue_id)
                            proc_id = harq_proc.process_id

                            # расчёт максимального размера TB
                            max_bytes = (len(rb_indices) * self.amc.GET_BITS_PER_RB(user['cqi']) * 2) // 8
                            
                            # формируем TB из буфера UE
                            tb_data = user['ue'].buffer.GET_BYTES(max_bytes)  # или ваш pack_transport_block
                            harq_proc.start_transmission(tb_data, user['cqi'], rb_indices)
                            is_retx = False

                        # передаём TB и получаем ACK/NACK
                        ack = self.channel_model.receive_tb(
                            ue_id=ue_id,
                            rb_list=rb_indices,
                            cqi=user['cqi'],
                            process_id=proc_id,
                            is_retransmission=is_retx
                        )

                        # обрабатываем обратную связь
                        self.harq_manager.handle_feedback(ue_id, proc_id, ack)

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

class BestCQIScheduler_OLD:
    
    def __init__(self, lte_grid, bs: BaseStation, 
                 max_dl_ue_tti: Optional[int] = None,
                 pcfich: int = 2,
                 max_dl_cce_allowance: Optional[int] = None,
                 verbose_pdcch: bool = False):

        self.lte_grid = lte_grid
        self.lte_grid.SET_BS(bs)
        self.max_dl_ue_tti = max_dl_ue_tti
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

class ProportionalFairScheduler_OLD:
    
    def __init__(self, lte_grid, bs: BaseStation, 
                 max_dl_ue_tti: Optional[int] = None,
                 pcfich: int = 2,
                 max_dl_cce_allowance: Optional[int] = None,
                 verbose_pdcch: bool = False):
        super().__init__(lte_grid, max_dl_ue_tti)
        self.lte_grid = lte_grid
        self.lte_grid.SET_BS(bs)
        self.max_dl_ue_tti = max_dl_ue_tti
        self.amc = AdaptiveModulationAndCoding()
        self.channel_model = ChannelModel()
        
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

        # Инициализация для AMC результатов 
        amc_results = {user['UE_ID']: None for user in active_users}
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

                # Расчет AMC только один раз для пользователя
                if amc_results[ue_id] is None:
                    # Получить количество выделенных RB
                    num_rb = len(allocation[ue_id])
                    cqi = best_user['cqi']
                
                    # Вызов AMC функции
                    amc_results[ue_id] = self.channel_model.calculate_tbs(cqi=cqi, nprb=num_rb)
                
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

            # Использование AMC результатов
            if amc_results[ue_id]:
                amc = amc_results[ue_id]
                mcs = amc['mcs']
                tbs_bits = amc['tbs_bits']
                tbs_bytes = amc['tbs_bytes']
                throughput = amc['throughput_mbps']
                bits_per_rb = amc['qm']
                
                # Логирование AMC результатов
                print(f"[TTI {tti}] UE{ue_id}: CQI={amc['cqi']} → MCS={mcs} → "
                    f"NPRB={amc['nprb']} → TBS={tbs_bits} бит → {throughput:.3f} Мбит/с")
                
                max_bytes = tbs_bytes  # ← Используем TBS из AMC
            else:
                # Fallback на старый метод
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