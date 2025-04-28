"""
#------------------------------------------------------------------------------
# Модуль: SCHEDULER - Планировщик ресурсов для сети LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет классы и методы для распределения ресурсных блоков между
# пользовательскими устройствами (UE) в сети LTE. Реализует алгоритм
# планирования Round Robin.
#
# Версия: 1.0.7
# Дата последнего изменения: 2025-04-09
# Версия Python Kernel: 3.12.9
# Автор: Брагин Кирилл, Норицин Иван
#
# Зависимости:
# - UE_MODULE.py (модели пользовательского оборудования)
# - RES_GRID (модель ресурсной сетки LTE)
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
#------------------------------------------------------------------------------
"""

from typing import Dict, List, Optional, Union, Tuple
from RES_GRID import RES_GRID_LTE, SchedulerInterface

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
        symbols_per_rb = 12 * 7  # 84 символа в RB
        return int(symbols_per_rb * modulation * code_rate)
	#@sherokiddo: "Добавить зависимость от CP"
    
    def calculate_throughput(self, allocation: Dict, users: List[Dict]) -> Dict:
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
        stats = {
            'total_allocated_rbs': 0,
            'user_throughput': {u['UE_ID']: 0 for u in users},  # Инициализация для всех UE
            'average_throughput': 0.0,
            'total_effective_bits': 0
        }
        
        total_effective_bits = 0
    
        for user in users:
            ue_id = user['UE_ID']
            rb_count = len(allocation.get(ue_id, []))
            
            if rb_count == 0:
                continue
    
            # 1. Реальные переданные биты из буфера
            real_bits = user['ue'].current_throughput * 1e-3  # бит/мс → бит/с
    
            # 2. Расчет максимальной ёмкости RB для данного CQI
            bits_per_rb = self.GET_BITS_PER_RB(user['cqi'])

            # 3. Расчет эффективно использованных бит
            max_bits = rb_count * bits_per_rb
            effective_bits = min(real_bits, max_bits)
            
            # 4. Обновление статистики
            stats['user_throughput'][ue_id] = effective_bits
            total_effective_bits += effective_bits
            stats['total_allocated_rbs'] += rb_count
    
        # 5. Расчет средней пропускной способности
        active_users = [u for u in users if stats['user_throughput'][u['UE_ID']] > 0]
        
        if active_users:
            stats['average_throughput'] = total_effective_bits / len(active_users)
        
        stats['total_effective_bits'] = total_effective_bits
        return stats

class RoundRobinScheduler(SchedulerInterface):
    
    def __init__(self, lte_grid: RES_GRID_LTE):
        super().__init__(lte_grid)
        self.last_served_index = -1
        self.amc = AdaptiveModulationAndCoding()  
    
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
        # Фильтрация активных пользователей
        active_users = [
            user for user in users 
            if user['buffer_size'] > 0 and 1 <= user['cqi'] <= 15
        ]
        if not active_users:
            return {'allocation': {}, 'statistics': {}}

        # 2. Получение уникальных частотных индексов для TTI
        free_rbs = self.lte_grid.GET_FREE_RB_FOR_TTI(tti)
        freq_indices = list({rb.freq_idx for rb in free_rbs})

        # 3. Инициализация структур данных
        allocation = {user['UE_ID']: [] for user in active_users}
        current_idx = (self.last_served_index + 1) % len(active_users)

        # 4. Основной цикл распределения
        for freq_idx in freq_indices:
            user = active_users[current_idx]
            
            # Попытка выделить оба слота
            if self.lte_grid.ALLOCATE_RB_PAIR(tti, freq_idx, user['UE_ID']):
                allocation[user['UE_ID']].extend([freq_idx]*2)
                current_idx = (current_idx + 1) % len(active_users)

        # 5. Обновление состояния
        self.last_served_index = current_idx

        # 6. Обработка буфера и статистики
        for user in users:
            ue = user['ue']
            if user in active_users:
                bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
                max_bytes = (len(allocation[user['UE_ID']]) * bits_per_rb) // 8
                packets, total = ue.buffer.GET_PACKETS(max_bytes, bits_per_rb, current_time=tti)
                ue.UPD_THROUGHPUT(total * 8, 1)
            else:
                ue.UPD_THROUGHPUT(0, 1)

        return {
            'allocation': allocation,
            'statistics': self.amc.calculate_throughput(allocation, active_users)
        }
    
    #@sherokiddo в рамках оптимизации можно разбить весь планировщик на несколько методов
    #для удобного логирования и подсчета времени. Например - подготовка данных один метод
    #затем идет непосредственно все планирование, и метод формирования статистики

class BestCQIScheduler(SchedulerInterface):
    
    def __init__(self, lte_grid: RES_GRID_LTE):
        super().__init__(lte_grid)
        self.last_served_index = -1
        self.amc = AdaptiveModulationAndCoding()  
    
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
        # Фильтрация активных пользователей
        active_users = [
            user for user in users 
            if user['buffer_size'] > 0 and 1 <= user['cqi'] <= 15
        ]
        if not active_users:
            return {'allocation': {}, 'statistics': {}}

        # 2. Сортировка по CQI (по убыванию)
        active_users.sort(key=lambda u: u['cqi'], reverse=True)
        max_cqi = active_users[0]['cqi']
        bcqi_users = [u for u in active_users if u['cqi'] == max_cqi]

        # 3. Получение уникальных свободных частотных индексов
        free_rbs = self.lte_grid.GET_FREE_RB_FOR_TTI(tti)
        freq_indices = list({rb.freq_idx for rb in free_rbs})

        # 4. Инициализация структур данных
        allocation = {user['UE_ID']: [] for user in active_users}
        current_idx = (self.last_served_index + 1) % len(bcqi_users)

        # 5. Основной цикл распределения
        for freq_idx in freq_indices:
            user = bcqi_users[current_idx % len(bcqi_users)]
            
            # Попытка выделить группу RB
            if self.lte_grid.ALLOCATE_RB_PAIR(tti, freq_idx, user['UE_ID']):
                allocation[user['UE_ID']].append(freq_idx)
                current_idx += 1

        # 6. Обновление индекса последнего пользователя
        self.last_served_index = current_idx % len(bcqi_users)

        # 7. Обработка буфера и статистики
        for user in users:
            ue = user['ue']
            if user in active_users:
                allocated_pairs = len(allocation[user['UE_ID']])
                total_rb = allocated_pairs * 2  # 2 RB на пару
                
                bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
                max_bytes = (total_rb * bits_per_rb) // 8
                
                packets, total = ue.buffer.GET_PACKETS(max_bytes, bits_per_rb, current_time=tti)
                ue.UPD_THROUGHPUT(total * 8, 1)
            else:
                ue.UPD_THROUGHPUT(0, 1)

        stats = self.amc.calculate_throughput(allocation, active_users)
        return {'allocation': allocation, 'statistics': stats}
    
    #@sherokiddo в рамках оптимизации можно разбить весь планировщик на несколько методов
    #для удобного логирования и подсчета времени. Например - подготовка данных один метод
    #затем идет непосредственно все планирование, и метод формирования статистики

class ProportionalFairScheduler(SchedulerInterface):
    
    def __init__(self, lte_grid: RES_GRID_LTE):
        super().__init__(lte_grid)
        self.last_served_index = -1
        self.amc = AdaptiveModulationAndCoding()
        
    def calculate_pf_metric(self, users: List[Dict]):
        """
        Расчёт PF-метрики для каждого пользователя.
        
        Args:
            users: Список пользователей с параметрами:
                - 'UE_ID': Идентификатор
                - 'buffer_size': Размер буфера в байтах
                - 'cqi': Индекс качества канала
                - 'ue': Объект UserEquipment
        
        Returns:
            users: Список пользователей с добавленной PF-метрикой
        """
        for user in users:
            bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
            rb_per_slot = self.lte_grid.rb_per_slot
            instant_throughput = rb_per_slot * bits_per_rb * 2 * 1000
            
            if user['ue'].average_throughput <= 0:
                user['ue'].average_throughput = 1e-6
            
            PF_metric = instant_throughput / user['ue'].average_throughput
            user['PF_metric'] = PF_metric
            user['ue'].PF_metric = PF_metric
            print(f"PF metric: UE{user['UE_ID']} = {PF_metric}")
        return users
            
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
        # Фильтрация активных пользователей
        active_users = [user for user in users if user['buffer_size'] > 0 and 1 <= user['cqi'] <= 15]
        if not active_users:
            return {'allocation': {}, 'statistics': {}}
        
        # Расчёт PF-метрики для каждого пользователя
        active_users = self.calculate_pf_metric(active_users)
        
        # Сортировка по PF-метрике
        active_users.sort(key=lambda u: u['PF_metric'], reverse=True)
        max_pf_metric = active_users[0]['PF_metric']
        pf_users = [u for u in active_users if u['PF_metric'] == max_pf_metric]
        
        # Получение всех свободных RB и разделение по слотам
        free_rbs = self.lte_grid.GET_FREE_RB_FOR_TTI(tti)
        slots = {
            0: [rb for rb in free_rbs if rb.freq_idx < self.lte_grid.rb_per_slot],
            1: [rb for rb in free_rbs if rb.freq_idx >= self.lte_grid.rb_per_slot]
        }
        
        allocation = {user['UE_ID']: [] for user in active_users}
        start_index = (self.last_served_index + 1) % len(pf_users)
        
        # Распределение RB для каждого слота
        for slot_id in [0, 1]:
            current_index = start_index
            for rb in slots[slot_id]:
                user = pf_users[current_index % len(pf_users)]
                if self.lte_grid.ALLOCATE_RB(tti, rb.slot_id, rb.freq_idx, user['UE_ID']):
                    allocation[user['UE_ID']].append(rb.freq_idx)
                    current_index += 1
                    
        # Обновление индекса последнего пользователя
        self.last_served_index = (start_index + len(free_rbs)) % len(pf_users)
        
        # Обработка буфера и статистики
        for user in users:
            ue = user['ue']
            if user in active_users:
                bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
                max_bytes = (len(allocation[user['UE_ID']]) * bits_per_rb) // 8
                packets, total = ue.buffer.GET_PACKETS(max_bytes, bits_per_rb, current_time=tti*1)
                ue.UPD_THROUGHPUT(total * 8, 1)
            else:
                ue.UPD_THROUGHPUT(0, 1)
            
            # Обновление средней пропускной способности пользователей
            if user in pf_users:
                average_throughput_past = ue.average_throughput
                ue.average_throughput = (1 - 0.2) * average_throughput_past + 0.2 * ue.current_throughput 
            else:
                average_throughput_past = ue.average_throughput
                ue.average_throughput = (1 - 0.2) * average_throughput_past
        
        stats = self.amc.calculate_throughput(allocation, active_users)
        return {'allocation': allocation, 'statistics': stats}
    
class ProportionalFairScheduler_v2(SchedulerInterface):
    
    def __init__(self, lte_grid: RES_GRID_LTE):
        super().__init__(lte_grid)
        self.amc = AdaptiveModulationAndCoding()
        self.history = {}  # Словарь для хранения истории throughput пользователей
        self.alpha = 0.1   # Коэффициент сглаживания для EMA

    def update_history(self, UE_ID: int, throughput: float):
        """Обновление истории пропускной способности пользователя"""
        if UE_ID not in self.history:
            self.history[UE_ID] = throughput
        else:
            self.history[UE_ID] = (1 - self.alpha) * self.history[UE_ID] + self.alpha * throughput

    def get_average_throughput(self, UE_ID: int) -> float:
        """Получение средней пропускной способности пользователя"""
        return self.history.get(UE_ID, 1e-6)  # Защита от деления на ноль
        
    def calculate_pf_metric(self, user: Dict) -> float:
        """Расчет PF-метрики с обработкой исключений"""
        try:
            UE_ID = user['UE_ID']
            cqi = user['cqi']
            
            bits_per_rb = self.amc.GET_BITS_PER_RB(cqi)
            instant_throughput = 2 * bits_per_rb * 1000  # бит/с (с учетом двух слотов)
            
            avg_throughput = self.get_average_throughput(UE_ID)
            if avg_throughput <= 0:
                avg_throughput = 1e-6  # Защита от деления на ноль
                
            return instant_throughput / avg_throughput
        except (KeyError, ValueError) as e:
            # Объединенная обработка ошибок
            print(f"Ошибка расчета метрики для UE {user.get('UE_ID', 'неизвестен')}: {e}")
            return 0  # Возвращаем 0 вместо генерации исключения
    
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
        # Фильтрация активных пользователей
        active_users = []
        for user in users:
            try:
                if (user['buffer_size'] > 0 
                    and 1 <= user['cqi'] <= 15 
                    and 'UE_ID' in user 
                    and 'ue' in user):
                    active_users.append(user)
            except KeyError as e:
                print(f"Некорректные данные пользователя: отсутствует ключ {e}")
                continue
    
        if not active_users:
            return {'allocation': {}, 'statistics': {}}
    
        # Расчет PF-метрик с обработкой ошибок
        pf_metrics = {}
        for user in active_users:
            UE_ID = user['UE_ID']
            try:
                metric = self.calculate_pf_metric(user)
                pf_metrics[UE_ID] = metric
            except (KeyError, ValueError) as e:
                print(f"Ошибка расчета метрики для UE {UE_ID}: {e}")
                pf_metrics[UE_ID] = 0  # Значение по умолчанию

        # Получение уникальных частотных индексов
        free_rbs = self.lte_grid.GET_FREE_RB_FOR_TTI(tti)
        freq_indices = list({rb.freq_idx for rb in free_rbs})  # <-- Добавлено
    
        # Сортировка с гарантией числовых значений
        sorted_users = sorted(
            active_users,
            key=lambda u: pf_metrics.get(u['UE_ID'], 0),
            reverse=True
        )
        # Инициализация распределения
        allocation = {u['UE_ID']: [] for u in active_users}
        current_idx = 0

        # Распределение ресурсов
        for freq_idx in freq_indices:
            if current_idx >= len(sorted_users):
                current_idx = 0
            
            user = sorted_users[current_idx]
            if self.lte_grid.ALLOCATE_RB_PAIR(tti, freq_idx, user['UE_ID']):
                allocation[user['UE_ID']].append(freq_idx)
                current_idx += 1

        # Обработка буфера и обновление статистики
        for user in active_users:
            ue = user['ue']
            allocated_pairs = len(allocation[user['UE_ID']])
            
            bits_per_pair = 2 * self.amc.GET_BITS_PER_RB(user['cqi'])
            max_bytes = (allocated_pairs * bits_per_pair) // 8
            
            packets, total = ue.buffer.GET_PACKETS(max_bytes, bits_per_pair, current_time=tti)
            ue.UPD_THROUGHPUT(total * 8, 1)
            self.update_history(user['UE_ID'], total * 8)

        # Расчет итоговой статистики
        stats = self.amc.calculate_throughput(allocation, active_users)
        return {'allocation': allocation, 'statistics': stats}