"""
#------------------------------------------------------------------------------
# Модуль: SCHEDULER - Планировщик ресурсов для сети LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет классы и методы для распределения ресурсных блоков между
# пользовательскими устройствами (UE) в сети LTE. Реализует алгоритм
# планирования Round Robin.

# Версия: 1.0.5
# Дата последнего изменения: 2025-03-23
# Версия Python Kernel: 3.12.9
# Автор: Брагин Кирилл

# Зависимости:
# - UE_MODULE.py (модели пользовательского оборудования)
# - LTE_GRID_ver_alpha.py (модель ресурсной сетки LTE)

# Изменения v1.0.5:
# - пофикшен баг с выделением ресурсов пользователям в ситуациях, когда количество
# юзеров намного больше чем количество RB в слоте. Выявлена ошибка в расчете последнего
# обслуженного UE last_served_user метода schedule
# - LTE_GRID_ver_alpha.py (модель ресурсной сетки LTE)
#------------------------------------------------------------------------------
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from UE_MODULE import UserEquipment, UECollection
from LTE_GRID_ver_alpha import RES_GRID_LTE, SchedulerInterface, RES_BLCK

class RoundRobinScheduler(SchedulerInterface):
    """
    Реализация алгоритма планирования Round Robin.
    """
    
    def __init__(self, lte_grid: RES_GRID_LTE):
        """
        Инициализация планировщика Round Robin.
        """
        super().__init__(lte_grid)
        self.last_served_index = -1  # Индекс последнего обслуженного пользователя
    
    def schedule(self, tti: int, users: List[Dict]) -> Dict:
        """
        Распределить ресурсные блоки между пользователями по алгоритму Round Robin.
        """
        # Фильтруем только активных пользователей (с данными в буфере)
        active_users = [user for user in users if user.get('buffer_size', 0) > 0]
        
        if not active_users:
            return {'allocation': {}, 'statistics': {'allocated_rbs': 0, 'active_users': 0}}
        
        # Сортируем пользователей по ID для предсказуемого порядка
        active_users = sorted(active_users, key=lambda u: u['UE_ID'])
        
        # Получаем свободные ресурсные блоки для текущего TTI
        free_rbs = self.lte_grid.GET_FREE_RB_FOR_TTI(tti)
        
        # Подготавливаем результат
        allocation = {user['UE_ID']: [] for user in active_users}
        
        # Распределяем RB между пользователями по кругу
        if active_users and free_rbs:
            num_users = len(active_users)
            
            # Определяем начальный индекс для этого TTI
            start_user_index = (self.last_served_index + 1) % num_users
            
            # Отслеживаем последнего обслуженного пользователя
            last_user_index = None
            
            # Назначаем RB пользователям
            for i, rb in enumerate(free_rbs):
                # Определяем пользователя для текущего RB
                user_index = (start_user_index + i) % num_users
                user = active_users[user_index]
                
                # Назначаем RB пользователю
                if self.lte_grid.ALLOCATE_RB(tti, rb.freq_idx, user['UE_ID']):
                    allocation[user['UE_ID']].append(rb.freq_idx)
                    last_user_index = user_index
            
            # Обновляем индекс последнего обслуженного пользователя
            if last_user_index is not None:
                self.last_served_index = last_user_index
        
        # Собираем статистику
        statistics = {
            'allocated_rbs': sum(len(rbs) for rbs in allocation.values()),
            'active_users': len(active_users),
            'allocation_per_user': {ue_id: len(rbs) for ue_id, rbs in allocation.items()}
        }
        
        return {'allocation': allocation, 'statistics': statistics}
    
    def schedule_with_ue_collection(self, tti: int, ue_collection: UECollection) -> Dict:
        """
        Распределить ресурсные блоки между пользователями по алгоритму Round Robin,
        используя коллекцию UE.
        """
        # Получаем список всех пользователей и сортируем их по ID
        all_users = ue_collection.GET_ALL_USERS()
        all_users = sorted(all_users, key=lambda ue: ue.UE_ID)
        
        # Преобразуем список объектов UserEquipment в список словарей
        users_data = []
        for ue in all_users:
            buffer_status = ue.GET_BUFFER_STATUS(tti)
            channel_quality = ue.GET_CH_QUALITY()
            
            users_data.append({
                'UE_ID': ue.UE_ID,
                'buffer_size': buffer_status['size'],
                'packet_count': buffer_status['packet_count'],
                'cqi': channel_quality['cqi'],
                'SINR': channel_quality['SINR']
            })
        
        # Вызываем основной метод планирования
        result = self.schedule(tti, users_data)
        
        # Обновляем список назначенных RB для каждого пользователя
        for ue_id, rb_indices in result['allocation'].items():
            user = ue_collection.GET_USER(ue_id)
            if user:
                user.assigned_rbs = rb_indices
        
        return result

#возможно, следующую часть кода стоит выпилить из модуля, но пока работает не трогаю, перенесу потом
    
    def update_throughput(self, ue_collection: UECollection, allocation: Dict,
                         time_interval_ms: int):
        """
        Обновить пропускную способность для каждого пользователя на основе назначенных RB.
        """
        for ue_id, rb_indices in allocation.items():
            user = ue_collection.GET_USER(ue_id)
            if user and rb_indices:
                # Рассчитываем количество бит, которое может быть передано
                # через назначенные RB с учетом CQI пользователя
                bits_transmitted = self._calculate_bits_transmitted(user.cqi, len(rb_indices))
                
                # Обновляем статистику пропускной способности пользователя
                user.UPD_THROUGHPUT(bits_transmitted, time_interval_ms)
    
    def _calculate_bits_transmitted(self, cqi: int, num_rbs: int) -> int:
        """
        Рассчитать количество бит, которое может быть передано через заданное
        количество RB с учетом CQI.
        """
        # Таблица соответствия CQI и скорости передачи данных (бит/RB)
        # Значения взяты из спецификации 3GPP
        cqi_to_bits = {
            1: 78,     # QPSK, кодовая скорость 0.076
            2: 120,    # QPSK, кодовая скорость 0.117
            3: 193,    # QPSK, кодовая скорость 0.188
            4: 308,    # QPSK, кодовая скорость 0.300
            5: 449,    # QPSK, кодовая скорость 0.438
            6: 602,    # QPSK, кодовая скорость 0.588
            7: 378,    # 16QAM, кодовая скорость 0.369
            8: 490,    # 16QAM, кодовая скорость 0.478
            9: 616,    # 16QAM, кодовая скорость 0.601
            10: 466,   # 64QAM, кодовая скорость 0.455
            11: 567,   # 64QAM, кодовая скорость 0.554
            12: 666,   # 64QAM, кодовая скорость 0.650
            13: 772,   # 64QAM, кодовая скорость 0.754
            14: 873,   # 64QAM, кодовая скорость 0.852
            15: 948    # 64QAM, кодовая скорость 0.926
        }
        
        # Получаем количество бит на один RB
        bits_per_rb = cqi_to_bits.get(cqi, 78)  # По умолчанию используем значение для CQI=1
        
        # Рассчитываем общее количество бит
        return bits_per_rb * num_rbs
