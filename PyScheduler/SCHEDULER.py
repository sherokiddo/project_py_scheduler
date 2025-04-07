"""
#------------------------------------------------------------------------------
# Модуль: SCHEDULER - Планировщик ресурсов для сети LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет классы и методы для распределения ресурсных блоков между
# пользовательскими устройствами (UE) в сети LTE. Реализует алгоритм
# планирования Round Robin.
#
# Версия: 1.0.6
# Дата последнего изменения: 2025-04-07
# Версия Python Kernel: 3.12.9
# Автор: Брагин Кирилл
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
#------------------------------------------------------------------------------
"""

from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from UE_MODULE import UserEquipment, UECollection
from RES_GRID import RES_GRID_LTE, SchedulerInterface, RES_BLCK

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
        if cqi < 1 or cqi > 15:
            raise ValueError(f"Invalid CQI: {cqi}. Must be 1-15.")
        
        modulation, code_rate = self.CQI_TO_MCS[cqi]
        symbols_per_rb = 12 * 7  # 84 символа в RB
        return int(symbols_per_rb * modulation * code_rate)


class RoundRobinScheduler(SchedulerInterface):
    def __init__(self, lte_grid: RES_GRID_LTE):
        super().__init__(lte_grid)
        self.last_served_index = -1
        self.amc = AdaptiveModulationAndCoding()

    def schedule(self, tti: int, users: List[Dict]) -> Dict:
        """
        Планирование ресурсов по алгоритму Round Robin с учетом CQI и AMC.
        
        Args:
            tti: Индекс TTI
            users: Список пользователей с их параметрами
            
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
        
        # Сортировка по UE_ID для детерминированного порядка
        active_users.sort(key=lambda u: u['UE_ID'])
        
        # Получение свободных RB
        free_rbs = self.lte_grid.GET_FREE_RB_FOR_TTI(tti)
        allocation = {user['UE_ID']: [] for user in active_users}
        
        # Распределение RB
        start_index = (self.last_served_index + 1) % len(active_users)
        for i, rb in enumerate(free_rbs):
            user_idx = (start_index + i) % len(active_users)
            user = active_users[user_idx]
            
            if self.lte_grid.ALLOCATE_RB(tti, rb.freq_idx, user['UE_ID']):
                allocation[user['UE_ID']].append(rb.freq_idx)
                self.last_served_index = user_idx
        
        # Расчет пропускной способности через AMC
        stats = self._calculate_throughput(allocation, active_users)
        
        return {
            'allocation': allocation,
            'statistics': stats
        }

    def _calculate_throughput(self, allocation: Dict, users: List[Dict]) -> Dict:
        """
        Рассчитать пропускную способность для каждого пользователя.
        
        Args:
            allocation: Распределение RB {UE_ID: [freq_indices]}
            users: Список активных пользователей
            
        Returns:
            Dict: Статистика пропускной способности
        """
        stats = {
            'total_allocated_rbs': 0,
            'user_throughput': {},
            'average_throughput': 0.0
        }
        
        total_bits = 0
        for user in users:
            ue_id = user['UE_ID']
            rb_count = len(allocation.get(ue_id, []))
            if rb_count == 0:
                continue
                
            bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
            user_bits = bits_per_rb * rb_count
            
            stats['user_throughput'][ue_id] = user_bits
            total_bits += user_bits
            stats['total_allocated_rbs'] += rb_count
            
        if users:
            stats['average_throughput'] = total_bits / len(users)
            
        return stats    
    
def test_round_robin_scheduler():
    """
    Тестирование работы планировщика Round Robin.
    Проверяет:
    1. Корректность распределения RB между пользователями
    2. Учет CQI при расчете пропускной способности
    3. Работу с разным количеством пользователей и RB
    """
    print("\n=== Тестирование Round Robin Scheduler ===")
    
    # 1. Инициализация ресурсной сетки (10 МГц, 1 кадр = 10 TTI)
    lte_grid = RES_GRID_LTE(bandwidth=10, num_frames=1)
    scheduler = RoundRobinScheduler(lte_grid)
    
    # 2. Создание тестовых пользователей
    users = [
        {"UE_ID": 1, "buffer_size": 5000, "cqi": 10},  # Высокий CQI
        {"UE_ID": 2, "buffer_size": 3000, "cqi": 5},   # Средний CQI
        {"UE_ID": 3, "buffer_size": 0, "cqi": 15}      # Пустой буфер (не должен получить RB)
    ]
    
    # 3. Запуск симуляции для 3 TTI
    for tti in range(3):
        print(f"\nTTI {tti}:")
        result = scheduler.schedule(tti, users)
        
        # Проверки
        free_rbs = len(lte_grid.GET_FREE_RB_FOR_TTI(tti))
        allocated_rbs = sum(len(rbs) for rbs in result['allocation'].values())

        print(f"Свободные RB: {free_rbs}")
        print(f"Выделенные RB: {allocated_rbs}")
        print(f"Ожидалось RB: {lte_grid.num_rb}")
        
        assert allocated_rbs + free_rbs == 50, f"Ожидается 50 RB на TTI, получено {allocated_rbs + free_rbs}"
        
        # Основные проверки
        expected_rbs = len(lte_grid.GET_FREE_RB_FOR_TTI(tti)) + allocated_rbs
        assert expected_rbs == lte_grid.num_rb, f"Ошибка: {allocated_rbs} + {free_rbs} ≠ {lte_grid.num_rb}"
                
        # Вывод результатов
        print(f"Распределение RB: {result['allocation']}")
        print(f"Пропускная способность: {result['statistics']}")
        
        # Проверка ротации пользователей
        if tti == 0:
            assert len(result['allocation'][1]) > 0, "Первый пользователь не получил RB"
        elif tti == 1:
            assert len(result['allocation'][2]) > 0, "Второй пользователь не получил RB"
    
    print("\nТест пройден успешно!")

# Запуск теста при выполнении модуля
if __name__ == "__main__":
    test_round_robin_scheduler()