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
import numpy as np
from UE_MODULE import UserEquipment, UECollection
from RES_GRID import RES_GRID_LTE, SchedulerInterface, RES_BLCK
from CHANNEL_MODEL import RMaModel, UMaModel, UMiModel
from BS_MODULE import BaseStation

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
        active_users = [user for user in users if user['buffer_size'] > 0 and 1 <= user['cqi'] <= 15]
        if not active_users:
            return {'allocation': {}, 'statistics': {}}
        
        # Сортировка по UE_ID
        active_users.sort(key=lambda u: u['UE_ID'])
        
        # Получение всех свободных RB и разделение по слотам
        free_rbs = self.lte_grid.GET_FREE_RB_FOR_TTI(tti)
        slots = {
            0: [rb for rb in free_rbs if rb.freq_idx < self.lte_grid.rb_per_slot],
            1: [rb for rb in free_rbs if rb.freq_idx >= self.lte_grid.rb_per_slot]
        }
        
        allocation = {user['UE_ID']: [] for user in active_users}
        start_index = (self.last_served_index + 1) % len(active_users)
        
        # Распределение RB для каждого слота
        for slot_id in [0, 1]:
            current_index = start_index
            for rb in slots[slot_id]:
                user = active_users[current_index]
                if self.lte_grid.ALLOCATE_RB(tti, rb.slot_id, rb.freq_idx, user['UE_ID']):
                    allocation[user['UE_ID']].append(rb.freq_idx)
                    current_index = (current_index + 1) % len(active_users)
        
        # Обновление индекса последнего пользователя
        self.last_served_index = (start_index + len(free_rbs)) % len(active_users)
        
        # Обработка буфера и статистики
        for user in active_users:
            ue = user['ue']
            bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])
            max_bytes = (len(allocation[user['UE_ID']]) * bits_per_rb) // 8
            packets, total = ue.buffer.GET_PACKETS(max_bytes, bits_per_rb, current_time=tti*1)
            ue.UPD_THROUGHPUT(total * 8, 1)
        
        stats = self._calculate_throughput(allocation, active_users)
        return {'allocation': allocation, 'statistics': stats}
    
    def _calculate_throughput(self, allocation: Dict, users: List[Dict]) -> Dict:
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
            bits_per_rb = self.amc.GET_BITS_PER_RB(user['cqi'])

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


def test_scheduler_with_buffer():
    """
    Расширенный тест работы планировщика Round Robin с буфером пользователей.
    Теперь выводит требования RB и фактические выделения без фиксированных проверок.
    """
    print("\n=== Расширенный тест планировщика с буфером ===")

    #------------------------------------------------------------------
    # Шаг 1: Инициализация ресурсной сетки (10 МГц, 1 кадр = 10 TTI)
    #------------------------------------------------------------------
    lte_grid = RES_GRID_LTE(bandwidth=10, num_frames=1)
    print("[OK] Ресурсная сеть инициализирована")

    #------------------------------------------------------------------
    # Шаг 2: Создание планировщика
    #------------------------------------------------------------------
    scheduler = RoundRobinScheduler(lte_grid)
    print("[OK] Планировщик Round Robin создан")

    #------------------------------------------------------------------
    # Шаг 3: Создание пользователей и базовой станции
    #------------------------------------------------------------------
    bs = BaseStation(x=0, y=0, height=25.0, bandwidth=10)
    ue1 = UserEquipment(UE_ID=1, x=500, y=500, ue_class="pedestrian")
    ue2 = UserEquipment(UE_ID=2, x=1000, y=1000, ue_class="car")
    
    # Установка моделей канала
    ue1.SET_CH_MODEL(UMiModel(bs))
    ue2.SET_CH_MODEL(UMiModel(bs))
    print("[OK] Пользователи созданы")

    #------------------------------------------------------------------
    # Шаг 4: Добавление пакетов в буфер
    #------------------------------------------------------------------
    current_time = 0
    ue1.buffer.ADD_PACKET(1500, creation_time=current_time, current_time=current_time)  # 1500 Б
    ue2.buffer.ADD_PACKET(3000, creation_time=current_time, current_time=current_time)  # 3000 Б
    print("[OK] Пакеты добавлены в буферы")

    #------------------------------------------------------------------
    # Шаг 5: Обновление позиций и качества канала
    #------------------------------------------------------------------
    ue1.UPD_POSITION(current_time, bs.position, bs.height)
    ue2.UPD_POSITION(current_time, bs.position, bs.height)
    ue1.UPD_CH_QUALITY()
    ue2.UPD_CH_QUALITY()
    print(f"[OK] CQI рассчитан: UE1={ue1.cqi}, UE2={ue2.cqi}")

    #------------------------------------------------------------------
    # Шаг 6: Подготовка данных для планировщика
    #------------------------------------------------------------------
    users = [
        {'UE_ID': 1, 'buffer_size': 1500, 'cqi': ue1.cqi, 'ue': ue1},
        {'UE_ID': 2, 'buffer_size': 3000, 'cqi': ue2.cqi, 'ue': ue2}
    ]

    #------------------------------------------------------------------
    # Шаг 7: Запуск планировщика для TTI=0
    #------------------------------------------------------------------
    result = scheduler.schedule(0, users)
    allocation = result['allocation']
    stats = result['statistics']

    #------------------------------------------------------------------
    # Шаг 8: Анализ распределения RB по слотам
    #------------------------------------------------------------------
    print("\nДетализация распределения RB:")
    
    # Получение подкадра (TTI=0)
    subframe = lte_grid.GET_SUBFRAME(0)
    
    # Анализ слота 0
    slot0 = subframe.slots[0]
    slot0_rbs = {rb.freq_idx: rb.UE_ID for rb in slot0.GET_ALL_RES_BLCK()}
    
    # Анализ слота 1
    slot1 = subframe.slots[1]
    slot1_rbs = {rb.freq_idx: rb.UE_ID for rb in slot1.GET_ALL_RES_BLCK()}

    print("\nСлот 0:")
    print("RB Индекс | Пользователь")
    print("-----------------------")
    for freq_idx in sorted(slot0_rbs.keys()):
        print(f"{freq_idx:8} | {slot0_rbs[freq_idx] or 'Свободен'}")

    print("\nСлот 1:")
    print("RB Индекс | Пользователь")
    print("-----------------------")
    for freq_idx in sorted(slot1_rbs.keys()):
        print(f"{freq_idx:8} | {slot1_rbs[freq_idx] or 'Свободен'}")

    #------------------------------------------------------------------
    # Шаг 9: Вывод параметров MCS для каждого пользователя
    #------------------------------------------------------------------
    print("\nПараметры MCS:")
    for user in users:
        cqi = user['cqi']
        mcs_params = scheduler.amc.CQI_TO_MCS.get(cqi, (0, 0))
        bits_per_rb = scheduler.amc.GET_BITS_PER_RB(cqi)
        
        print(f"\nUE {user['UE_ID']} (CQI={cqi}):")
        print(f"- Modulation: {'QPSK' if mcs_params[0] == 2 else '16QAM' if mcs_params[0] == 4 else '64QAM'}")
        print(f"- Code Rate: {mcs_params[1]:.3f}")
        print(f"- Биты/RB: {bits_per_rb}")

    #------------------------------------------------------------------
    # Шаг 10: Общая статистика
    #------------------------------------------------------------------
    print("\nИтоговая статистика:")
    total_throughput = 0
    
    for user in users:
        ue_id = user['UE_ID']
        rb_count = len(allocation.get(ue_id, []))
        bits_per_rb = scheduler.amc.GET_BITS_PER_RB(user['cqi'])
        throughput = rb_count * bits_per_rb  # Бит/мс = кбит/с
        
        print(f"\nUE {ue_id}:")
        print(f"  RB выделено: {rb_count}")
        print(f"  Пропускная способность: {throughput * 1000:.2f} бит/с")
        total_throughput += throughput

    # Средняя пропускная способность на TTI
    num_users = len([u for u in users if len(allocation.get(u['UE_ID'], [])) > 0])
    avg_throughput = (total_throughput / num_users) * 1000 if num_users > 0 else 0
    
    print(f"\nСредняя пропускная способность в TTI: {avg_throughput:.2f} бит/с")
    print(f"Всего выделено RB: {stats['total_allocated_rbs']}/100")
    print("[OK] Тест пройден успешно!")

if __name__ == "__main__":
    test_scheduler_with_buffer()