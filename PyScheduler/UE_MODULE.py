"""
#------------------------------------------------------------------------------
# Модуль: UE_MODULE - Модель пользовательского устройства (UE) для сети LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет классы и методы для моделирования пользовательских устройств
# в сети LTE, включая их перемещение, генерацию трафика, управление буфером
# и оценку качества канала.
#
# Версия: 1.0.0
# Дата последнего изменения: 2025-03-16
# Автор: Брагин Кирилл
# Версия Python Kernel: 3.12.9
#
# Зависимости:
# - MOVEMENT_MODEL.py (модели перемещения пользователей)
# - CHANNEL_MODEL.py (модели распространения сигнала)
# - TRAFFIC_MODEL.py (модели генерации трафика)
#------------------------------------------------------------------------------
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
# Импорты из других модулей (будут созданы позже)
# from MOVEMENT_MODEL import MovementModel, RandomWalkModel, LinearModel, CircularModel
# from CHANNEL_MODEL import ChannelModel, FreeSpaceModel, OkumuraHataModel
# from TRAFFIC_MODEL import TrafficModel, ConstantBitRate, VoipModel, WebBrowsingModel
#upd

class UserEquipment:
    """
    Класс, представляющий пользовательское устройство (UE) в сети LTE.
    """
    def __init__(self, UE_ID: int, x: float = 0.0, y: float = 0.0,
                 buffer_size: int = 1048576, ue_class: str = "pedestrian"):
        """
        Инициализация пользовательского устройства.
        
        Args:
            UE_ID: Уникальный идентификатор пользователя
            x: Начальная координата X (м)
            y: Начальная координата Y (м)
            buffer_size: Размер буфера в байтах
            ue_class: Класс пользователя (стационарный, пешеход, машина, поезд)
        """
        self.UE_ID = UE_ID
        self.position = (x, y)  # Координаты (x, y) в метрах
        self.ue_class = ue_class
        
        # Параметры движения (будут настроены через модель движения)
        self.velocity = 0.0  # Скорость в м/с
        self.direction = 0.0  # Направление в радианах (ранее angle)
        self.movement_model = None  # Установить позже
        
        # Буфер данных
        self.buffer = Buffer(buffer_size)
        
        # Модель трафика
        self.traffic_model = None  # Установить позже
        
        # Параметры канала связи
        self.channel_model = None  # Установить позже
        self.cqi = 1  # Текущий CQI (1-15)
        self.SINR = 0.0  # Текущее отношение сигнал/шум+помехи в dB
        self.distance_to_bs = 0.0  # Расстояние до базовой станции в метрах
        
        # Параметры для алгоритмов планирования
        self.current_throughput = 0.0  # Текущая пропускная способность (бит/с)
        self.average_throughput = 0.0  # Средняя пропускная способность (бит/с)
        self.alpha = 0.1  # Коэффициент сглаживания для средней пропускной способности

        self.assigned_rbs = []  # Список выделенных ресурсных блоков
        self.mcs_index = 0  # Индекс MCS (0-28)
        
        # Статистика
        self.total_transmitted_bits = 0
        self.total_transmitted_packets = 0
        self.total_dropped_packets = 0
    
    def SET_MOVEMENT_MODEL(self, model):
        """
        Установить модель движения пользователя. Метод можно будет вызвать
        глобально и по конкретному UE_ID
        
        Args:
            model: Объект модели движения
        """
        self.movement_model = model
    
    def SET_CH_MODEL(self, model):
        """
        Установить модель канала связи.
        
        Args:
            model: Объект модели канала
        """
        self.channel_model = model
    
    def SET_TRAFFIC_MODEL(self, model):
        """
        Установить модель генерации трафика.
        
        Args:
            model: Объект модели трафика
        """
        self.traffic_model = model
    
    def UPD_POSITION(self, time_ms: int, bs_position: Tuple[float, float] = (0, 0)):
        """
        Обновить позицию пользователя согласно модели движения.
        
        Args:
            time_ms: Текущее время в миллисекундах
            bs_position: Координаты базовой станции (x, y) в метрах
        """
        if self.movement_model:
            self.position, self.velocity, self.direction = self.movement_model.update(
                self.position, self.velocity, self.direction, time_ms
            )
        
        # Обновить расстояние до БС
        self.distance_to_bs = np.sqrt(
            (self.position[0] - bs_position[0])**2 + 
            (self.position[1] - bs_position[1])**2
        )
    
    def UPD_CH_QUALITY(self, time_ms: int, bs_position: Tuple[float, float] = (0, 0)):
        """
        Обновить качество канала согласно модели распространения.
        
        Args:
            time_ms: Текущее время в миллисекундах
            bs_position: Координаты базовой станции (x, y) в метрах
        """
        if self.channel_model:
            self.SINR = self.channel_model.calculate_SINR(
                self.position, bs_position, time_ms
            )
            self.cqi = self.SINR_TO_CQI(self.SINR)
    
    def GEN_TRFFC(self, time_ms: int):
        """
        Сгенерировать новый трафик согласно заданной модели.
        
        Args:
            time_ms: Текущее время в миллисекундах
        """
        if self.traffic_model:
            packets = self.traffic_model.generate(time_ms)
            for packet_size, priority in packets:
                self.buffer.ADD_PACKET(packet_size, time_ms, priority)
    
    def UPD_THROUGHPUT(self, bits_transmitted: int, time_interval_ms: int):
        """
        Обновить статистику пропускной способности.
        
        Args:
            bits_transmitted: Количество переданных бит
            time_interval_ms: Интервал времени в мс
        """
        # Текущая пропускная способность в бит/с
        self.current_throughput = (bits_transmitted * 1000) / time_interval_ms if time_interval_ms > 0 else 0
        
        # Обновление экспоненциально сглаженного среднего
        if self.average_throughput == 0:
            self.average_throughput = self.current_throughput
        else:
            self.average_throughput = (1 - self.alpha) * self.average_throughput + self.alpha * self.current_throughput
        
        # Обновление общей статистики
        self.total_transmitted_bits += bits_transmitted
    
    def CALC_PF_METRIC(self) -> float:
        """
        Рассчитать метрику Proportional Fair (PF).
        
        Returns:
            float: Значение метрики PF
        """
        if self.average_throughput <= 0:
            return float('inf')  # Если не было передачи, присваиваем высокий приоритет
        return self.current_throughput / self.average_throughput
    
    def GET_BUFFER_STATUS(self, current_time: int) -> Dict:
        """
        Получить текущий статус буфера.
        
        Args:
            current_time: Текущее время в мс
            
        Returns:
            Dict: Статус буфера
        """
        return self.buffer.GET_STATUS(current_time)
    
    def GET_CH_QUALITY(self) -> Dict:
        """
        Получить текущее качество канала.
        
        Returns:
            Dict: Параметры качества канала
        """
        return {
            'cqi': self.cqi,
            'SINR': self.SINR,
            'distance': self.distance_to_bs #возможно оно тут нафиг не надо я пока не вставлял расчеты SINR
        }
    
    def SINR_TO_CQI(self, SINR: float) -> int:
        """
        Преобразовать SINR в CQI согласно спецификации LTE.
        Решил сделать не через множество elif, чтобы сократить код.
        Но можно вернуть твой метод.
        
        Args:
            SINR: Отношение сигнал/шум+помехи в dB
            
        Returns:
            int: Значение CQI (1-15)
        """
        if SINR <= -6.934:
            return 1
        elif SINR >= 22.976:
            return 15
        else:
            # Линейная интерполяция
            step = (22.976 + 6.934) / 14
            return int(1 + (SINR + 6.934) / step)

class Buffer:
    """
    Класс для моделирования буфера пользовательского устройства.
    """
    def __init__(self, max_size: int = 1048576):  # 1 MB по умолчанию
        """
        Инициализация буфера.
        
        Args:
            max_size: Максимальный размер буфера в байтах
        """
        self.max_size = max_size
        self.current_size = 0
        self.packets = []  # Список пакетов в буфере
        self.dropped_packets = 0  # Счетчик отброшенных пакетов
    
    def ADD_PACKET(self, packet_size: int, creation_time: int, priority: int = 0) -> bool:
        """
        Добавить пакет в буфер. Пока я понятия не имею, по каким моделям мы
        будем генерировать трафик и каким макаром, но сделал такую заглушку
        
        Args:
            packet_size: Размер пакета в байтах
            creation_time: Время создания пакета (в мс)
            priority: Приоритет пакета (0-10)
            
        Returns:
            bool: True, если пакет добавлен, False если отброшен
        """
        if self.current_size + packet_size > self.max_size:
            self.dropped_packets += 1
            return False
        
        packet = {
            'size': packet_size,
            'creation_time': creation_time,
            'priority': priority
        }
        
        self.packets.append(packet)
        self.current_size += packet_size
        return True
    
    def GET_PACKETS(self, max_bytes: int, current_time: int) -> Tuple[List[Dict], int]:
        """
        Получить пакеты из буфера для передачи.
        
        Args:
            max_bytes: Максимальное количество байт для передачи
            current_time: Текущее время (в мс)
            
        Returns:
            Tuple[List[Dict], int]: (список пакетов, общий размер в байтах)
        """
        # Сортировка пакетов по приоритету и времени ожидания
        sorted_packets = sorted(
            self.packets, 
            key=lambda p: (10 - p['priority'], current_time - p['creation_time']), 
            reverse=True
        )
        
        selected_packets = []
        total_size = 0
        
        for packet in sorted_packets:
            if total_size + packet['size'] <= max_bytes:
                selected_packets.append(packet)
                total_size += packet['size']
        
        # Удаляем выбранные пакеты из буфера
        for packet in selected_packets:
            self.packets.remove(packet)
            self.current_size -= packet['size']
        
        return selected_packets, total_size
    
    def GET_STATUS(self, current_time: int) -> Dict:
        """
        Получить статистику состояния буфера.
        
        Args:
            current_time: Текущее время (в мс)
            
        Returns:
            Dict: Статистика буфера
        """
        if not self.packets:
            return {
                'size': 0,
                'packet_count': 0,
                'oldest_packet_delay': 0,
                'average_delay': 0,
                'utilization': 0.0
            }
        
        delays = [current_time - packet['creation_time'] for packet in self.packets]
        
        return {
            'size': self.current_size,
            'packet_count': len(self.packets),
            'oldest_packet_delay': max(delays),
            'average_delay': sum(delays) / len(delays),
            'utilization': (self.current_size / self.max_size) * 100
        }
    
    def DESTROY_BUFFER(self):
        """
        Полностью очистить буфер.
        
        Удаляет все пакеты и сбрасывает текущий размер буфера. Вдруг пригодится
        
        Returns:
            None
        """
        self.packets.clear()
        self.current_size = 0

class UECollection:
    """
    Класс для управления коллекцией пользовательских устройств.
    Задел, чтобы не создавать их вручную и можно было регулировать количество.
    """
    def __init__(self, bs_position: Tuple[float, float] = (0, 0)):
        """
        Инициализация коллекции UE.
        
        Args:
            bs_position: Координаты базовой станции (x, y) в метрах
        """
        self.users = {}  # Словарь {UE_ID: UserEquipment}
        self.bs_position = bs_position
    
    def ADD_USER(self, ue: UserEquipment) -> bool:
        """
        Добавить пользователя в коллекцию.
        
        Args:
            ue: Объект пользовательского устройства
            
        Returns:
            bool: True, если пользователь добавлен, False если уже существует
        """
        if ue.UE_ID in self.users:
            return False
        
        self.users[ue.UE_ID] = ue
        return True
    
    def REMOVE_USER(self, UE_ID: int) -> bool:
        """
        Удалить пользователя из коллекции.
        
        Args:
            UE_ID: Идентификатор пользователя
            
        Returns:
            bool: True, если пользователь удален, False если не найден
        """
        if UE_ID in self.users:
            del self.users[UE_ID]
            return True
        return False
    
    def GET_USER(self, UE_ID: int) -> Optional[UserEquipment]:
        """
        Получить пользователя по ID.
        
        Args:
            UE_ID: Идентификатор пользователя
            
        Returns:
            UserEquipment или None, если пользователь не найден
        """
        return self.users.get(UE_ID)
    
    def GET_ALL_USERS(self) -> List[UserEquipment]:
        """
        Получить список всех пользователей.
        
        Returns:
            List[UserEquipment]: Список всех пользователей
        """
        return list(self.users.values())
    
    def UPDATE_ALL_USERS(self, time_ms: int):
        """
        Обновить состояние всех пользователей.
        
        Args:
            time_ms: Текущее время в миллисекундах
        """
        for ue in self.users.values():
            # Обновление позиции
            ue.UPD_POSITION(time_ms, self.bs_position)
            
            # Обновление качества канала
            ue.UPD_CH_QUALITY(time_ms, self.bs_position)
            
            # Генерация нового трафика
            ue.GEN_TRFFC(time_ms)
    
    def GET_ACTIVE_USERS(self) -> List[UserEquipment]:
        """
        Получить список активных пользователей (с данными в буфере).
        
        Returns:
            List[UserEquipment]: Список активных пользователей
        """
        return [ue for ue in self.users.values() 
                if ue.buffer.GET_STATUS(0)['size'] > 0]

def prepare_users_for_scheduler(ue_collection: UECollection, time_ms: int) -> List[Dict]:
    """
    Подготовить данные о пользователях для планировщика.Пока пример функции.
    Я не представляю, какой коннектор мы будем делать, но скорее всего он будет
    жить в коллекции пользователей. А коллекцию вынесем в ENVIRONMENT.
    
    Args:
        ue_collection: Коллекция пользователей
        time_ms: Текущее время в миллисекундах
        
    Returns:
        List[Dict]: Список словарей с данными о пользователях
    """
    users_data = []
    
    for ue in ue_collection.GET_ACTIVE_USERS():
        buffer_status = ue.GET_BUFFER_STATUS(time_ms)
        channel_quality = ue.GET_CH_QUALITY()
        
        users_data.append({
            'UE_ID': ue.UE_ID,
            'buffer_size': buffer_status['size'],
            'packet_count': buffer_status['packet_count'],
            'oldest_packet_delay': buffer_status['oldest_packet_delay'],
            'cqi': channel_quality['cqi'],
            'SINR': channel_quality['SINR'],
            'distance': channel_quality['distance'],
            'current_throughput': ue.current_throughput,
            'average_throughput': ue.average_throughput,
            'pf_metric': ue.CALC_PF_METRIC()
        })
    
    return users_data

