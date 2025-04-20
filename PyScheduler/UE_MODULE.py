"""
#------------------------------------------------------------------------------
# Модуль: UE_MODULE - Модель пользовательского устройства (UE) для сети LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет классы и методы для моделирования пользовательских устройств
# в сети LTE, включая их перемещение, генерацию трафика, управление буфером
# и оценку качества канала.
#
# Версия: 1.0.5
# Дата последнего изменения: 2025-04-06
# Автор: Брагин Кирилл, Норицин Иван
# Версия Python Kernel: 3.12.9
#
# Зависимости:
# - MOBILITY_MODEL.py (модели перемещения пользователей)
# - CHANNEL_MODEL.py (модели распространения сигнала)
# - TRAFFIC_MODEL.py (модели генерации трафика)
#
# Изменения:
#   v1.0.1 - 2025-03-20
#      - movement_model -> mobility_model
#      - Добавлены новые параметры при инициализации UE: 
#        velocity_min, velocity_max
#      - Добавлены параметры для работы моделей передвижения:
#        destination, is_paused, pause_timer, is_first_move
#      - Добавлены параметры для хранения координат пользователя:
#        x_coordinates, y_coordinates
#      - Разные примеры вызова функции mobility_model.update в UPD_POSITION
#      - Добавление новых координат UE в x_coordinates, y_coordinates
#
#   v1.0.2 - 2025-03-22
#      - Метод UPD_POSITION теперь подстраивается под модель перемещения
#      - Временно размещены функции визуализации перемещения пользователей и 
#        тестирования моделей перемещения
#      - Добавлены параметры для работы модели передвижения Gauss-Markov:
#        mean_velocity, mean_direction   
#
#   v1.0.3 - 2025-03-30
#      - Параметр расстояния от UE до базовой станции разбит на 4 отдельных:
#        dist_to_BS_2D, dist_to_BS_2D_in, dist_to_BS_2D_out, dist_to_BS_3D  
#      - Добавлена функция расчёта дистанций от UE до базовой станции:
#        _calculate_distances_to_BS    
#      - Добавлен новый параметр, обозначающий indoor/outdoor состояние UE
#      - Теперь параметры скоростей и направлений выбираются автоматически
#        в зависимости от типа UE (ue_class) при помощи новой функции:
#        _set_scenario_parameters
#      - Из модуля удалены функции визуализации перемещения пользователей и 
#        тестирования моделей перемещения
#
#   v1.0.4 - 2025-04-06
#      - Общий рефакторинг и код-ревью  
#      - Добавлена валидация для некоторых методов. Даны рекомендации по
#        по реализации следующих валидаций   
#      - Изменен буфер пользователя для подготовки интеграции с моделями
#       генерации трафика. Теперь буфер FIFO, но приоритеты оставлены
#       заглушкой для реализации гибридного буфера с учетом QoS
#      - Подготовка к реализации адаптивной кодовой модуляции
#
#   v1.0.5 - 2025-04-06
#      - Добавлен класс Packet. Логика буфера и тесты переписаны с учетом
#       новых реалий
#------------------------------------------------------------------------------
"""
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Union, Tuple
from MOBILITY_MODEL import RandomWalkModel, RandomWaypointModel, RandomDirectionModel, GaussMarkovModel
from CHANNEL_MODEL import RMaModel, UMaModel, UMiModel
from TRAFFIC_MODEL import PoissonModel, OnOffModel, MMPPModel
from BS_MODULE import BaseStation

class Packet:
    """Класс для представления сетевого пакета"""
    _id_counter = 0  # Счетчик для генерации уникальных ID
    
    def __init__(self, size: int, creation_time: int, priority: int = 0):
        if size <= 0:
            raise ValueError("Размер пакета должен быть > 0")
        if not (0 <= priority <= 10):
            raise ValueError("Приоритет должен быть 0-10")

        Packet._id_counter += 1
        self.id = Packet._id_counter  # Простой числовой идентификатор
        self.size = size
        self.creation_time = creation_time  # Время в мс (ожидается передача извне)
        self.priority = priority # для реализации QoS буфера нужно будет уйти от FIFO
        self.retry_count = 0 # Заготовка для HARQ

    @property
    def age(self, current_time: int) -> int:
        """Возраст пакета в мс (требует явной передачи текущего времени)"""
        return current_time - self.creation_time     

class Buffer:
    """
    Класс для моделирования буфера пользовательского устройства. Пока функционирует по логике
    FIFO. Есть костыль для приоритетов пакетов, но не раскрыт. Для реализации QoS буфера нужно будет уйти от FIFO
    """
    def __init__(self, max_size: int = 1048576):  # 1 MB по умолчанию
        """
        Инициализация буфера.
        
        Args:
            max_size: Максимальный размер буфера в байтах
        """
        self.max_size = max_size
        self.current_size = 0
        self.queue = deque()  # Теперь очередь для хранения пакетов
        self.dropped_packets = 0  # Для переполнения
        self.expired_packets = 0  # Для TTL
        self.dropped_info = []    # Параметры отброшенных пакетов
    
    def ADD_PACKET(self, packet_size: int, creation_time: int, current_time: int, 
              priority: int = 0, ttl_ms: int = 1000) -> bool:
        """
        Добавить пакет в буфер. Пока я понятия не имею, по каким моделям мы
        будем генерировать трафик и каким макаром, но сделал такую заглушку
        
        Args:
            Наследуется у класса Packet
            
        Returns:
            bool: True, если пакет добавлен, False если отброшен
        """
        # Шаг 1: Удаление устаревших пакетов перед добавлением
        self.queue = deque([
            p for p in self.queue 
            if current_time - p['creation_time'] <= ttl_ms
        ])
        
        # Обновление current_size после очистки
        self.current_size = sum(p['size'] for p in self.queue)
        
        # Шаг 2: Проверка на переполнение после очистки
        if self.current_size + packet_size > self.max_size:
            self.dropped_packets += 1
            self.dropped_info.append({
                'size': packet_size,
                'creation_time': creation_time,
                'priority': priority,
                'reason': 'overflow'
            })
            return False
        
        # Шаг 3: Добавление нового пакета
        self.queue.append({
            'size': packet_size,
            'creation_time': creation_time,
            'priority': priority
        })
        self.current_size += packet_size
        return True

        #@sherokiddo: "Возможно, у пакета появится атрибут метки QoS или приоритет
        # заглушку для него сделал. В дальнейшем реализовать функцию CHCK_PRIORITY или CHCK_PR
        # а также реализовать логику переполнения буфера и отбрасывания пакетов
        # а также, добавить возможность менять приоритет пакета через метод
        # а напоследок, метод для получения пакетов определенного приоритета GET_PCKT_BY_PR"

    def GET_PACKETS(self, max_bytes: int, bits_per_rb: int, current_time: int, ttl_ms: int = 1000) -> Tuple[List[Dict], int]:
        """
        Извлечение данных из буфера с фрагментацией.
        
        Args:
            max_bytes: Максимальный объём данных в байтах
            bits_per_rb: Количество бит на ресурсный блок
            
        Returns:
            Tuple[List[Dict], int]: (список пакетов/фрагментов, общий размер в байтах)
        """
        
        # Объединенная фильтрация и сбор статистики
        filtered = []
        expired = []
        
        for p in self.queue:
            if current_time - p['creation_time'] > ttl_ms:
                expired.append(p)
            else:
                filtered.append(p)
        
        # Обновление статистики
        self.expired_packets += len(expired)
        self.dropped_info.extend(expired)
        
        # Обновление очереди
        self.queue = deque(filtered)
        self.current_size = sum(p['size'] for p in self.queue)
        
        selected = []
        total_bits = 0
        max_bits = max_bytes * 8  # Конвертация в биты
        
        while self.queue and total_bits < max_bits:
            packet = self.queue[0]
            packet_bits = packet['size'] * 8
            
            # Доступное место в текущей итерации
            remaining_bits = max_bits - total_bits
            fragment_bits = min(packet_bits, remaining_bits)
            fragment_size = fragment_bits // 8  # Размер фрагмента в байтах
            
            if fragment_size >= packet['size']:
                # Весь пакет помещается
                selected_packet = self.queue.popleft()
                selected.append(selected_packet)
                total_bits += packet_bits
                self.current_size -= selected_packet['size']
            else:
                # Создание фрагмента
                fragment = {
                    'size': fragment_size,
                    'creation_time': packet['creation_time'],
                    'priority': packet['priority'],
                    'parent_id': id(packet)
                }
                selected.append(fragment)
                total_bits += fragment_bits
                
                # Обновление исходного пакета
                self.queue[0]['size'] -= fragment_size
                self.current_size -= fragment_size
        
        return selected, total_bits // 8

    def GET_STATUS(self, current_time: int) -> Dict:
        """
        Получить статистику состояния буфера.

        Args:
            current_time: Текущее время (в мс)

        Returns:
            Dict: Статистика буфера
        """
        if not self.queue:
            return {
                'size': 0,
                'packet_count': 0,
                'oldest_packet_delay': 0,
                'average_delay': 0.0,
                'utilization': 0.0
            }

        delays = [current_time - p['creation_time'] for p in self.queue]
        return {
            'size': self.current_size,
            'packet_count': len(self.queue),
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
        self.queue.clear()
        self.current_size = 0
        self.dropped_packets = 0
    #@sherokiddo: "Возможно пригодится метод для удаления пакетов, у которых
    # капнула уже большая задержка. типа REMOVE_OLD_PCKT


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
        
        # Параметры движения (настраиваются автоматически, в зависимости от сценария движения)
        self.velocity_min = 0.0 # Минимальная скорость в м/с
        self.velocity_max = 0.0 # Максимальная скорость в м/с
        self.mean_velocity = 0.0 # Средняя скорость (Для Gauss-Markov)
        self.mean_direction = 0.0 # Среднее направление (Для Gauss-Markov)
        #@sherokiddo пишет: "В рамках рефакторинга выпилить mean_velocity и mean_direction из атрибутов юзера
        #пусть живут в модели Гаусс-Маркова"
        self.is_indoor = False # Находится ли UE в помещении
        self._set_scenario_parameters()
        
        self.mobility_model = None  # Установить позже
        self.velocity = 0.0  # Скорость в м/с
        self.direction = 0.0  # Направление в радианах (ранее angle)
        
        self.destination = (0, 0) # Место назначения (для Random Waypoint)
        self.is_paused = True # Флаг, который обозначает стоит ли устройство на паузе
        self.pause_timer = 0.0 # Таймер паузы устройства
        self.is_first_move = True # Флаг первого движения устройства
  
        self.x_coordinates = [self.position[0]] # Координаты X для карты передвижения
        self.y_coordinates = [self.position[1]] # Координаты Y для карты передвижения
        
        # Буфер данных
        self.buffer = Buffer(buffer_size)
        
        # Модель трафика
        self.traffic_model = None  # Установить позже
        
        # Параметры канала связи
        self.channel_model = None  # Установить позже
        self.cqi = 1  # Текущий CQI (1-15)
        self.SINR = 0.0  # Текущее отношение сигнал/шум+помехи в dB
        
        self.SINR_values = [] # Временный параметр для демонстрации результатов
        self.CQI_values = [] # Временный параметр для демонстрации результатов
        
        self.UE_height = 0.0
        self.dist_to_BS_2D = 0.0  # 2D-Расстояние до базовой станции в метрах
        self.dist_to_BS_2D_in = 0.0 # 2D-Расстояние до БС в (часть помещения)
        self.dist_to_BS_2D_out = 0.0 # 2D-Расстояние до БС в (часть улицы)
        self.dist_to_BS_3D = 0.0 # 3D-Расстояние до базовой станции в метрах
        
        # Параметры для алгоритмов планирования
        self.current_throughput = 0.0  # Текущая пропускная способность (бит/с)
        self.average_throughput = 0.0  # Средняя пропускная способность (бит/с) 

        self.assigned_rbs = []  # Список выделенных ресурсных блоков
        self.mcs_index = 0  # Индекс MCS (0-28)
        
        # Статистика
        self.total_transmitted_bits = 0
        self.total_transmitted_packets = 0
        self.total_dropped_packets = 0
    
    def SET_MOBILITY_MODEL(self, model):
        """
        Установить модель движения пользователя. Метод можно будет вызвать
        глобально и по конкретному UE_ID
        
        Args:
            model: Объект модели движения
        """
        self.mobility_model = model
        
        #@sherokiddo: "А вот как валидировать метод, по корректному model...
        # если у нас нет единого парента для всех моделей...?"
    
    def SET_CH_MODEL(self, model):
        """
        Установить модель канала связи.
        
        Args:
            model: Объект модели канала
        """
        self.channel_model = model
        
        #@sherokiddo: "Абаюнда тому что в SET_MOBILITY_MODEL"
    
    def SET_TRAFFIC_MODEL(self, model):
        """
        Установить модель генерации трафика.
        
        Args:
            model: Объект модели трафика
        """
        self.traffic_model = model
        
        #@sherokiddo: "Предусмотреть валидацию"
    
    def UPD_POSITION(self, time_ms: int, bs_position: Tuple[float, float], bs_height: float,
                     indoor_boundaries: Tuple[float, float, float, float] = (0, 0, 0, 0)):
        """
        Обновить позицию пользователя согласно модели движения.
        
        Args:
            time_ms: Текущее время в миллисекундах
            bs_position: Координаты базовой станции (x, y) в метрах
        """       
        # Вызов функции update для модели Random Walk:
        if isinstance(self.mobility_model, RandomWalkModel):
            self.position, self.velocity, self.direction, self.is_first_move = self.mobility_model.update(
                self.position, self.velocity, self.velocity_min, self.velocity_max, self.direction, self.is_first_move, time_ms
            )
        
        # Вызов функции update для модели Random Waypoint:
        if isinstance(self.mobility_model, RandomWaypointModel):
            self.position, self.velocity, self.direction, self.destination, self.is_paused, self.pause_timer = self.mobility_model.update(
                self.position, self.velocity, self.velocity_min, self.velocity_max, self.direction,
                self.destination, self.is_paused, self.pause_timer, time_ms
            )
            
        # Вызов функции update для модели Random Direction:
        if isinstance(self.mobility_model, RandomDirectionModel):
            self.position, self.velocity, self.direction, self.destination, self.is_paused, self.pause_timer, self.is_first_move = self.mobility_model.update(
                self.position, self.velocity, self.velocity_min, self.velocity_max, self.direction,
                self.destination, self.is_paused, self.pause_timer, self.is_first_move, time_ms
            )   
            
        if isinstance(self.mobility_model, GaussMarkovModel):
            self.position, self.velocity, self.direction, self.mean_direction = self.mobility_model.update(
                self.position, self.velocity, self.direction, self.mean_velocity, self.mean_direction, time_ms
            )
            
        self.x_coordinates.append(self.position[0])
        self.y_coordinates.append(self.position[1])
        
        # Обновление 2D и 3D расстояний до базовой станции
        if self.is_indoor:
            self._calculate_distances_to_BS(bs_position, bs_height, indoor_boundaries)
            
        else:
            self.dist_to_BS_2D = np.hypot(self.position[0] - bs_position[0],
                                          self.position[1] - bs_position[1])
            
            self.dist_to_BS_2D_out = self.dist_to_BS_2D
            
            self.dist_to_BS_3D = np.hypot(self.dist_to_BS_2D, bs_height - self.UE_height)
            
    
    def UPD_CH_QUALITY(self):
        """
        Обновить качество канала согласно модели распространения.
        
        Args:
            time_ms: Текущее время в миллисекундах
            bs_position: Координаты базовой станции (x, y) в метрах
        """
        
        if not self.channel_model:
            raise ValueError("Ошибка! Модель канала не определена! {}".format(self.UE_ID))
            
        # @shrokiddo: "Добавил валидацию назначения модели, а то небыло"
        # рекомендую сделать валидацию вызова UPD_POSITION, потому что он
        # должен вызываться раньше UPD_CH_QUALITY 
        # (сначала получаем координаты а потом качество канала), а это неявно
        # а если вдруг координаты не менялись, можно упростить расчеты и просто
        # дублировать предыдущий SINR...хотя...зачем тогда модели.
        # вариант чисто на подумать
        
        if isinstance(self.channel_model, RMaModel):
            if self.UE_height == 0.0:
                if self.is_indoor == True:
                    self.UE_height = np.random.uniform(1, 10)
                else:
                    self.UE_height = 1.0
            
            self.SINR = self.channel_model.calculate_SINR(
                self.dist_to_BS_2D, self.dist_to_BS_2D_in, self.dist_to_BS_3D, self.UE_height, self.ue_class
            )
            
        if isinstance(self.channel_model, UMaModel):
            if self.UE_height == 0.0:
                if self.is_indoor == True:
                    N_fl = np.random.uniform(4, 8)
                    n_fl = np.random.uniform(1, N_fl)
                    self.UE_height = 3 * (n_fl - 1) + 1.5
                else:
                    self.UE_height = 1.5
                    
            self.SINR = self.channel_model.calculate_SINR(
                self.dist_to_BS_2D, self.dist_to_BS_2D_in, self.dist_to_BS_3D, self.UE_height, self.ue_class
            )
            
        if isinstance(self.channel_model, UMiModel):
            if self.UE_height == 0.0:
                if self.is_indoor == True:
                    N_fl = np.random.uniform(4, 8)
                    n_fl = np.random.uniform(1, N_fl)
                    self.UE_height = 3 * (n_fl - 1) + 1.5
                else:
                    self.UE_height = 1.5
                    
            self.SINR = self.channel_model.calculate_SINR(
                self.dist_to_BS_2D, self.dist_to_BS_2D_in, self.dist_to_BS_3D, self.UE_height, self.ue_class
            )
        
        self.cqi = self.SINR_TO_CQI(self.SINR)
                
        self.SINR_values.append(self.SINR)
        self.CQI_values.append(self.cqi)
        
        #@sherokiddo пишет: "Ваня, я понимаю, что в моделях передвижения юзеров разные вводы и выводы.
        # Но блин тут то для всех моделей одинаково всё! Не по чистокоду. В рамках рефакторинга
        # сделай единый метод вызова для любой модели (желательно в MOBILITY_MODEL)
        # тут же столько оптимизировать можно...Подумай на досуге в общем"
    
    def GEN_TRFFC(self, current_time: int, update_interval: int) -> None:
        """
        Генерирует трафик и добавляет пакет в буфер с автоматической очисткой старых пакетов.
        
        Args:
            current_time: Текущее время в мс (используется для TTL)
        """
        
        if not self.traffic_model:
            raise ValueError("Ошибка! Модель генерации трафика не определена! {}".format(self.UE_ID))
        
        if isinstance(self.traffic_model, PoissonModel):
            packets = self.traffic_model.generate_traffic(
                current_time, update_interval
                )
            
        if isinstance(self.traffic_model, OnOffModel) or isinstance(self.traffic_model, MMPPModel):
            packets = self.traffic_model.generate_traffic(
                self.UE_ID, current_time, update_interval
                )
        
        # Скорость поступления пакетов в буфер
        total_bytes = sum(p['size'] for p in packets)
        total_bits = total_bytes * 8
        interval_seconds = update_interval / 1000
        bitrate = total_bits / interval_seconds if interval_seconds > 0 else 0    
            
        for packet in packets:
            if not self.buffer.ADD_PACKET(
                packet_size=packet['size'],
                creation_time=packet['creation_time'],
                current_time=current_time,
                priority=packet['priority'],
                ttl_ms=1000
            ):
                self.total_dropped_packets += 1
                print(f"UE {self.UE_ID}: Пакет {packet['size']}B отброшен (буфер полный)")
            
        status = self.buffer.GET_STATUS(current_time)
        print(f"\nИнтервал [{current_time-update_interval}-{current_time} ms]: Создано {len(packets)} пакетов")
        print(f"Скорость поступления: {bitrate:.2f} бит/с")
        print(f"Статус буфера: {status}")

    
    def UPD_THROUGHPUT(self, bits_transmitted: int, time_interval_ms: int):
        """
        Обновить статистику пропускной способности.
        
        Args:
            bits_transmitted: Количество переданных бит
            time_interval_ms: Интервал времени в мс
        """
        # Текущая пропускная способность в бит/с
        self.current_throughput = (bits_transmitted * 1000) / time_interval_ms if time_interval_ms > 0 else 0
        
        # Обновление общей статистики
        self.total_transmitted_bits += bits_transmitted
    
    def UPD_BUFFER(self, current_time: int):
        """Обновление задержки пакетов в буфере"""
        for packet in self.buffer.queue:
            packet['age'] = current_time - packet['creation_time']
        
        # Удаление устаревших пакетов (пример: TTL = 1000 мс)
        self.buffer.queue = deque([
            p for p in self.buffer.queue
            if p['age'] <= 1000
        ])

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
            'distance': self.dist_to_BS_2D #возможно оно тут нафиг не надо я пока не вставлял расчеты SINR
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
        
    def _calculate_distances_to_BS(self, bs_position: Tuple[float, float], bs_height: float,
                                   indoor_boundaries: Tuple[float, float, float, float]) -> None:
        """
        Вычисляет расстояние от пользователя до базовой станции с учетом нахождения внутри здания.
        Разделяет расстояние на часть внутри здания (indoor) и снаружи (outdoor).
    
        Args:
            bs_position: Координаты базовой станции (x, y) в метрах.
            bs_height: Высота антенны базовой станции над землёй в метрах.
            indoor_boundaries: Границы здания в формате (x_min, y_min, x_max, y_max).
        """
        x_min, y_min, x_max, y_max = indoor_boundaries
        
        ue_x, ue_y = self.position
        bs_x, bs_y = bs_position
        
        if (x_min <= bs_x <= x_max) and (y_min <= bs_y <= y_max):
            distance = np.hypot(bs_x - ue_x, bs_y - ue_y)
            self.dist_to_BS_2D = distance
            self.dist_to_BS_2D_in = distance
            self.dist_to_BS_2D_out = 0.0
            return
        
        dx = bs_x - ue_x
        dy = bs_y - ue_y
        
        t_values = []
        
        if dx != 0:
            t_x_min = (x_min - ue_x) / dx
            t_x_max = (x_max - ue_x) / dx
            t_values.extend([t_x_min, t_x_max])
            
        if dy != 0:
            t_y_min = (y_min - ue_y) / dy
            t_y_max = (y_max - ue_y) / dy
            t_values.extend([t_y_min, t_y_max])
            
        t_valid = [t for t in t_values if t > 0]
        
        if not t_valid:
            self.dist_to_BS_2D = np.hypot(dx, dy)
            self.dist_to_BS_2D_in = self.dist_to_BS_2D
            self.dist_to_BS_2D_out = 0.0
            return
        
        t_exit = min(t_valid)
    
        exit_x = ue_x + dx * t_exit
        exit_y = ue_y + dy * t_exit
        
        d_in = np.hypot(exit_x - ue_x, exit_y - ue_y)
        d_total = np.hypot(dx, dy)
        d_out = d_total - d_in

        self.dist_to_BS_2D = d_total
        self.dist_to_BS_2D_in = d_in
        self.dist_to_BS_2D_out = d_out
        self.dist_to_BS_3D = np.hypot(self.dist_to_BS_2D, bs_height - self.UE_height)

    def _set_scenario_parameters(self):
        
        if self.ue_class == "indoor":
            self.velocity_min = 0.0
            self.velocity_max = 1.0
            self.mean_velocity = 0.5
            self.mean_direction = np.random.randint(0, 360)
            self.is_indoor = True
        
        elif self.ue_class == "pedestrian":
            self.velocity_min = 0.5
            self.velocity_max = 1.7
            self.mean_velocity = 1.2
            self.mean_direction = np.random.randint(0, 360)
            self.is_indoor = False
            
        elif self.ue_class == "cyclist":
            self.velocity_min = 2.0
            self.velocity_max = 5.5
            self.mean_velocity = 3.9
            self.mean_direction = np.random.randint(0, 360)
            self.is_indoor = False
            
        elif self.ue_class == "car":
            self.velocity_min = 0.0
            self.velocity_max = 16.7
            self.mean_velocity = 11.1
            self.mean_direction = np.random.randint(0, 360)
            self.is_indoor = False
            
        else:
            raise ValueError("Недопустимое значение типа передвижения устройства!")
        # @sherokiddo пишет: "В рамках рефакторинга выпилить mean_velocity и mean_direction из атрибутов юзера
        # пусть живут в модели Гаусс-Маркова.
        # атрибут self.mean_direction = np.random.randint(0, 360) сделать глобальным, чего тут его дублировать
        # можно было сделать через словари, но так тоже лаконично. с точки зрения оптимизации еще вариантов не знаю"
        
class UECollection:
    """
    Класс для управления коллекцией пользовательских устройств.
    Задел, чтобы не создавать их вручную и можно было регулировать количество.
    """
    def __init__(self):
        """
        Инициализация коллекции UE.
        
        Args:
            bs_position: Координаты базовой станции (x, y) в метрах
        """
        self.users = {}  # Словарь {UE_ID: UserEquipment}
    
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
    
    def UPDATE_ALL_USERS(self, time_ms: int, update_interval: int, 
                         bs_position: Tuple[float, float], bs_height: float, 
                         indoor_boundaries: Tuple[float, float, float, float] = (0, 0, 0, 0)):
        """
        Обновить состояние всех пользователей.
        
        Args:
            time_ms: Текущее время в миллисекундах
        """
        for ue in self.users.values():
            # Обновление позиции
            ue.UPD_POSITION(update_interval, bs_position, bs_height, indoor_boundaries)
            
            # Обновление качества канала
            ue.UPD_CH_QUALITY()
            
            # Генерация нового трафика
            ue.GEN_TRFFC(time_ms, update_interval)
    
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

# Далее тесты для проверки работоспособности буфера и примеры работы с ним. 
# Можно удалить или закомментить после того, как будут сделаны генераторы трафика.
def test_buffer_fifo():
    """
    Расширенный тест работы буфера с проверкой задержек
    """
    print("\n=== Начало теста буфера ===")
    
    # 1. Инициализация буфера
    buffer = Buffer(max_size=10000)
    current_time = 1000
    
    # 2. Добавление пакетов с разным временем создания
    buffer.ADD_PACKET(2000, 1000, current_time - 1500)  # Задержка 1500 мс
    buffer.ADD_PACKET(3000, 1000, current_time - 500)  # Задержка 500 мс
    buffer.ADD_PACKET(5000, 1000, current_time - 100)  # Задержка 100 мс
    
    # 3. Проверка статуса
    status = buffer.GET_STATUS(current_time)
    print(f"Статус после добавления: {status}")
    
    # 4. Извлечение пакетов (добавлен параметр bits_per_rb)
    bits_per_rb = 8000  # Пример: 1000 байт на RB
    packets, total = buffer.GET_PACKETS(6000, bits_per_rb, current_time)
    print(f"\nИзвлечено {len(packets)} пакетов ({total} байт):")
    for i, p in enumerate(packets, 1):
        print(f"Пакет {i}: размер={p['size']} задержка={current_time - p['creation_time']} мс")
    
    # 5. Проверка статуса после извлечения
    status = buffer.GET_STATUS(current_time)
    print(f"\nСтатус после извлечения: {status}")
    
    # 6. Проверка переполнения
    print("\nПопытка переполнения буфера:")
    for size in [4000, 3000, 5000]:
        success = buffer.ADD_PACKET(size, 1000, current_time)
        print(f"Пакет {size} байт: {'успех' if success else 'переполнение'}")
    
    # 7. Очистка буфера
    buffer.DESTROY_BUFFER()
    print(f"\nПосле очистки: размер={buffer.current_size}, пакетов={len(buffer.queue)}")
    print("\nОтброшенные пакеты:")
    for p in buffer.dropped_info:
        age = current_time - p['creation_time']
        print(f"Размер: {p['size']}B, Приоритет: {p['priority']}, Задержка: {age} мс")

if __name__ == "__main__":
    test_buffer_fifo()