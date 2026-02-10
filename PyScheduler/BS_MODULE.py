"""
#------------------------------------------------------------------------------
# Модуль: BS_MODULE - Модель базовой станции (BS) для сети LTE
#------------------------------------------------------------------------------
# Описание:
#   Модуль содержит класс BaseStation, реализующий модель базовой станции LTE.
#   Предоставляет параметры конфигурации и характеристики базовой станции,
#   включая мощность передачи, антенные параметры и частотные характеристики.
#
# Версия: 1.1.0
# Дата последнего изменения: 2026-01-23
# Автор: Норицин Иван, Дворников Андрей
# Версия Python Kernel: 3.12.9
# v.1.1.0:
# - Удален Packet, перенесен в TRAFFIC_MODEL.py
#------------------------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import GLOBALS
import numpy as np
from TRAFFIC_MODEL import Packet
from UE_MODULE import UserEquipment

@dataclass
class BufferStatus:
    """
    """
    ue_id: int
    buffer_size: int # Байты
    timestamp: int
    lcid: Optional[int] = None
    qci: Optional[int] = None
    priority: int = 0
    hol_delay: Optional[int] = None
    
    def __post_init__(self):
        """
        """
            
        if self.buffer_size < 0:
            raise ValueError(
                f"The buffer size cannot be negative. "
                f"The obtained value: {self.buffer_size}"
            )
            
    def is_empty(self) -> bool:
        """
        """
        return self.buffer_size == 0
    
    def to_dict(self) -> Dict:
        """
        """
        return {
            'ue_id': self.ue_id,
            'buffer_size': self.buffer_size,
            'timestamp': self.timestamp,
            'lcid': self.lcid,
            'qci': self.qci,
            'priority': self.priority,
            'hol_delay': self.hol_delay,
        }


class SimpleBuffer:
    """
    """
    def __init__(self, ue_id: int, max_size: int = 1000000):
        """
        """
        self.ue_id = ue_id
        self.max_size = max_size # байты
        self.buffer = deque()
        self.current_size = 0 # байты
        
        self.total_packets_added = 0
        self.total_packets_dropped = 0
        self.total_packets_expired = 0
        
    def add_packet(self, packet: Packet) -> bool:
        """
        """
        if self.current_size + packet.size > self.max_size:
            self.total_packets_dropped += 1
            return False
        
        self.buffer.append(packet)
        self.current_size += packet.size
        
        self.total_packets_added += 1
        
        return True
    
    def get_packets(self, num_bytes: int) -> Tuple[List[Packet], int]:
        """
        """
        extracted_packets = []
        extracted_bits = 0
        num_bits_to_extract = GLOBALS.bytes_to_bits(num_bytes)
        
        while self.buffer and extracted_bits < num_bits_to_extract:
            packet = self.buffer[0]
            packet_size_bits = GLOBALS.bytes_to_bits(packet.size)
            
            if (extracted_bits + packet_size_bits) <= num_bits_to_extract:
                extracted_packet = self.buffer.popleft()
                extracted_packets.append(extracted_packet)
                extracted_bits += packet_size_bits
                
            else:
                remaining_bits = num_bits_to_extract - extracted_bits
                fragment_size = remaining_bits // 8
                
                fragment = Packet(
                    size=fragment_size, 
                    ue_id=packet.ue_id, 
                    creation_time=packet.creation_time,
                    qci=packet.qci,
                    traffic_type=packet.traffic_type,
                    priority=packet.priority,
                    ttl_ms=packet.ttl_ms,
                    deadline=packet.deadline,
                    is_fragment=True,
                    bearer_id=packet.bearer_id,
                )
                
                packet.size -= fragment_size
                packet.creation_time = GLOBALS.CURRENT_TIME
                
                extracted_packets.append(fragment)
                extracted_bits += GLOBALS.bytes_to_bits(fragment_size)
                break
            
        extracted_bytes = GLOBALS.bits_to_bytes(extracted_bits)
        self.current_size -= extracted_bytes
            
        return extracted_packets, extracted_bytes
    
    def upd_buffer(self) -> None:
        """
        """
        valid_packets = []
        expired_count = 0
        
        for packet in self.buffer:
            if packet.deadline > GLOBALS.CURRENT_TIME:
                valid_packets.append(packet)
            else:
                expired_count += 1
                
        self.buffer = deque(valid_packets)
        self.current_size = max(0, sum(p.size for p in valid_packets))
        
        self.total_packets_expired += expired_count
            
    
    def get_buffer_status(self) -> BufferStatus:
        """
        """
        return BufferStatus(
            ue_id=self.ue_id, 
            buffer_size=self.current_size, 
            timestamp=GLOBALS.CURRENT_TIME
        )
    
    def clear_buffer(self) -> None:
        """
        """
        self.buffer.clear()
        self.current_size = 0
        
        self.total_packets_added = 0
        self.total_packets_dropped = 0
        self.total_packets_expired = 0
        

class IBufferManager(ABC):
    """
    """
    @abstractmethod
    def add_packet(self, ue_id: int, packet: Packet) -> bool:
        """
        """
        pass
    
    @abstractmethod
    def get_buffer_status(self, ue_id: int) -> List[BufferStatus]:
        """
        """
        pass
    
    @abstractmethod
    def get_packets(self, grants: List) -> Tuple[List[Packet], int]:
        """
        """
        pass
    
    @abstractmethod
    def upd_buffers_all(self) -> None:
        """
        """
        pass
    
    @abstractmethod
    def create_ue_buffer(self, ue_id: int, max_size: Optional[int] = None) -> None:
        """
        """
        pass
    
    @abstractmethod
    def remove_ue_buffer(self, ue_id: int) -> None:
        """
        """
        pass
       

class SimpleBufferManager(IBufferManager):
    """
    """
    def __init__(self, default_max_size: int = 1000000):
        """
        """
        self.buffers: Dict[int, SimpleBuffer] = {}
        self.default_max_size = default_max_size
        
    def add_packet(self, ue_id: int, packet: Packet) -> bool:
        """
        """
        if ue_id not in self.buffers:
            raise ValueError(
                f"UE {ue_id} does not have a buffer. The buffer must "
                f"have been created during UE registration at the BS"
            )
            
        return self.buffers[ue_id].add_packet(packet)
        
    def get_buffer_status(self, ue_id: int) -> List[BufferStatus]:
        """
        """
        if ue_id not in self.buffers:
            return None
        
        buffer_status = self.buffers[ue_id].get_buffer_status()
        return [buffer_status]
    
    def get_packets(self, grants: List) -> Tuple[List[Packet], int]:
        """
        """
        if len(grants) != 1:
            raise ValueError(
                "error"
            )
            
        grant = grants[0]
        ue_id = grant.ue_id
        num_bytes = grant.num_bytes
        
        packets, extracted_bytes = self.buffers[ue_id].get_packets(num_bytes)
        
        return packets, extracted_bytes
    
    def upd_buffers_all(self) -> None:
        """
        """
        for buffer in self.buffers.values():
            buffer.upd_buffer()
    
    def create_ue_buffer(self, ue_id: int, max_size: Optional[int] = None) -> None:
        """
        """
        buffer_size = self.default_max_size if max_size is None else max_size
        self.buffers[ue_id] = SimpleBuffer(ue_id=ue_id, max_size=buffer_size)
        
    def remove_ue_buffer(self, ue_id: int) -> None:
        """
        """
        if ue_id in self.buffers:
            self.buffers[ue_id].clear_buffer()
            del self.buffers[ue_id]


class Buffer:
    """
    Класс для моделирования буфера базовой станции (DL). Пока функционирует по логике
    FIFO. Есть костыль для приоритетов пакетов, но не раскрыт. Для реализации QoS буфера нужно будет уйти от FIFO
    """

    def __init__(self, global_max: int = 1048576, per_ue_max: int = 262144):
        """
        Инициализация буфера.

        Args:
            max_size: Максимальный размер буфера в байтах
        """
        if per_ue_max > global_max:
            raise ValueError("per_ue_max не может превышать global_max")
        self.global_max = global_max
        self.per_ue_max = per_ue_max
        self.total_size = 0
        self.queues = defaultdict(deque)  # {ue_id: очередь пакетов}
        self.sizes = defaultdict(int)  # {ue_id: текущий размер}
        self.dropped = defaultdict(int)  # {ue_id: счетчик отброшенных}
        self.expired = defaultdict(int)  # {ue_id: счетчик устаревших}
        self.dropped_info = defaultdict(list)
        self.ingress_stats = defaultdict(
            lambda: {
                "total_bytes": 0,
                "start_time": None,
            }
        )

    def ADD_PACKET(self, packet: Packet, current_time: int) -> bool:
        """
        Добавить пакет в буфер БС в очередь конкретного UE_ID. Пока я понятия не имею, по каким моделям мы
        будем генерировать трафик и каким макаром, но сделал такую заглушку

        Args:
            packet: Объект Packet для добавления
            current_time: Текущее время симуляции (мс)

        Returns:
            bool: True, если пакет добавлен, False если отброшен
        """

        # Шаг 1: Удаление устаревших пакетов перед добавлением
        original_queue = self.queues[packet.ue_id]
        valid_packets = []
        expired_count = 0

        for p in original_queue:
            if current_time - p.creation_time <= p.ttl_ms:
                valid_packets.append(p)
            else:
                expired_count += 1

        # Обновление данных буфера сразу после фильтрации
        self.queues[packet.ue_id] = deque(valid_packets)
        self.sizes[packet.ue_id] = sum(p.size for p in valid_packets)
        self.total_size = sum(self.sizes.values())
        self.expired[packet.ue_id] += expired_count

        # Шаг 2: Проверка на переполнение после очистки
        current_ue_size = self.sizes[packet.ue_id]
        reject_reason = None

        if current_ue_size + packet.size > self.per_ue_max:
            reject_reason = "ue_limit"
        elif self.total_size + packet.size > self.global_max:
            reject_reason = "global_limit"

        if reject_reason:
            self.dropped[packet.ue_id] += 1
            self.dropped_info[packet.ue_id].append(
                {
                    "size": packet.size,
                    "creation_time": packet.creation_time,
                    "priority": packet.priority,
                    "reason": reject_reason,
                }
            )
            return False

        # Шаг 3: Обновление статистики скорости
        stats = self.ingress_stats[packet.ue_id]
        if not stats["start_time"]:
            stats["start_time"] = current_time
        stats["total_bytes"] += packet.size
        stats["last_update"] = current_time

        # Шаг 4: Добавление пакета с обновлением размеров
        self.queues[packet.ue_id].append(packet)
        self.sizes[packet.ue_id] += packet.size
        self.total_size += packet.size

        return True

        # @sherokiddo: "Возможно, у пакета появится атрибут метки QoS или приоритет
        # заглушку для него сделал. В дальнейшем реализовать функцию CHCK_PRIORITY или CHCK_PR
        # а также реализовать логику переполнения буфера и отбрасывания пакетов
        # а также, добавить возможность менять приоритет пакета через метод
        # а напоследок, метод для получения пакетов определенного приоритета GET_PCKT_BY_PR"

    def GET_PACKETS(
        self, ue_id: int, max_bytes: int, bits_per_rb: int, current_time: int
    ) -> Tuple[List[Packet], int]:
        """
        Извлечение данных из буфера с фрагментацией.

        Args:
            ue_id: ID пользователя
            max_bytes: Максимальный объём данных в байтах
            bits_per_rb: Количество бит на ресурсный блок
            current_time: Текущее время симуляции (мс)

        Returns:
            Tuple[List[Packet], int]: (список пакетов/фрагментов, общий размер в байтах)
        """

        # 1. Предварительная очистка буфера от устаревших пакетов
        self.queues[ue_id] = deque(
            [p for p in self.queues[ue_id] if (current_time - p.creation_time) <= p.ttl_ms]
        )

        # 2. Инициализация структур данных
        selected = []
        total_bits = 0
        max_bits = max_bytes * 8  # Конвертация в биты
        extracted_size = 0

        # 3. Основной цикл извлечения
        while self.queues[ue_id] and total_bits < max_bits:
            packet = self.queues[ue_id][0]
            packet_size_bits = packet.size * 8

            # 3.1. Полное извлечение пакета
            if (total_bits + packet_size_bits) <= max_bits:
                selected_packet = self.queues[ue_id].popleft()
                selected.append(selected_packet)
                total_bits += packet_size_bits
                extracted_size += selected_packet.size

            # 3.2. Фрагментация пакета
            else:
                remaining_bits = max_bits - total_bits
                fragment_size = remaining_bits // 8

                # Создание фрагмента с наследованием параметров
                fragment = Packet(
                    size=fragment_size,
                    ue_id=ue_id,
                    creation_time=packet.creation_time,
                    priority=packet.priority,
                    ttl_ms=packet.ttl_ms,
                    is_fragment=True,
                )

                # Модификация исходного пакета
                packet.size -= fragment_size
                packet.creation_time = current_time  # Обновление времени для TTL

                # Обновление статистики
                selected.append(fragment)
                total_bits += fragment_size * 8
                extracted_size += fragment_size
                break

        # 4. Корректное обновление буфера
        self.sizes[ue_id] -= extracted_size
        self.total_size -= extracted_size

        # 5. Точный расчет без округления
        exact_bytes = total_bits // 8

        return selected, exact_bytes

    # @sherokiddo: тут мог закрасться какой-то баг, но
    # меня еще надо убедить в этом

    def GET_UE_STATUS(self, current_time: int) -> Dict:
        """
        Получить статистику состояния буфера.

        Args:
            current_time: Текущее время (в мс)

        Returns:
            Dict: {
                'total_size': общий размер данных в буфере (байты),
                'total_packets': общее количество пакетов,
                'per_ue': {
                    ue_id: {
                        'size': размер данных (байты),
                        'packet_count': количество пакетов,
                        'oldest_delay': макс. задержка (мс),
                        'avg_delay': средняя задержка (мс),
                        'dropped': отброшено пакетов
                    }
                }
            }
        """
        status = {"total_size": 0, "total_packets": 0, "total_expired": 0, "per_ue": {}}

        for ue_id in self.queues:
            queue = self.queues[ue_id]
            if not queue:
                status["per_ue"][ue_id] = {
                    "size": 0,
                    "packet_count": 0,
                    "oldest_delay": 0,
                    "avg_delay": 0.0,
                    "dropped": self.dropped.get(ue_id, 0),
                    "expired": self.expired.get(ue_id, 0),
                    "packets": 0,
                    "bitrate": 0.0,
                }
                continue

            # Статистика для конкретного UE
            delays = [current_time - p.creation_time for p in queue]
            ue_status = {
                "size": self.sizes[ue_id],
                "packet_count": len(queue),
                "oldest_delay": max(delays) if delays else 0,
                "avg_delay": sum(delays) / len(delays) if delays else 0.0,
                "dropped": self.dropped[ue_id],
                "expired": self.expired[ue_id],
                "ingress_bytes": self.ingress_stats[ue_id]["total_bytes"],
                "packets": len(queue),
            }

            # Агрегированная статистика
            status["per_ue"][ue_id] = ue_status
            status["total_size"] += ue_status["size"]
            status["total_packets"] += ue_status["packet_count"]
            status["total_expired"] += ue_status["expired"]
            time_interval = current_time - self.ingress_stats[ue_id]["start_time"]
            ue_status["ingress_rate_bps"] = (
                (ue_status["ingress_bytes"] * 8 / time_interval) * 1000 if time_interval > 0 else 0
            )

            status["per_ue"][ue_id] = ue_status

        return status

    def DESTROY_UE_PACKETS(self, ue_id: int) -> None:
        """
        Полностью очистить буфер от указанного пользователя.

        Удаляет все пакеты и сбрасывает текущий размер буфера. Вдруг пригодится

        Returns:
            None
        """
        if ue_id in self.queues:
            self.queues[ue_id].clear()
            self.sizes[ue_id] = 0
            self.dropped[ue_id] = 0
            self.expired[ue_id] = 0

    def UPD_UE_BUFFER(self, ue_id: int, current_time: int) -> int:
        """
        Обновление буфера конкретного UE: удаление устаревших пакетов.

        Args:
            ue_id: ID пользователя
            current_time: Текущее время симуляции (мс)

        Returns:
            Количество удалённых устаревших пакетов
        """
        queue = self.queues[ue_id]
        valid_packets = []
        expired_count = 0

        # Фильтрация пакетов по TTL
        for packet in queue:
            if (current_time - packet.creation_time) >= packet.ttl_ms:
                expired_count += 1
            else:
                valid_packets.append(packet)

        # Обновление данных буфера
        self.queues[ue_id] = deque(valid_packets)
        self.sizes[ue_id] = max(0, sum(p.size for p in valid_packets))
        self.expired[ue_id] += expired_count

        return expired_count

    def get_ingress_speed_mbps(self, ue_id: int, current_time: int) -> float:
        stats = self.ingress_stats[ue_id]
        total_bytes = stats["total_bytes"]
        start_time = stats.get("start_time", None)
        if start_time is None:
            return 0.0
        delta_time_ms = current_time - start_time
        if delta_time_ms <= 0:
            return 0.0
        return (total_bytes * 8) / delta_time_ms * 0.001


class BaseStation:
    """
    Класс базовой станции LTE.

    Содержит конфигурационные параметры и методы для работы с характеристиками
    базовой станции в моделировании сетей LTE.
    """

    # Мощность передачи для макросотовых станций (дБм) по полосам пропускания (МГц)
    MACROCELL_TX_POWER = {1.4: 39, 3: 41, 5: 43, 10: 44, 15: 45, 20: 46}

    # Мощность передачи для микросотовых станций (дБм) по полосам пропускания (МГц)
    MICROCELL_TX_POWER = {1.4: 30, 3: 32, 5: 34, 10: 36, 15: 37, 20: 38}

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        height: float = 35.0,
        frequency_GHz: float = 1.8,
        bandwidth: float = 10,
        global_max: int = 1048576,
        per_ue_max: int = 262144,
        ch_model_type: str = None,
        ch_model_params: dict = None,
        enable_tdl: bool = False,
    ):
        """
        Инициализация базовой станции.
        #TODO: Перейти на фабричный паттерн реализации


        Args:
            x: Координата X расположения станции
            y: Координата Y расположения станции
            height: Высота установки антенны (метры)
            frequency_GHz: Рабочая частота (ГГц)
            bandwidth: Полоса пропускания (МГц)
        """
        self.position = (x, y)  # Позиция станции (x, y)
        self.height = height  # Высота антенны

        # Частотные параметры
        self.frequency_GHz = frequency_GHz  # Частота в ГГц
        self.frequency_Hz = frequency_GHz * 1e9  # Частота в Гц
        self.bandwidth = bandwidth  # Полоса пропускания
        self.rb_per_slot = GLOBALS.BANDWIDTH_TO_RB[bandwidth]

        # Характеристики передачи
        self.tx_power = None  # Мощность передачи (устанавливается отдельно)
        self.antenna_gain = 15  # Коэффициент усиления антенны (дБи)

        # Апдейт по буферу
        self.global_max = global_max
        self.per_ue_max = per_ue_max
        self.ue_buffers = defaultdict(Buffer)

        # Связь с моделью канала
        self.ch_model_type = ch_model_type
        self.channel_model = None
        # TODO: enable_tdl необходимо заменить на Strategy-паттерн
        self.enable_tdl = enable_tdl
        self.registered_ues = {}

        if ch_model_type is not None:
            self._init_channel_model(ch_model_params or {})

    def _init_channel_model(self, params: dict) -> None:
        """
        Инициализация модели канала для базовой станции.
        Использует ленивый импорт для избежания циклических зависимостей.
        (от ленивого импорта можно избавиться)


        Args:
            params: Параметры для конкретной модели канала
                - RMa: W (ширина улицы), h (высота здания), cond_update_period
                - UMa: cond_update_period, o2i_model
                - UMi: cond_update_period, o2i_model


        Raises:
            ValueError: Если указан неизвестный тип модели
        """
        # Ленивый импорт (избегаем циклических зависимостей)
        from CHANNEL_MODEL import ChannelInterface

        self.channel_model = ChannelInterface(bs=self).create(
            cond=self.ch_model_type,
            cond_update_period=params.get("cond_update_period", 0.0),
            o2i_model=params.get("o2i_model", "low"),
            freq_fad_nlos_model=params.get("freq_fad_nlos_model", "TDL-A"),
            freq_fad_los_model=params.get("freq_fad_los_model", "TDL-D"),
            ds_profile=params.get("ds_profile", "normal"),
            los_arrival_angle=params.get("los_arrival_angle", np.pi / 4),
            **{k: v for k, v in params.items() if k in ("W", "h")},
        )

    def REG_UE(self, ue: UserEquipment):
        """
        Регистрация пользователя на базовой станции.


        Выполняет:
        - Создание буфера для DL данных пользователя
        - Привязку модели трафика пользователя
        - Сохранение ссылки на объект UE (для расчета SINR)
        - Установку обратной связи UE -> BS (для доступа к модели канала)


        Args:
            ue: Объект UserEquipment для регистрации
        """
        self.ue_buffers[ue.UE_ID] = Buffer(global_max=self.global_max, per_ue_max=self.per_ue_max)
        self.registered_ues[ue.UE_ID] = ue
        ue.serving_bs = self

        # TODO: сделать метод DEREG_UE и сопутствующие изменения

    def SET_TRAFFIC_MODEL(self, ue: UserEquipment, model):
        """
        Установить модель генерации трафика для конкретного пользователя.

        Args:
            ue: объект UserEquipment
            model: объект модели трафика
        """
        ue.SET_TRAFFIC_MODEL(model)

        # @sherokiddo: "Предусмотреть валидацию"

    def UPD_GLOBAL_BUFFER(self, current_time: int) -> None:
        """
        Глобальное обновление всех буферов пользователей.
        Вызывает UPD_UE_BUFFER для каждого зарегистрированного UE.
        """
        for ue_id, buffer in self.ue_buffers.items():
            expired_count = buffer.UPD_UE_BUFFER(ue_id, current_time)
            if expired_count > 0:
                print(f"BS: Для UE {ue_id} удалено {expired_count} пакетов")

        # Обновление общего размера буфера
        for buffer in self.ue_buffers.values():
            buffer.total_size = sum(buffer.sizes.values())

    def GET_GLOBAL_BUFFER_STATUS(self, current_time: int) -> Dict:
        """
        Возвращает агрегированную статистику буфера всей базовой станции.

        Args:
            current_time: Текущее время симуляции (мс)

        Returns:
            Dict: {
                'total_size': int,           # Общий размер данных (байты)
                'total_packets': int,        # Общее количество пакетов
                'total_dropped': int,        # Всего отброшено пакетов
                'total_expired': int,        # Всего устаревших пакетов
                'avg_delay': float,          # Средняя задержка по станции (мс)
                'max_delay': int,            # Максимальная задержка (мс)
                'per_ue_avg': Dict[int, float]  # Средний размер буфера на UE
            }
        """
        status = {
            "total_size": 0,
            "total_packets": 0,
            "total_dropped": 0,
            "total_expired": 0,
            "avg_delay": 0.0,
            "max_delay": 0,
            "per_ue": {},  # Явная инициализация
            "per_ue_avg": {},
        }

        total_delay = 0
        packet_count = 0

        for ue_id, buffer in self.ue_buffers.items():
            # Получаем статус через GET_UE_STATUS
            buffer_status = buffer.GET_UE_STATUS(current_time)
            ue_status = buffer_status["per_ue"].get(ue_id, {})

            status["per_ue"][ue_id] = {
                "size": ue_status.get("size", 0),
                "packet_count": ue_status.get("packet_count", 0),
                "oldest_delay": ue_status.get("oldest_delay", 0),
                "avg_delay": ue_status.get("avg_delay", 0.0),
                "dropped": buffer.dropped.get(ue_id, 0),
                "expired": buffer.expired.get(ue_id, 0),
            }

            # Агрегируем показатели
            status["total_size"] += ue_status.get("size", 0)
            status["total_packets"] += ue_status.get("packet_count", 0)
            status["total_dropped"] += buffer.dropped.get(ue_id, 0)
            status["total_expired"] += buffer.expired.get(ue_id, 0)

            # Рассчитываем задержки
            if ue_status.get("packet_count", 0) > 0:
                total_delay += ue_status["avg_delay"] * ue_status["packet_count"]
                packet_count += ue_status["packet_count"]

            status["max_delay"] = max(status["max_delay"], ue_status.get("oldest_delay", 0))

            # Средняя загрузка буфера (нормализованная)
            max_size = buffer.per_ue_max
            current_size = ue_status.get("size", 0)
            status["per_ue_avg"][ue_id] = current_size / max_size if max_size > 0 else 0

        # Расчёт средней задержки
        if packet_count > 0:
            status["avg_delay"] = total_delay / packet_count

        return status

    def CLEAR_ALL_BUFFERS(self) -> None:
        """
        Полностью очищает все буферы базовой станции, удаляя данные всех пользователей.
        """
        for ue_id in list(self.ue_buffers.keys()):  # Используем list для безопасной итерации
            buffer = self.ue_buffers[ue_id]
            buffer.DESTROY_UE_PACKETS(ue_id)  # Очистка буфера конкретного UE

        # print("Все буферы базовой станции успешно очищены")

    def _handle_generated_packets(self, packets: List[Packet]):
        """
        Callback для обработки сгенерированных пакетов.

        Автоматически вызывается PacketManager при генерации.
        Маршрутизирует пакеты в правильные буферы.
        """
        # print(f"[✓ CALLBACK] _handle_generated_packets вызван! {len(packets)} пакетов")
        for pkt in packets:
            # Получаем буфер для UE
            if pkt.ue_id not in self.ue_buffers:
                continue

            buffer = self.ue_buffers[pkt.ue_id]

            # TODO: Если есть LayeredBuffer - маршрутизация по QCI
            # buffer_layer = buffer.get_buffer_for_qci(pkt.qci)
            # buffer_layer.ADD_PACKET(pkt, pkt.creation_time)

            # Пока просто добавляем в общий буфер
            success = buffer.ADD_PACKET(pkt, pkt.creation_time)

            if not success:
                # Пакет задропан буфером
                pass

    def setup_ue_traffic_multi_bearer(self, ue_id: int, bearers: List[Dict]):
        """
        Настройка multi-bearer трафика для UE.

        Args:
            ue_id: ID пользователя
            bearers: Список конфигураций bearers

        Example:
            bs.setup_ue_traffic_multi_bearer(
                 ue_id=1,
                 bearers=[
                     {
                         'model_type': 'Poisson',
                         'qci': 1,
                         'traffic_type': TrafficType.VOIP,
                         'packet_rate': 50,
                         'max_bitrate': 0.064
                     },
                     {
                         'model_type': 'OnOff',
                         'qci': 7,
                         'traffic_type': TrafficType.VIDEO_STREAM,
                         'duration_on': 2,
                         'duration_off': 3,
                         'packet_rate': 200,
                         'max_bitrate': 2.0
                     }
                 ]
             )
        """
        for bearer_config in bearers:
            self.traffic_manager.add_bearer(ue_id=ue_id, **bearer_config)

    def generate_traffic_all_ues(self, current_time: int, update_interval: int):
        """Генерация трафика для ВСЕХ UE."""
        for ue_id in self.registered_ues:
            packets = self.traffic_manager.generate_packets(
                ue_id=ue_id, current_time=current_time, update_interval=update_interval
            )
            # Если callback не вызван (почему-то), добавляем вручную
            if packets:
                for pkt in packets:
                    self.ue_buffers[ue_id].ADD_PACKET(pkt, current_time)


def test_bs_buffer_fifo():
    print("\n=== Тест буфера BS: FIFO, TTL, скорость и BaseStation ===")
    bs = BaseStation()  # Используем экземпляр BaseStation
    current_time = 2000

    # Регистрация пользователей
    ue1 = UserEquipment(1)
    ue2 = UserEquipment(2)
    bs.REG_UE(ue1)
    bs.REG_UE(ue2)

    # Настройка буфера BS (через BaseStation)
    bs.ue_buffers[1] = Buffer(global_max=8000, per_ue_max=5000)
    bs.ue_buffers[2] = Buffer(global_max=8000, per_ue_max=5000)

    # Добавление пакетов с разным TTL
    packets = [
        # UE1: 2 пакета (TTL=1000)
        Packet(size=2000, ue_id=1, creation_time=current_time - 400, ttl_ms=500),
        Packet(size=500, ue_id=1, creation_time=current_time - 300, ttl_ms=500),
        # UE2: 3 пакета (TTL=300 мс)
        Packet(size=3000, ue_id=2, creation_time=current_time - 500, ttl_ms=1000),
        Packet(size=2500, ue_id=2, creation_time=current_time - 300, ttl_ms=1000),
    ]

    # Этап 1: Добавление пакетов
    for pkt in packets:
        bs.ue_buffers[pkt.ue_id].ADD_PACKET(pkt, current_time)

    print("\n[Этап 1] Статус после добавления:")
    status = bs.GET_GLOBAL_BUFFER_STATUS(current_time)
    for ue_id in [1, 2]:
        print(f"UE {ue_id}: {status['per_ue'].get(ue_id, {})}")

    # Этап 2: Извлечение пакетов для UE1
    extracted1, size1 = bs.ue_buffers[1].GET_PACKETS(
        ue_id=1, max_bytes=3000, bits_per_rb=229, current_time=current_time
    )
    extracted2, size2 = bs.ue_buffers[2].GET_PACKETS(
        ue_id=2, max_bytes=3000, bits_per_rb=229, current_time=current_time
    )
    print(f"\n[Этап 2] Извлечено для UE1: {len(extracted1)} пакетов ({size1} байт)")
    print(f"\n[Этап 2] Извлечено для UE2: {len(extracted2)} пакетов ({size2} байт)")
    print("\n[Этап 2] Статус после извлечения:")
    status = bs.GET_GLOBAL_BUFFER_STATUS(current_time)
    for ue_id in [1, 2]:
        print(f"UE {ue_id}: {status['per_ue'].get(ue_id, {})}")

    # =============================================================================
    #     packets = [Packet(size=500, ue_id=2, creation_time=current_time - 100, ttl_ms=500),
    #                Packet(size=500, ue_id=2, creation_time=current_time - 200, ttl_ms=500),
    #                Packet(size=5000, ue_id=2, creation_time=current_time - 100, ttl_ms=500)]
    #     for pkt in packets:
    #         bs.ue_buffers[pkt.ue_id].ADD_PACKET(pkt, current_time)
    # =============================================================================

    # Этап 3: Продвижение времени на 400 мс (TTL UE2 истек)
    current_time += 801
    bs.UPD_GLOBAL_BUFFER(current_time)
    packets = [
        Packet(size=500, ue_id=2, creation_time=current_time - 100, ttl_ms=1000),
        Packet(size=500, ue_id=2, creation_time=current_time - 200, ttl_ms=1000),
        Packet(size=500, ue_id=2, creation_time=current_time - 100, ttl_ms=1000),
    ]
    for pkt in packets:
        bs.ue_buffers[pkt.ue_id].ADD_PACKET(pkt, current_time)

    print("\n[Этап 3] После 801 мс:")
    status = bs.GET_GLOBAL_BUFFER_STATUS(current_time)
    for ue_id in [1, 2]:
        ue_data = status["per_ue"].get(ue_id, {})
        print(f"UE {ue_id}: {ue_data}")

    # Этап 4: Проверка скорости
    print("\n[Этап 4] Статистика скорости:")
    for ue_id in [1, 2]:
        buffer = bs.ue_buffers[ue_id]
        speed = buffer.get_ingress_speed_mbps(ue_id, current_time)
        print(f"UE {ue_id}: {speed:.2f} Mbps")

    # Этап 5: Очистка буфера
    bs.CLEAR_ALL_BUFFERS()
    print("\n[Этап 5] После очистки:")
    status = bs.GET_GLOBAL_BUFFER_STATUS(current_time)
    assert status["total_size"] == 0, "Буфер не очищен"
    print("Все буферы пусты")

    # Ассерты
    # assert bs.buffer.sizes[1] == 3000, "Некорректный размер буфера UE1"
    # assert bs.buffer.expired[2] == 3, "Не удалены все устаревшие пакеты UE2"
    # assert bs.buffer.dropped[2] == 0, "Ложное отбрасывание пакетов UE2"

    print("\n=== Тест завершён успешно ===")


if __name__ == "__main__":
    test_bs_buffer_fifo()
