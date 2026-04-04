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
# Версия: 1.1.1
# Дата последнего изменения: 2026-04-01
# Автор: Македон Никита
# Версия Python Kernel: 3.12.9
# v.1.1.1:
# - Оптимизирован метод обновления буфера
# - Ускорено удаление просроченных пакетов
# - Добавлен быстрый режим обработки FIFO-очереди
# - Сохранена прежняя логика работы и статистики
#------------------------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import GLOBALS
import numpy as np
from TRAFFIC_MODEL import Packet
from UE_MODULE import UserEquipment

@dataclass(slots=True)
class BufferStatus:
    """
    Структура состояния буфера конкретного UE.

    Attributes:
        ue_id (int): Уникальный идентификатор UE.
        buffer_size (int): Текущий размер буфера (байты).
        timestamp (int): Временная метка формирования статуса.
        lcid (Optional[int], optional): Идентификатор логического канала.
        qci (Optional[int], optional): Идентификатор класса QoS.
        priority (int, optional): Приоритет трафика.
        hol_delay (Optional[int], optional): Задержка HOL (мс).
        
    """
    ue_id: int
    buffer_size: int
    timestamp: int
    lcid: Optional[int] = None
    qci: Optional[int] = None
    priority: int = 0
    hol_delay: Optional[int] = None
    
    def __post_init__(self):
        """
        Валидация после инициализации объекта.

        Raises:
            ValueError: Если размер буфера или UE ID отрицательное значение.
        """
        if self.ue_id < 0:
            raise ValueError(
                f"The UE ID cannot be negative. "
                f"The obtained value: {self.ue_id}"
            )

        if self.buffer_size < 0:
            raise ValueError(
                f"The buffer size cannot be negative. "
                f"The obtained value: {self.buffer_size}"
            )
            
    def is_empty(self) -> bool:
        """
        Проверка буфера на пустоту.

        Returns:
            bool: True, если буфер пуст, иначе False.

        """
        return self.buffer_size == 0
    
    def to_dict(self) -> Dict:
        """
        Преобразование объекта состояния буфера в словарь.

        Returns:
            Dict: Словарь с параметрами состояния буфера:
                ue_id: int - Уникальный идентификатор UE.
                buffer_size: int - Текущий размер буфера (байты).
                timestamp: int - Временная метка формирования статуса.
                lcid: Optional[int] - Идентификатор логического канала.
                qci: Optional[int] - Идентификатор класса QoS.
                priority: int - Приоритет трафика.
                hol_delay: Optional[int] - Задержка HOL (мс).

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
    Класс для моделирования простого FIFO-буфера одного UE.
    """
    def __init__(self, ue_id: int, max_size: int = 262144):
        """
        Инициализация буфера для конкретного UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.
            max_size (int, optional): Максимальный размер буфера (байты).
                По умолчанию 262144.

        """
        self.ue_id = ue_id
        self.max_size = max_size
        self.buffer = deque()
        self.current_size = 0 # байты
        
        self.packets_added = 0
        self.packets_dropped = 0
        self.packets_expired = 0
        self._deadline_monotonic_ok = True
        self._tail_deadline = None
        
    def add_packet(self, packet: Packet) -> bool:
        """
        Добавление пакета в буфер.

        Args:
            packet (Packet): Добавляемый пакет.

        Returns:
            bool: True, если пакет успешно добавлен, иначе False.

        """
        # Если пакет не помещается - отбрасываем
        if self.current_size + packet.size > self.max_size:
            self.packets_dropped += 1
            return False
        
        self.buffer.append(packet)
        self.current_size += packet.size
        
        self.packets_added += 1
        
        return True
    
    def get_packets(self, num_bytes: int) -> Tuple[List[Packet], int]:
        """
        Извлечение пакетов из буфера с фрагментацией.

        Args:
            num_bytes (int): Количество байт, которое нужно извлечь.

        Returns:
            (Tuple[List[Packet], int]):
                - список извлечённых (или фрагментированных) пакетов, 
                - фактически извлечённое количество байт.

        """
        extracted_packets = []
        extracted_bits = 0
        num_bits_to_extract = GLOBALS.bytes_to_bits(num_bytes)
        
        while self.buffer and extracted_bits < num_bits_to_extract:
            packet = self.buffer[0]
            packet_size_bits = GLOBALS.bytes_to_bits(packet.size)
            
            # Полное извлечение пакета
            if (extracted_bits + packet_size_bits) <= num_bits_to_extract:
                extracted_packet = self.buffer.popleft()
                extracted_packets.append(extracted_packet)
                extracted_bits += packet_size_bits
                
            # Фрагментация пакета
            else:
                remaining_bits = num_bits_to_extract - extracted_bits
                fragment_size = remaining_bits // 8
                
                # Создание фрагмента
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
                
                # Модификация исходного пакета
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
        Удаление просроченных пакетов из буфера.
        """
        if not self.buffer:
            self._tail_deadline = None
            self._deadline_monotonic_ok = True
            return

        if self._deadline_monotonic_ok:
            expired_bytes = 0
            expired_count = 0

            while self.buffer and self.buffer[0].deadline <= GLOBALS.CURRENT_TIME:
                pkt = self.buffer.popleft()
                expired_bytes += pkt.size
                expired_count += 1

            if expired_count:
                self.current_size -= expired_bytes
                if self.current_size < 0:
                    self.current_size = 0
                self.packets_expired += expired_count

            if self.buffer:
                self._tail_deadline = self.buffer[-1].deadline
            else:
                self._tail_deadline = None
                self._deadline_monotonic_ok = True
            return

        valid_packets = deque()
        current_size = 0
        expired_count = 0
        prev_deadline = None
        monotonic_ok = True

        for packet in self.buffer:
            if packet.deadline > GLOBALS.CURRENT_TIME:
                valid_packets.append(packet)
                current_size += packet.size
                if prev_deadline is not None and packet.deadline < prev_deadline:
                    monotonic_ok = False
                prev_deadline = packet.deadline
            else:
                expired_count += 1

        self.buffer = valid_packets
        self.current_size = current_size
        self.packets_expired += expired_count
        self._deadline_monotonic_ok = monotonic_ok
        self._tail_deadline = prev_deadline

    def get_buffer_status(self) -> BufferStatus:
        """
        Формирование статуса буфера на текущий момент. 

        Returns:
            BufferStatus: Объект с информацией о состоянии буфера, содержащий:
                ue_id: int - Уникальный идентификатор UE.
                buffer_size: int - Текущий размер буфера (байты).
                timestamp: int - Временная метка формирования статуса.

        """
        return BufferStatus(
            ue_id=self.ue_id, 
            buffer_size=self.current_size, 
            timestamp=GLOBALS.CURRENT_TIME
        )
    
    def clear_buffer(self) -> None:
        """
        Полная очистка буфера и сброс статистики.
        
        """
        self.buffer.clear()
        self.current_size = 0
        
        self.packets_added = 0
        self.packets_dropped = 0
        self.packets_expired = 0
        self._deadline_monotonic_ok = True
        self._tail_deadline = None
        

class IBufferManager(ABC):
    """
    Абстрактный интерфейс менеджера буферов.
    """
    @abstractmethod
    def add_packet(self, ue_id: int, packet: Packet) -> bool:
        """
        Добавление пакета в буфер соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.
            packet (Packet): Добавляемый пакет.

        Returns:
            bool: True, если пакет успешно добавлен, иначе False.

        """
        pass
    
    @abstractmethod
    def get_buffer_status(self, ue_id: int) -> List[BufferStatus]:
        """
        Формирование статусов всех буферов соответствующего UE на текущий
        момент.

        Args:
            ue_id (int): Уникальный идентификатор UE.

        Returns:
            List[BufferStatus]: Список объектов о состоянии буферов.

        """
        pass
    
    @abstractmethod
    def get_packets(self, grants: List) -> Tuple[List[Packet], int]:
        """
        Извлечение пакетов из буферов соответствующего UE на основании грантов.

        Args:
            grants (List): Список грантов.

        Returns:
            (Tuple[List[Packet], int]): 
                - список извлечённых (или фрагментированных) пакетов, 
                - фактически извлечённое количество байт.

        """
        pass
    
    @abstractmethod
    def upd_buffers_all(self) -> None:
        """
        Обновление буферов всех UE.
        """
        pass
    
    @abstractmethod
    def ue_has_buffer(self, ue_id: int) -> bool:
        """
        Проверка наличия буфера для соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.

        Returns:
            bool: True, если буфер существует, иначе False.
        """
        pass
    
    @abstractmethod
    def create_ue_buffer(self, ue_id: int, max_size: Optional[int] = None) -> None:
        """
        Создание буфера для соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.
            max_size (Optional[int], optional): Максимальный размер буфера (байты). 
                По умолчанию None.

        """
        pass
    
    @abstractmethod
    def remove_ue_buffer(self, ue_id: int) -> None:
        """
        Удаление буфера для соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.

        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Получение агрегированной статистики по буферам всех UE.

        Returns:
            Dict: Словарь, хранящий статистику.

        """
        pass
       

class SimpleBufferManager(IBufferManager):
    """
    Класс простого менеджера буферов, работающего с простыми FIFO-буферами 
    (SimpleBuffer). 
    """
    def __init__(self, global_max: int = 1048576, per_ue_max: int = 262144):
        """
        Инициализация менеджера буферов.

        Args:
            global_max (int, optional): Максимальный размер буфера базовой станции. 
                По умолчанию 1048576.
            per_ue_max (int, optional): Максимальный размер буфера одного UE. 
                По умолчанию to 262144.

        """
        self.buffers: Dict[int, SimpleBuffer] = {}
        self.global_max = global_max
        self.per_ue_max = per_ue_max
        self.current_total_size = 0
        
    def add_packet(self, ue_id: int, packet: Packet) -> bool:
        """
        Добавление пакета в буфер соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.
            packet (Packet): Добавляемый пакет.

        Returns:
            bool: True, если пакет успешно добавлен, иначе False.
            
        Raises:
            ValueError: Если буфер UE не существует.

        """
        if not self.ue_has_buffer(ue_id):
            raise ValueError(
                f"UE {ue_id} does not have a buffer. The buffer must "
                f"have been created during UE registration at the BS"
            )
            
        if self.current_total_size + packet.size > self.global_max:
            return False
            
        if self.buffers[ue_id].add_packet(packet):
            self.current_total_size += packet.size
            return True
        
        else:
            return False
        
    def get_buffer_status(self, ue_id: int) -> List[BufferStatus]:
        """
        Формирование статусов всех буферов соответствующего UE на текущий
        момент.

        Args:
            ue_id (int): Уникальный идентификатор UE.

        Returns:
            List[BufferStatus]: Список объектов о состоянии буферов.
            
        Raises:
            ValueError: Если буфер UE не существует.

        """
        if not self.ue_has_buffer(ue_id):
            raise ValueError(
                f"UE {ue_id} does not have a buffer. The buffer must "
                f"have been created during UE registration at the BS"
            )
        
        buffer_status = self.buffers[ue_id].get_buffer_status()
        return [buffer_status]
    
    def get_packets(self, grants: List) -> Tuple[List[Packet], int]:
        """
        Извлечение пакетов из буферов соответствующего UE на основании грантов.

        Args:
            grants (List): Список грантов.

        Returns:
            (Tuple[List[Packet], int]): 
                - список извлечённых (или фрагментированных) пакетов, 
                - фактически извлечённое количество байт.
                
        Raises:
            ValueError: Если количество полученных грантов не равно 1 или буфер 
            UE не существует.

        """
        if len(grants) != 1:
            raise ValueError(
                "The size of the grant list for Simple Buffer must be 1"
            )
            
        grant = grants[0]
        ue_id = grant.ue_id
        num_bytes = grant.num_bytes
        
        if not self.ue_has_buffer(ue_id):
            raise ValueError(
                f"UE {ue_id} does not have a buffer. The buffer must "
                f"have been created during UE registration at the BS"
            )
        
        packets, extracted_bytes = self.buffers[ue_id].get_packets(num_bytes)
        self.current_total_size -= extracted_bytes
        
        return packets, extracted_bytes
    
    def upd_buffers_all(self) -> None:
        """
        Обновление буферов всех UE.
        """
        new_total_size = 0
        for buffer in self.buffers.values():
            buffer.upd_buffer()
            new_total_size += buffer.current_size
            
        self.current_total_size = new_total_size    
        
    def ue_has_buffer(self, ue_id: int) -> bool:
        """
        Проверка наличия буфера для соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.

        Returns:
            bool: True, если буфер существует, иначе False.
        """
        return ue_id in self.buffers
    
    def create_ue_buffer(self, ue_id: int, max_size: Optional[int] = None) -> None:
        """
        Создание буфера для соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.
            max_size (Optional[int], optional): Максимальный размер буфера (байты). 
                По умолчанию None.

        """
        ue_buffer_size = self.per_ue_max if max_size is None else max_size
        self.buffers[ue_id] = SimpleBuffer(ue_id=ue_id, max_size=ue_buffer_size)
        
    def remove_ue_buffer(self, ue_id: int) -> None:
        """
        Удаление буфера для соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.

        """
        if self.ue_has_buffer(ue_id):
            self.buffers[ue_id].clear_buffer()
            del self.buffers[ue_id]

    def get_stats(self) -> Dict:
        """
        Получение агрегированной статистики по буферам всех UE.

        Returns:
            Dict: Словарь со статистикой, содержащий:
                total_buffer_size: int - Текущий размер буфера БС (байты).
                total_buffer_capacity: int - Максимальный размер буфера БС (байты).
                total_packets_added: int - Общее кол-во добавленных пакетов.
                total_packets_dropped: int - Общее кол-во отброшенных пакетов.
                total_packets_expired: int - Общее кол-во просроченных пакетов.
                buffer_size_per_ue: Dict - Текущий размер буфера для каждого UE.
                packets_added_per_ue: Dict - Кол-во добавленных пакетов для каждого UE.
                packets_dropped_per_ue: Dict - Кол-во отброшенных пакетов для каждого UE.
                packets_expired_per_ue: Dict - Кол-во просроченных пакетов для каждого UE.
                oldest_delay_per_ue: Dict - Максимальная задержка пакетов для каждого UE.
                avg_delay_per_ue: Dict - Средняя задержка пакетов в буфере каждого UE

        """
        # Общая статистика со всех буфеов
        total_packets_added = 0
        total_packets_dropped = 0
        total_packets_expired = 0

        # Статистика для каждого отдельного буфера
        buffer_size_per_ue = {}
        packets_added_per_ue = {}
        packets_dropped_per_ue = {}
        packets_expired_per_ue = {}

        # Задержки для каждого отдельного буфера
        oldest_delay_per_ue = {}
        avg_delay_per_ue = {}

        for buffer in self.buffers.values():
            total_packets_added += buffer.packets_added
            total_packets_dropped += buffer.packets_dropped
            total_packets_expired += buffer.packets_expired

            buffer_size_per_ue[buffer.ue_id] = buffer.current_size
            packets_added_per_ue[buffer.ue_id] = buffer.packets_added
            packets_dropped_per_ue[buffer.ue_id] = buffer.packets_dropped
            packets_expired_per_ue[buffer.ue_id] = buffer.packets_expired

            buffer_queue = buffer.buffer
            delays = [p.age(GLOBALS.CURRENT_TIME) for p in buffer_queue]
            oldest_delay_per_ue[buffer.ue_id] = max(delays) if delays else 0
            avg_delay_per_ue[buffer.ue_id] = sum(delays) / len(delays) if delays else 0.0

        stats = {
            "total_buffer_size": self.current_total_size,
            "total_buffer_capacity": self.global_max,
            "total_packets_added": total_packets_added,
            "total_packets_dropped": total_packets_dropped,
            "total_packets_expired": total_packets_expired,
            "buffer_size_per_ue": buffer_size_per_ue,
            "packets_added_per_ue": packets_added_per_ue,
            "packets_dropped_per_ue": packets_dropped_per_ue,
            "packets_expired_per_ue": packets_expired_per_ue,
            "oldest_delay_per_ue": oldest_delay_per_ue,
            "avg_delay_per_ue": avg_delay_per_ue,
        }

        return stats


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
        use_simple_buffer: bool = True,
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
        self.use_simple_buffer = use_simple_buffer
        
        if use_simple_buffer:
            self.buffer_manager = SimpleBufferManager(global_max, per_ue_max)
        else:
            raise NotImplementedError(
                "Currently, only Simple Buffer is supported. To start the "
                "simulation, set the use_simple_buffer flag to True"
            )

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
        if not self.buffer_manager.ue_has_buffer(ue.UE_ID):
            self.buffer_manager.create_ue_buffer(ue.UE_ID, self.per_ue_max)
            
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
