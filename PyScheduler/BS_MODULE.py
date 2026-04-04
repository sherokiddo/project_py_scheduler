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
from TRAFFIC_MODEL import Packet, QCI, UeBearersInfo
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
    def create_ue_buffer(self, ue_id: int, max_size: Optional[int] = None, bearers_info: UeBearersInfo = None) -> None:
        """
        Создание буфера для соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.
            max_size (Optional[int], optional): Максимальный размер буфера (байты). 
                По умолчанию None.
            bearers_info (UeBearersInfo, optional): Информация о сконфигурированных bearer'ах UE. 
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

        self.global_packets_dropped = 0
        
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
                f"have been created during start simulation"
            )
            
        if self.current_total_size + packet.size > self.global_max:
            self.global_packets_dropped += 1
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
                f"have been created during start simulation"
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
                f"have been created during start simulation"
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
    
    def create_ue_buffer(self, ue_id: int, max_size: Optional[int] = None, bearers_info: UeBearersInfo = None) -> None:
        """
        Создание буфера для соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.
            max_size (Optional[int], optional): Максимальный размер буфера (байты). 
                По умолчанию None.
            bearers_info (UeBearersInfo, optional): Информация о сконфигурированных bearer'ах UE. 
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
            self.current_total_size -= self.buffers[ue_id].current_size
            self.buffers[ue_id].clear_buffer()
            del self.buffers[ue_id]

    def get_stats(self) -> Dict:
        """
        Получение агрегированной статистики по буферам всех UE.

        Returns:
            Dict: Словарь со статистикой, содержащий:
                total_buffer_size: int - Текущий размер буфера БС (байты).
                total_buffer_capacity: int - Максимальный размер буфера БС (байты).
                global_packets_dropped: int - Кол-во отброшенных пакетов из-за переполнения буфера БС.
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
            "global_packets_dropped": self.global_packets_dropped,
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
    

class RLCEntity(ABC):
    """
    Абстрактный класс для RLC Entity.
    """
    def __init__(self, ue_id: int, lcid: int, qci: int, max_tx_buffer_size: int):
        """
        Инициализация RLC Entity.

        Args:
            ue_id (int): Уникальный идентификатор UE.
            lcid (int): Идентификатор логического канала.
            qci (int): Идентификатор класса QoS.
            max_tx_buffer_size (int): Максимальный размер буфера передачи (байты).

        """
        self.ue_id = ue_id
        self.lcid = lcid
        self.qci = qci
        self.max_tx_buffer_size = max_tx_buffer_size

        self.priority = QCI(qci).get_priority()
        self.pdcp_discard_timer = GLOBALS.PDCP_DISCARD_TIMERS.get(qci)

    @abstractmethod
    def add_packet(self, packet: Packet) -> bool:
        """
        Добавление пакета в буфер передачи.

        Args:
            packet (Packet): Добавляемый пакет.

        Returns:
            bool: True, если пакет успешно добавлен, иначе False.

        """
        pass

    @abstractmethod
    def get_packets(self, num_bytes_to_extract: int) -> Tuple[List[Packet], int]:
        """
        Извлечение пакетов из буфера для передачи.

        Args:
            num_bytes_to_extract (int): Количество байт, которое нужно извлечь.

        Returns:
            (Tuple[List[Packet], int]):
                - список извлечённых (или фрагментированных) пакетов, 
                - фактически извлечённое количество байт.

        """
        pass

    @abstractmethod
    def upd_buffer(self) -> None:
        """
        Обновление состояния буфера.
        
        """
        pass

    @abstractmethod
    def get_buffer_status(self) -> BufferStatus:
        """
        Получение текущего состояния буфера.

        Returns:
            BufferStatus: Объект с информацией о состоянии буфера, содержащий:
                ue_id: int - Уникальный идентификатор UE.
                buffer_size: int - Текущий размер буфера (байты).
                timestamp: int - Временная метка формирования статуса.
                lcid: int - Идентификатор логического канала.
                qci: int - Идентификатор класса QoS.
                priority: int - Приоритет трафика.
                hol_delay: int = Задержка HOL (мс).

        """
        pass

    @abstractmethod
    def clear_buffer(self) -> None:
        """
        Очистка буфера передачи.

        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Получение статистики работы буфера.

        Returns:
            Dict: Словарь, хранящий статистику.

        """
        pass


class RLCUM(RLCEntity):
    """
    Реализация RLC Entity в режиме Unacknowledged Mode (UM).
    
    Буферизация пакетов выполняется в FIFO очереди. Поддерживается
    сегментация пакетов при извлечении данных и удаление устаревших
    пакетов согласно PDCP discard timer.
    """
    def __init__(self, ue_id: int, lcid: int, qci: int, max_tx_buffer_size: int):
        """
        Инициализация RLC Entity в режиме Unacknowledged Mode (UM).

        Args:
            ue_id (int): Уникальный идентификатор UE.
            lcid (int): Идентификатор логического канала.
            qci (int): Идентификатор класса QoS.
            max_tx_buffer_size (int): Максимальный размер буфера передачи (байты).

        """
        super().__init__(ue_id, lcid, qci, max_tx_buffer_size)

        self.tx_buffer = deque()
        self.current_tx_buffer_size = 0

        self.packets_added = 0
        self.packets_dropped = 0
        self.packets_expired = 0
        self.extracted_bytes = 0

    def add_packet(self, packet: Packet) -> bool:
        """
        Добавление пакета в буфер передачи.
        
        Проверяется доступное место в буфере. При переполнении
        пакет отбрасывается.

        Args:
            packet (Packet): Добавляемый пакет.

        Returns:
            bool: True, если пакет успешно добавлен, иначе False.

        """
        if self.current_tx_buffer_size + packet.size > self.max_tx_buffer_size:
            self.packets_dropped += 1
            return False
        
        self.tx_buffer.append(packet)
        self.current_tx_buffer_size += packet.size
        
        self.packets_added += 1
        
        return True
    
    def get_packets(self, num_bytes_to_extract: int) -> Tuple[List[Packet], int]:
        """
        Извлечение пакетов из буфера для передачи.
        
        При необходимости выполняется сегментация пакета для
        соответствия запрошенному количеству байт.

        Args:
            num_bytes_to_extract (int): Количество байт, которое нужно извлечь.

        Returns:
            (Tuple[List[Packet], int]):
                - список извлечённых (или фрагментированных) пакетов, 
                - фактически извлечённое количество байт.

        """
        extracted_packets = []
        extracted_bytes = 0

        while self.tx_buffer and extracted_bytes < num_bytes_to_extract:
            packet = self.tx_buffer[0]
            packet_size = packet.size

            # Сегментация пакета
            if (extracted_bytes + packet_size) > num_bytes_to_extract:
               segment_size = num_bytes_to_extract - extracted_bytes
               segment = packet.split_packet(segment_size)

               extracted_packets.append(segment)
               extracted_bytes += segment_size
               break

            # Полное извлечение пакета
            else:
                extracted_packet = self.tx_buffer.popleft()
                extracted_packets.append(extracted_packet)
                extracted_bytes += packet_size

        self.current_tx_buffer_size -= extracted_bytes
        self.extracted_bytes = extracted_bytes

        return extracted_packets, extracted_bytes

    def upd_buffer(self) -> None:
        """
        Обновление состояния буфера.
        
        Удаляются пакеты, возраст которых превышает значение
        PDCP discard timer (если пакет не был сегментирован).
        """
        new_buffer_size = 0

        for _ in range(len(self.tx_buffer)):
            packet = self.tx_buffer.popleft()
            
            # Отбрасываем пакет, если  его возраст > PDCP discard timer
            # и этот пакет не сегментировался (см. 3GPP TS 36.322 p. 5.3)
            if (packet.age(GLOBALS.CURRENT_TIME) > self.pdcp_discard_timer 
                and not packet.is_fragment):
                self.packets_expired += 1
            else:
                self.tx_buffer.append(packet)
                new_buffer_size += packet.size

        self.current_tx_buffer_size = new_buffer_size

    def get_buffer_status(self) -> BufferStatus:
        """
        Получение текущего состояния буфера.

        Returns:
            BufferStatus: Объект с информацией о состоянии буфера, содержащий:
                ue_id: int - Уникальный идентификатор UE.
                buffer_size: int - Текущий размер буфера (байты).
                timestamp: int - Временная метка формирования статуса.
                lcid: int - Идентификатор логического канала.
                qci: int - Идентификатор класса QoS.
                priority: int - Приоритет трафика.
                hol_delay: int = Задержка HOL (мс).

        """
        hol_delay = self.tx_buffer[0].age(GLOBALS.CURRENT_TIME) if self.tx_buffer else 0

        return BufferStatus(
            ue_id=self.ue_id, 
            buffer_size=self.current_tx_buffer_size, 
            timestamp=GLOBALS.CURRENT_TIME,
            lcid=self.lcid,
            qci=self.qci,
            priority=self.priority,
            hol_delay=hol_delay
        )
    
    def clear_buffer(self) -> None:
        """
        Очистка буфера передачи и сброс статистики.

        """
        self.tx_buffer.clear()
        self.current_tx_buffer_size = 0

        self.packets_added = 0
        self.packets_dropped = 0
        self.packets_expired = 0
        self.extracted_bytes = 0

    def get_stats(self) -> Dict:
        """
        Получение статистики работы буфера.

        Returns:
            Dict: Словарь со статистикой, содержащий:
                qci: int - Идентификатор класса QoS.
                buffer_size: int - Текущий размер буфера передачи.
                extracted_bytes: int - Кол-во извлечённых байт.
                packets_added: int - Кол-во добавленных пакетов.
                packets_dropped: int - Кол-во отброшенных пакетов.
                packets_expired: int - Кол-во просроченных пакетов.
                oldest_delay: int - Максимальная задержка пакетов в буфере.
                avg_delay: float - Средняя задержка пакетов в буфере.

        """
        delays = [p.age(GLOBALS.CURRENT_TIME) for p in self.tx_buffer]
        oldest_delay = max(delays) if delays else 0
        avg_delay = sum(delays) / len(delays) if delays else 0.0

        stats = {
            "qci": self.qci,
            "buffer_size": self.current_tx_buffer_size,
            "extracted_bytes": self.extracted_bytes,
            "packets_added": self.packets_added,
            "packets_dropped": self.packets_dropped,
            "packets_expired": self.packets_expired,
            "oldest_delay": oldest_delay,
            "avg_delay": avg_delay,
        }

        return stats


class UeProtocolStack:
    """
    Стек протоколов пользователя.
    
    Управляет набором сущностей RLC и PDCP (на данный момент PDCP реализация
    отсутствует), каждые из которых соответствует отдельному логическому каналу 
    (LC).
    """
    def __init__(self, ue_id: int):
        """
        Инициализация стека протоколов пользователя.

        Args:
            ue_id (int): Уникальный идентификатор UE.

        """
        self.ue_id = ue_id
        self.rlc_entities: Dict[int, RLCEntity] = {}

    def create_entities(self, lcid: int, qci: int, buffer_size: int):
        """
        Создание сущностей RLC и PDCP (на данный момент PDCP реализация
        отсутствует) для логического канала.

        Args:
            lcid (int): Идентификатор логического канала.
            qci (int): Идентификатор класса QoS.
            buffer_size (int): Максимальный размер буфера.

        Raises:
            ValueError: Если сущность с заданным LCID уже существует или у UE
            отсутствует данный LCID.

        """
        if lcid in self.rlc_entities:
            raise ValueError(
                f"LCID {lcid} already exists for UE {self.ue_id}"
            )
        
        # Пока что создаём RLC UM для всех логических каналов
        self.rlc_entities[lcid] = RLCUM(
            ue_id=self.ue_id, lcid=lcid, qci=qci, max_tx_buffer_size=buffer_size
        )

    def add_packet_to_lc(self, lcid: int, packet: Packet) -> bool:
        """
        Добавление пакета в буфер указанного логического канала.

        Args:
            lcid (int): Идентификатор логического канала.
            packet (Packet): Добавляемый пакет.

        Raises:
            ValueError: Если у UE отсутствует данный LCID.

        Returns:
            bool: True, если пакет успешно добавлен, иначе False.

        """
        rlc_entity = self.rlc_entities.get(lcid)
        if rlc_entity is None:
            raise ValueError(
                f"LCID {lcid} not configured for UE {self.ue_id}"
            )
        
        return rlc_entity.add_packet(packet)
    
    def get_buffer_status(self) -> List[BufferStatus]:
        """
        Получение состояния буферов всех логических каналов.

        Returns:
            List[BufferStatus]: Список статусов буферов.

        """
        buffer_status_list = []
        for rlc_entity in self.rlc_entities.values():
            buffer_status = rlc_entity.get_buffer_status()
            buffer_status_list.append(buffer_status)

        return buffer_status_list
    
    def get_packets_from_lc(self, lcid: int, num_bytes: int) -> Tuple[List[Packet], int]:
        """
        Извлечение пакетов из буфера указанного логического канала.

        Args:
            lcid (int): Идентификатор логического канала.
            num_bytes (int): Количество байт, которое нужно извлечь.

        Raises:
            ValueError: Если у UE отсутствует данный LCID.

        Returns:
            (Tuple[List[Packet], int]):
                - список извлечённых (или фрагментированных) пакетов, 
                - фактически извлечённое количество байт.

        """
        rlc_entity = self.rlc_entities.get(lcid)
        if rlc_entity is None:
            raise ValueError(
                f"LCID {lcid} not configured for UE {self.ue_id}"
            )
        
        return rlc_entity.get_packets(num_bytes)
    
    def upd_buffers(self) -> int:
        """
        Обновление состояния всех буферов.

        Returns:
            int: Суммарный размер всех буферов после обновления.

        """
        new_buffers_size = 0
        for rlc_entity in self.rlc_entities.values():
            rlc_entity.upd_buffer()
            new_buffers_size += rlc_entity.current_tx_buffer_size

        return new_buffers_size
    
    def ue_has_buffer(self) -> bool:
        """
        Проверка наличия буферов у UE.

        Returns:
            bool: True при наличии буферов, иначе False.

        """
        return bool(self.rlc_entities)
    
    def clear_buffers(self) -> None:
        """
        Очистка всех буферов UE.

        Returns:
            int: Общий размер очищенных буферов.

        """
        freed_size = 0

        for rlc_entity in self.rlc_entities.values():
            freed_size += rlc_entity.current_tx_buffer_size
            rlc_entity.clear_buffer()

        self.rlc_entities.clear()

        return freed_size

    def get_stats(self) -> Dict:
        """
        Получение статистики работы всех буферов UE.

        Returns:
            Dict: Словарь со статистикой, содержащий:
                ue_packets_added: int - Общее кол-во добавленных пакетов для UE.
                ue_packets_dropped: int - Общее кол-во отброшенных пакетов для UE.
                ue_packets_expired: int - Общее кол-во просроченных пакетов для UE.
                buffer_size_per_qci: Dict - Текущий размер буфера для каждого QCI.
                extracted_bytes_per_qci: Dict - Кол-во извлечённых байт для каждого QCI.
                packets_added_per_qci: Dict - Кол-во добавленных пакетов для каждого QCI.
                packets_dropped_per_qci: Dict - Кол-во отброшенных пакетов для каждого QCI.
                packets_expired_per_qci: Dict - Кол-во просроченных пакетов для каждого QCI.
                oldest_delay_per_qci: Dict - Максимальная задержка пакетов для каждого QCI.
                avg_delay_per_qci: Dict - Средняя задержка пакетов для каждого QCI.

        """
        # Общая статистика со всех буфеов UE
        ue_packets_added = 0
        ue_packets_dropped = 0
        ue_packets_expired = 0

        # Статистика для каждого отдельного QCI
        buffer_size_per_qci = {}
        extracted_bytes_per_qci = {}
        packets_added_per_qci = {}
        packets_dropped_per_qci = {}
        packets_expired_per_qci = {}

        # Задержки для каждого отдельного QCI
        oldest_delay_per_qci = {}
        avg_delay_per_qci = {}

        for rlc_entity in self.rlc_entities.values():
            rlc_entity_stats = rlc_entity.get_stats()
            qci = rlc_entity_stats.get("qci")

            buffer_size_per_qci[qci] = rlc_entity_stats.get("buffer_size")
            extracted_bytes_per_qci[qci] = rlc_entity_stats.get("extracted_bytes")
            packets_added_per_qci[qci] = rlc_entity_stats.get("packets_added")
            packets_dropped_per_qci[qci] = rlc_entity_stats.get("packets_dropped")
            packets_expired_per_qci[qci] = rlc_entity_stats.get("packets_expired")

            oldest_delay_per_qci[qci] = rlc_entity_stats.get("oldest_delay")
            avg_delay_per_qci[qci] = rlc_entity_stats.get("avg_delay")

            ue_packets_added += packets_added_per_qci[qci]
            ue_packets_dropped += packets_dropped_per_qci[qci]
            ue_packets_expired += packets_expired_per_qci[qci]

        stats = {
            "ue_packets_added": ue_packets_added,
            "ue_packets_dropped": ue_packets_dropped,
            "ue_packets_expired": ue_packets_expired,
            "buffer_size_per_qci": buffer_size_per_qci,
            "extracted_bytes_per_qci": extracted_bytes_per_qci,
            "packets_added_per_qci": packets_added_per_qci,
            "packets_dropped_per_qci": packets_dropped_per_qci,
            "packets_expired_per_qci": packets_expired_per_qci,
            "oldest_delay_per_qci": oldest_delay_per_qci,
            "avg_delay_per_qci": avg_delay_per_qci,
        }

        return stats


class LayeredBufferManager(IBufferManager):
    """
    Класс многоуровневого менеджера буферов, работающего с RLC и PDCP сущностями 
    (на данный момент реализация PDCP Entity отсутствует) и их буферами.
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
        self.ue_stacks: Dict[int, UeProtocolStack] = {}
        self.global_max = global_max
        self.per_ue_max = per_ue_max
        self.current_total_size = 0

        self.global_packets_dropped = 0

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
                f"have been created during start simulation"
            )
        
        if self.current_total_size + packet.size > self.global_max:
            self.global_packets_dropped += 1
            return False
        
        ue_stack = self.ue_stacks.get(ue_id)
        lcid = self._get_lcid_from_bearer_id(packet.bearer_id)

        if ue_stack.add_packet_to_lc(lcid, packet):
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
                f"have been created during start simulation"
            )
        
        ue_stack = self.ue_stacks.get(ue_id)
        
        return ue_stack.get_buffer_status()

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
            ValueError: Если списое полученных грантов пуст, у полученных грантов
            значения UE ID отличаются или буфер UE не существует.

        """
        if not grants or not all(grant.ue_id == grants[0].ue_id for grant in grants):
            raise ValueError(
                "The list of grants must not be empty. Also, the UE ID "
                "in all grants must be the same"
            )
        
        ue_id = grants[0].ue_id

        if not self.ue_has_buffer(ue_id):
            raise ValueError(
                f"UE {ue_id} does not have a buffer. The buffer must "
                f"have been created during start simulation"
            )
        
        ue_stack = self.ue_stacks.get(ue_id)
        
        packets = []
        extracted_bytes = 0
        
        for grant in grants:
            lcid = grant.lcid
            num_bytes = grant.num_bytes

            lc_packets, lc_extracted_bytes = ue_stack.get_packets_from_lc(lcid, num_bytes)
            
            packets.extend(lc_packets)
            extracted_bytes += lc_extracted_bytes

        self.current_total_size -= extracted_bytes

        return packets, extracted_bytes
            
    def upd_buffers_all(self) -> None:
        """
        Обновление буферов всех UE.
        """
        new_total_size = 0
        for ue_stack in self.ue_stacks.values():
            new_buffers_size = ue_stack.upd_buffers()
            new_total_size += new_buffers_size

        self.current_total_size = new_total_size

    def ue_has_buffer(self, ue_id: int) -> bool:
        """
        Проверка наличия буфера для соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.

        Returns:
            bool: True, если буфер существует, иначе False.
        """
        return ue_id in self.ue_stacks and self.ue_stacks[ue_id].ue_has_buffer()

    def create_ue_buffer(self, ue_id: int, max_size: Optional[int] = None, bearers_info: UeBearersInfo = None) -> None:
        """
        Создание буферов для соответствующего UE для каждого сконфигурированного
        bearer'а.

        Args:
            ue_id (int): Уникальный идентификатор UE.
            max_size (Optional[int], optional): Максимальный размер буфера (байты). 
                По умолчанию None.
            bearers_info (UeBearersInfo, optional): Информация о сконфигурированных bearer'ах UE. 
                По умолчанию None.

        """
        ue_buffer_size = self.per_ue_max if max_size is None else max_size
        self.ue_stacks[ue_id] = UeProtocolStack(ue_id=ue_id)

        if bearers_info is None:
            return

        bearers = bearers_info.bearers

        for bearer in bearers.values():
            bearer_id = bearer.bearer_id
            qci = bearer.qci

            lcid = self._get_lcid_from_bearer_id(bearer_id)

            self.ue_stacks[ue_id].create_entities(lcid, qci, ue_buffer_size)

    def remove_ue_buffer(self, ue_id: int) -> None:
        """
        Удаление буферов для соответствующего UE.

        Args:
            ue_id (int): Уникальный идентификатор UE.

        """
        if self.ue_has_buffer(ue_id):
            ue_stack = self.ue_stacks.get(ue_id)
            freed_size = ue_stack.clear_buffers()
            self.current_total_size -= freed_size

            del self.ue_stacks[ue_id]

    def get_stats(self) -> Dict:
        """
        Получение агрегированной статистики по буферам всех UE.

        Returns:
            Dict: Словарь со статистикой, содержащий:
                total_buffer_size: int - Текущий размер буфера БС (байты).
                total_buffer_capacity: int - Максимальный размер буфера БС (байты).
                global_packets_dropped: int - Кол-во отброшенных пакетов из-за переполнения буфера БС.
                total_packets_added: int - Общее кол-во добавленных пакетов.
                total_packets_dropped: int - Общее кол-во отброшенных пакетов.
                total_packets_expired: int - Общее кол-во просроченных пакетов.
                packets_added_per_ue: Dict - Кол-во добавленных пакетов для каждого UE.
                packets_dropped_per_ue: Dict - Кол-во отброшенных пакетов для каждого UE.
                packets_expired_per_ue: Dict - Кол-во просроченных пакетов для каждого UE.
                buffer_size_per_ue_per_qci: Dict - Текущий размер буфера для каждого QCI каждого UE.
                extracted_bytes_per_ue_per_qci: Dict - Кол-во извлечённых байт для каждого QCI каждого UE.
                packets_added_per_ue_per_qci: Dict - Кол-во добавленных пакетов для каждого QCI каждого UE.
                packets_dropped_per_ue_per_qci: Dict - Кол-во отброшенных пакетов для каждого QCI каждого UE.
                packets_expired_per_ue_per_qci: Dict - Кол-во просроченных пакетов для каждого QCI каждого UE.
                oldest_delay_per_ue_per_qci: Dict - Максимальная задержка пакетов для каждого QCI каждого UE.
                avg_delay_per_ue_per_qci: Dict - Средняя задержка пакетов для каждого QCI каждого UE.

        """
        # Общая статистика со всех буфеов всех UE
        total_packets_added = 0
        total_packets_dropped = 0
        total_packets_expired = 0

        # Статистика со всех буфеов для каждого UE
        packets_added_per_ue = {}
        packets_dropped_per_ue = {}
        packets_expired_per_ue = {}

        # Статистика для каждого отдельного QCI каждого UE
        buffer_size_per_ue_per_qci = {}
        extracted_bytes_per_ue_per_qci = {}
        packets_added_per_ue_per_qci = {}
        packets_dropped_per_ue_per_qci = {}
        packets_expired_per_ue_per_qci = {}

        # Задержки для каждого отдельного QCI каждого UE
        oldest_delay_per_ue_per_qci = {}
        avg_delay_per_ue_per_qci = {}

        for ue_stack in self.ue_stacks.values():
            ue_id = ue_stack.ue_id
            ue_stack_stats = ue_stack.get_stats()

            total_packets_added += ue_stack_stats.get("ue_packets_added")
            total_packets_dropped += ue_stack_stats.get("ue_packets_dropped")
            total_packets_expired += ue_stack_stats.get("ue_packets_expired")

            packets_added_per_ue[ue_id] = ue_stack_stats.get("ue_packets_added")
            packets_dropped_per_ue[ue_id] = ue_stack_stats.get("ue_packets_dropped")
            packets_expired_per_ue[ue_id] = ue_stack_stats.get("ue_packets_expired")

            buffer_size_per_ue_per_qci[ue_id] = ue_stack_stats.get("buffer_size_per_qci")
            extracted_bytes_per_ue_per_qci[ue_id] = ue_stack_stats.get("extracted_bytes_per_qci")
            packets_added_per_ue_per_qci[ue_id] = ue_stack_stats.get("packets_added_per_qci")
            packets_dropped_per_ue_per_qci[ue_id] = ue_stack_stats.get("packets_dropped_per_qci")
            packets_expired_per_ue_per_qci[ue_id] = ue_stack_stats.get("packets_expired_per_qci")

            oldest_delay_per_ue_per_qci[ue_id] = ue_stack_stats.get("oldest_delay_per_qci")
            avg_delay_per_ue_per_qci[ue_id] = ue_stack_stats.get("avg_delay_per_qci")

        stats = {
            "total_buffer_size": self.current_total_size,
            "total_buffer_capacity": self.global_max,
            "global_packets_dropped": self.global_packets_dropped,
            "total_packets_added": total_packets_added,
            "total_packets_dropped": total_packets_dropped,
            "total_packets_expired": total_packets_expired,
            "packets_added_per_ue": packets_added_per_ue,
            "packets_dropped_per_ue": packets_dropped_per_ue,
            "packets_expired_per_ue": packets_expired_per_ue,
            "buffer_size_per_ue_per_qci": buffer_size_per_ue_per_qci,
            "extracted_bytes_per_ue_per_qci": extracted_bytes_per_ue_per_qci,
            "packets_added_per_ue_per_qci": packets_added_per_ue_per_qci,
            "packets_dropped_per_ue_per_qci": packets_dropped_per_ue_per_qci,
            "packets_expired_per_ue_per_qci": packets_expired_per_ue_per_qci,
            "oldest_delay_per_ue_per_qci": oldest_delay_per_ue_per_qci,
            "avg_delay_per_ue_per_qci": avg_delay_per_ue_per_qci,
        }

        return stats

    def _get_lcid_from_bearer_id(self, bearer_id: int) -> int:
        """
        Преобразование bearer ID в идентификатор логического канала.
        
        LCID 0, 1 и 2 отводятся для SRB (Signaling Radio Bearers), поэтому
        LCID для DRB = Bearer ID + 2.

        Args:
            bearer_id (int): Уникальный идентификатор bearer.

        Returns:
            int: Идентификатор логического канала (LCID).

        """
        return bearer_id + GLOBALS.SRB_LCID_OFFSET


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
            self.buffer_manager = LayeredBufferManager(global_max, per_ue_max)

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
