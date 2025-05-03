"""
#------------------------------------------------------------------------------
# Модуль: BS_MODULE - Модель базовой станции (BS) для сети LTE
#------------------------------------------------------------------------------
# Описание:
#   Модуль содержит класс BaseStation, реализующий модель базовой станции LTE.
#   Предоставляет параметры конфигурации и характеристики базовой станции,
#   включая мощность передачи, антенные параметры и частотные характеристики.
#
# Версия: 1.0.0
# Дата последнего изменения: 2025-03-29
# Автор: Норицин Иван
# Версия Python Kernel: 3.12.9
#------------------------------------------------------------------------------
"""
from collections import deque, defaultdict
from typing import Dict, List, Optional, Union, Tuple
from UE_MODULE import UserEquipment

class Packet:
    """Класс для представления пакета данных в Downlink-буфере"""
    def __init__(
        self,
        size: int,
        ue_id: int,
        creation_time: int,  # Время в мс
        priority: int = 0,
        ttl_ms: int = 1000,
        is_fragment: bool = False
    ):
        self.size = size
        self.ue_id = ue_id
        self.creation_time = creation_time
        self.priority = priority
        self.ttl_ms = ttl_ms #сделай как тебе удобно
        self.is_fragment = is_fragment

    @property
    def age(self, current_time: int) -> int:
        """Возраст пакета в мс относительно текущего времени симуляции"""
        return current_time - self.creation_time

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
        self.sizes = defaultdict(int)     # {ue_id: текущий размер}
        self.dropped = defaultdict(int)   # {ue_id: счетчик отброшенных}
        self.expired = defaultdict(int)   # {ue_id: счетчик устаревших}
        self.dropped_info = defaultdict(list)
        self.ingress_stats = defaultdict(lambda: {
            'total_bytes': 0,
            'start_time': None,
        })
    
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
            reject_reason = 'ue_limit'
        elif self.total_size + packet.size > self.global_max:
            reject_reason = 'global_limit'
    
        if reject_reason:
            self.dropped[packet.ue_id] += 1
            self.dropped_info[packet.ue_id].append({
                'size': packet.size,
                'creation_time': packet.creation_time,
                'priority': packet.priority,
                'reason': reject_reason
            })
            return False

        # Шаг 3: Обновление статистики скорости
        stats = self.ingress_stats[packet.ue_id]
        if not stats['start_time']:
            stats['start_time'] = current_time
        stats['total_bytes'] += packet.size
        stats['last_update'] = current_time
    
        # Шаг 4: Добавление пакета с обновлением размеров
        self.queues[packet.ue_id].append(packet)
        self.sizes[packet.ue_id] += packet.size
        self.total_size += packet.size
        
        return True

        #@sherokiddo: "Возможно, у пакета появится атрибут метки QoS или приоритет
        # заглушку для него сделал. В дальнейшем реализовать функцию CHCK_PRIORITY или CHCK_PR
        # а также реализовать логику переполнения буфера и отбрасывания пакетов
        # а также, добавить возможность менять приоритет пакета через метод
        # а напоследок, метод для получения пакетов определенного приоритета GET_PCKT_BY_PR"

    def GET_PACKETS(self, ue_id: int, max_bytes: int, bits_per_rb: int, 
               current_time: int) -> Tuple[List[Packet], int]:
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
        self.queues[ue_id] = deque([
            p for p in self.queues[ue_id] 
            if (current_time - p.creation_time) <= p.ttl_ms
        ])
        
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
                    is_fragment=True
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

    #@sherokiddo: тут мог закрасться какой-то баг, но
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
        status = {
            'total_size': 0,
            'total_packets': 0,
            'total_expired': 0,
            'per_ue': {}
        }
        
        for ue_id in self.queues:
            queue = self.queues[ue_id]
            if not queue:
                status['per_ue'][ue_id] = {
                    'size': 0,
                    'packet_count': 0,
                    'oldest_delay': 0,
                    'avg_delay': 0.0,
                    'dropped': self.dropped.get(ue_id, 0),
                    'expired': self.expired.get(ue_id, 0)
                }
                continue
                
            # Статистика для конкретного UE
            delays = [current_time - p.creation_time for p in queue]
            ue_status = {
                'size': self.sizes[ue_id],
                'packet_count': len(queue),
                'oldest_delay': max(delays) if delays else 0,
                'avg_delay': sum(delays)/len(delays) if delays else 0.0,
                'dropped': self.dropped[ue_id],
                'expired': self.expired[ue_id],
                'ingress_bytes': self.ingress_stats[ue_id]['total_bytes']
            }
            
            # Агрегированная статистика
            status['per_ue'][ue_id] = ue_status
            status['total_size'] += ue_status['size']
            status['total_packets'] += ue_status['packet_count']
            status['total_expired'] += ue_status['expired']
            time_interval = current_time - self.ingress_stats[ue_id]['start_time']
            ue_status['ingress_rate_bps'] = (ue_status['ingress_bytes'] * 8 / time_interval) * 1000 if time_interval > 0 else 0
            
            status['per_ue'][ue_id] = ue_status
        
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
        total_bytes = stats['total_bytes']
        start_time = stats.get('start_time', None)
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
    MACROCELL_TX_POWER = {
        1.4: 39,
        3: 41,
        5: 43,
        10: 44,
        15: 45,
        20: 46
    }
    
    # Мощность передачи для микросотовых станций (дБм) по полосам пропускания (МГц)
    MICROCELL_TX_POWER = {
        1.4: 30,
        3: 32,
        5: 34,
        10: 36,
        15: 37,
        20: 38
    }
    
    def __init__(self, x: float = 0.0, y: float = 0.0, height: float = 35.0,
                 frequency_GHz: float = 1.8, bandwidth: float = 10):
        """
        Инициализация базовой станции.
        
        Args:
            x: Координата X расположения станции
            y: Координата Y расположения станции
            height: Высота установки антенны (метры)
            frequency_GHz: Рабочая частота (ГГц)
            bandwidth: Полоса пропускания (МГц)
        """
        self.position = (x, y) # Позиция станции (x, y)
        self.height = height # Высота антенны
        
        # Частотные параметры
        self.frequency_GHz = frequency_GHz # Частота в ГГц
        self.frequency_Hz = frequency_GHz * 1e9 # Частота в Гц
        self.bandwidth = bandwidth # Полоса пропускания
        
        # Характеристики передачи
        self.tx_power = None # Мощность передачи (устанавливается отдельно)
        self.antenna_gain = 15 # Коэффициент усиления антенны (дБи)

        # Апдейт по буферу
        self.ue_buffers = defaultdict(Buffer)
        self.ue_traffic_models = {}  # {ue_id: traffic_model}

    def REG_UE(self, ue: UserEquipment):
        """Регистрация UE с переносом модели трафика"""
        self.ue_buffers[ue.UE_ID] = Buffer()
        self.ue_traffic_models[ue.UE_ID] = ue.traffic_model
        
    def SET_TRAFFIC_MODEL(self, ue: UserEquipment, model):
        """
        Установить модель генерации трафика для конкретного пользователя.
        
        Args:
            ue: объект UserEquipment
            model: объект модели трафика
        """
        ue.SET_TRAFFIC_MODEL(model)
        
        #@sherokiddo: "Предусмотреть валидацию"
    
    def GEN_TRFFC(self, current_time: int, update_interval: int, 
             ue_id: int = None, ttl_ms: int = 1000) -> None:
        """
        Генерирует DL-трафик для пользователей по UE_ID
        
        Args:
            current_time: Текущее время в мс (используется для TTL)
            update_interval: Интервал обновления трафика (мс)
            ue_id: Опциональный ID конкретного пользователя
            ttl_ms: Время жизни пакетов в миллисекундах (по умолчанию 1000)
        """
        
        if not self.ue_traffic_models:
            raise ValueError("Нет зарегистрированных пользователей!")
    
        targets = [ue_id] if ue_id else self.ue_traffic_models.keys()
    
        for target_ue_id in targets:
            model = self.ue_traffic_models.get(target_ue_id)
            if not model:
                continue
            
            buffer = self.ue_buffers[target_ue_id]
    
        # Генерация сырых данных через модель трафика
        
            raw_data = model.generate(
                current_time=current_time,
                update_interval=update_interval
            )

            packets = [
                Packet(
                    size=pkt['size'],
                    ue_id=target_ue_id,
                    creation_time=pkt.get('creation_time', current_time),
                    priority=pkt.get('priority', 0),
                    ttl_ms=ttl_ms
                ) for pkt in raw_data
            ]

            total_bytes = sum(pkt.size for pkt in packets)
            bitrate = (total_bytes * 8) / (update_interval/1000) if update_interval > 0 else 0
    
            # Добавление пакетов в буфер BS для конкретного UE
        for packet in packets:
            success = buffer.ADD_PACKET(packet, current_time)
            if not success:
                print(f"BS: Пакет для UE {target_ue_id} отброшен (буфер полный)")
    
            # Логирование статистики
            status = buffer.GET_UE_STATUS(current_time)['per_ue'].get(target_ue_id, {})
            print(f"\nUE {target_ue_id} [DL]:")
            print(f"Сгенерировано пакетов: {len(packets)}")
            print(f"TTL пакетов: {ttl_ms} мс")
            print(f"Скорость: {bitrate/1e6:.2f} Mbps")
            print(f"Текущий размер буфера: {status.get('size', 0)} байт")
            print(f"Отброшено: {status.get('dropped', 0)}")        
    
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
            'total_size': 0,
            'total_packets': 0,
            'total_dropped': 0,
            'total_expired': 0,
            'avg_delay': 0.0,
            'max_delay': 0,
            'per_ue': {},  # Явная инициализация
            'per_ue_avg': {}
        }
            
        total_delay = 0
        packet_count = 0
        
        for ue_id, buffer in self.ue_buffers.items():
            # Получаем статус через GET_UE_STATUS
            buffer_status = buffer.GET_UE_STATUS(current_time)
            ue_status = buffer_status['per_ue'].get(ue_id, {})
            
            status['per_ue'][ue_id] = {
            'size': ue_status.get('size', 0),
            'packet_count': ue_status.get('packet_count', 0),
            'oldest_delay': ue_status.get('oldest_delay', 0),
            'avg_delay': ue_status.get('avg_delay', 0.0),
            'dropped': buffer.dropped.get(ue_id, 0),
            'expired': buffer.expired.get(ue_id, 0)
        }
            
            # Агрегируем показатели
            status['total_size'] += ue_status.get('size', 0)
            status['total_packets'] += ue_status.get('packet_count', 0)
            status['total_dropped'] += buffer.dropped.get(ue_id, 0)
            status['total_expired'] += buffer.expired.get(ue_id, 0)
            
            # Рассчитываем задержки
            if ue_status.get('packet_count', 0) > 0:
                total_delay += ue_status['avg_delay'] * ue_status['packet_count']
                packet_count += ue_status['packet_count']
            
            status['max_delay'] = max(
                status['max_delay'],
                ue_status.get('oldest_delay', 0)
            )
            
            # Средняя загрузка буфера (нормализованная)
            max_size = buffer.per_ue_max
            current_size = ue_status.get('size', 0)
            status['per_ue_avg'][ue_id] = current_size / max_size if max_size > 0 else 0
        
        # Расчёт средней задержки
        if packet_count > 0:
            status['avg_delay'] = total_delay / packet_count
            
        return status

    def CLEAR_ALL_BUFFERS(self) -> None:
        """
        Полностью очищает все буферы базовой станции, удаляя данные всех пользователей.
        """
        for ue_id in list(self.ue_buffers.keys()):  # Используем list для безопасной итерации
            buffer = self.ue_buffers[ue_id]
            buffer.DESTROY_UE_PACKETS(ue_id)  # Очистка буфера конкретного UE
            
        print("Все буферы базовой станции успешно очищены")

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
        Packet(size=1500, ue_id=1, creation_time=current_time - 300, ttl_ms=500),
        
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
        ue_id=1, 
        max_bytes=3000, 
        bits_per_rb=229, 
        current_time=current_time
    )
    extracted2, size2 = bs.ue_buffers[2].GET_PACKETS(
        ue_id=2, 
        max_bytes=4000, 
        bits_per_rb=229, 
        current_time=current_time
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
    packets = [Packet(size=500, ue_id=2, creation_time=current_time - 100, ttl_ms=1000),
               Packet(size=500, ue_id=2, creation_time=current_time - 200, ttl_ms=1000),
               Packet(size=500, ue_id=2, creation_time=current_time - 100, ttl_ms=1000)]
    for pkt in packets:
        bs.ue_buffers[pkt.ue_id].ADD_PACKET(pkt, current_time)

    print("\n[Этап 3] После 801 мс:")
    status = bs.GET_GLOBAL_BUFFER_STATUS(current_time)
    for ue_id in [1, 2]:
        ue_data = status['per_ue'].get(ue_id, {})
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
    assert status['total_size'] == 0, "Буфер не очищен"
    print("Все буферы пусты")

    # Ассерты
    #assert bs.buffer.sizes[1] == 3000, "Некорректный размер буфера UE1"
    #assert bs.buffer.expired[2] == 3, "Не удалены все устаревшие пакеты UE2"
    #assert bs.buffer.dropped[2] == 0, "Ложное отбрасывание пакетов UE2"

    print("\n=== Тест завершён успешно ===")


if __name__ == "__main__":
    test_bs_buffer_fifo()