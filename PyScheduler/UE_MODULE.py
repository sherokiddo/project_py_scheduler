"""
#------------------------------------------------------------------------------
# Модуль: UE_MODULE - Модель пользовательского устройства (UE) для сети LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет классы и методы для моделирования пользовательских устройств
# в сети LTE, включая их перемещение, генерацию трафика, управление буфером
# и оценку качества канала.
#
# Версия: 1.0.6
# Дата последнего изменения: 2025-11-05
# Автор: Брагин Кирилл, Норицин Иван, Дворников Андрей
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
#
#   v1.0.6 - 2025-11-05
#      - Интегрирован фактори-паттерн из MOBILITY_MODEL.
#      - Изменен SET_MOBILITY_MODEL в UserEquipment для корректного первого шага движения
#        моделей RandomWaypoint и RandomDirection.
#      - Добавлено динамическое установление атрибутов в UPD_POSITION.
#------------------------------------------------------------------------------
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import GLOBALS
import numpy as np
from TRAFFIC_MODEL import MMPPModel, OnOffModel, PoissonModel


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
        self.priority = priority  # для реализации QoS буфера нужно будет уйти от FIFO
        self.retry_count = 0  # Заготовка для HARQ

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
        self.dropped_info = []  # Параметры отброшенных пакетов

    def ADD_PACKET(
        self,
        packet_size: int,
        creation_time: int,
        current_time: int,
        priority: int = 0,
        ttl_ms: int = 1000,
    ) -> bool:
        """
        Добавить пакет в буфер. Пока я понятия не имею, по каким моделям мы
        будем генерировать трафик и каким макаром, но сделал такую заглушку

        Args:
            Наследуется у класса Packet

        Returns:
            bool: True, если пакет добавлен, False если отброшен
        """
        # Шаг 1: Удаление устаревших пакетов перед добавлением
        self.queue = deque([p for p in self.queue if current_time - p["creation_time"] <= ttl_ms])

        # Обновление current_size после очистки
        self.current_size = sum(p["size"] for p in self.queue)

        # Шаг 2: Проверка на переполнение после очистки
        if self.current_size + packet_size > self.max_size:
            self.dropped_packets += 1
            self.dropped_info.append(
                {
                    "size": packet_size,
                    "creation_time": creation_time,
                    "priority": priority,
                    "reason": "overflow",
                }
            )
            return False

        # Шаг 3: Добавление нового пакета
        self.queue.append(
            {"size": packet_size, "creation_time": creation_time, "priority": priority}
        )
        self.current_size += packet_size
        return True

        # @sherokiddo: "Возможно, у пакета появится атрибут метки QoS или приоритет
        # заглушку для него сделал. В дальнейшем реализовать функцию CHCK_PRIORITY или CHCK_PR
        # а также реализовать логику переполнения буфера и отбрасывания пакетов
        # а также, добавить возможность менять приоритет пакета через метод
        # а напоследок, метод для получения пакетов определенного приоритета GET_PCKT_BY_PR"

    def GET_PACKETS(
        self, max_bytes: int, bits_per_rb: int, current_time: int, ttl_ms: int = 1000
    ) -> Tuple[List[Dict], int]:
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
            if current_time - p["creation_time"] > ttl_ms:
                expired.append(p)
            else:
                filtered.append(p)

        # Обновление статистики
        self.expired_packets += len(expired)
        self.dropped_info.extend(expired)

        # Обновление очереди
        self.queue = deque(filtered)
        self.current_size = sum(p["size"] for p in self.queue)

        selected = []
        total_bits = 0
        max_bits = max_bytes * 8  # Конвертация в биты

        while self.queue and total_bits < max_bits:
            packet = self.queue[0]
            packet_bits = packet["size"] * 8

            # Доступное место в текущей итерации
            remaining_bits = max_bits - total_bits
            fragment_bits = min(packet_bits, remaining_bits)
            fragment_size = fragment_bits // 8  # Размер фрагмента в байтах

            if fragment_size >= packet["size"]:
                # Весь пакет помещается
                selected_packet = self.queue.popleft()
                selected.append(selected_packet)
                total_bits += packet_bits
                self.current_size -= selected_packet["size"]
            else:
                # Создание фрагмента
                fragment = {
                    "size": fragment_size,
                    "creation_time": packet["creation_time"],
                    "priority": packet["priority"],
                    "parent_id": id(packet),
                }
                selected.append(fragment)
                total_bits += fragment_bits

                # Обновление исходного пакета
                self.queue[0]["size"] -= fragment_size
                self.current_size -= fragment_size

        return selected, total_bits // 8

    # @sherokiddo: при переводе из бит в байты работает округление
    # оно приводит к погрешности, надо бы исправить....

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
                "size": 0,
                "packet_count": 0,
                "oldest_packet_delay": 0,
                "average_delay": 0.0,
                "utilization": 0.0,
            }

        delays = [current_time - p["creation_time"] for p in self.queue]
        return {
            "size": self.current_size,
            "packet_count": len(self.queue),
            "oldest_packet_delay": max(delays),
            "average_delay": sum(delays) / len(delays),
            "utilization": (self.current_size / self.max_size) * 100,
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

    # @sherokiddo: "Возможно пригодится метод для удаления пакетов, у которых
    # капнула уже большая задержка. типа REMOVE_OLD_PCKT


class UserEquipment:
    """
    Класс, представляющий пользовательское устройство (UE) в сети LTE.
    """

    def __init__(
        self,
        UE_ID: int,
        x: float = 0.0,
        y: float = 0.0,
        buffer_size: int = 1048576,
        ue_class: str = "pedestrian",
        indoor_boundaries: Tuple[float, float, float, float] = (0, 0, 0, 0),
    ):
        """
        Инициализация пользовательского устройства.

        Args:
            UE_ID (int): Уникальный идентификатор пользователя.
            x (float, optional): Начальная координата X (м). По умолчанию 0.0.
            y (float, optional): Начальная координата Y (м). По умолчанию 0.0.
            buffer_size (int, optional): Размер буфера (байты). По умолчанию 1048576.
            ue_class (str, optional): Класс UE. По умолчанию "pedestrian".
            indoor_boundaries (Tuple[float, float, float, float], optional): Границы
            помещения, в котором находится UE (при классе UE "indoor").
                По умолчанию (0, 0, 0, 0).

        """
        self.UE_ID = UE_ID
        self.position = (x, y)  # Координаты (x, y) в метрах
        self.ue_class = ue_class

        self.velocity_min = 0.0  # Минимальная скорость в м/с
        self.velocity_max = 0.0  # Максимальная скорость в м/с

        self.is_indoor = False  # Находится ли UE в помещении
        self.indoor_boundaries = indoor_boundaries  # Границы помещения

        self._set_scenario_parameters()
        self.serving_bs = None
        self.mobility_model = None  # Установить позже
        self.velocity = 0.0  # Скорость в м/с
        self.direction = 0.0  # Направление в радианах (ранее angle)

        self.coordinates = [self.position]  # Координаты

        # Буфер данных
        self.buffer = Buffer(buffer_size)

        # Модель трафика
        self.traffic_model = None  # Установить позже

        # Параметры канала связи
        self.serving_bs = None  # Установить позже
        self.cqi = 1  # Текущий CQI (1-15)
        self.cqi_subband = []
        self.SINR = 0.0  # Текущее отношение сигнал/шум+помехи в dB

        self.SINR_values = []  # Временный параметр для демонстрации результатов
        self.CQI_values = []  # Временный параметр для демонстрации результатов

        self.UE_height = 0.0
        self.dist_to_BS_2D = 0.0  # 2D-Расстояние до базовой станции в метрах
        self.dist_to_BS_2D_in = 0.0  # 2D-Расстояние до БС в (часть помещения)
        self.dist_to_BS_2D_out = 0.0  # 2D-Расстояние до БС в (часть улицы)
        self.dist_to_BS_3D = 0.0  # 3D-Расстояние до базовой станции в метрах

        # Параметры для алгоритмов планирования
        self.current_throughput = 0.0  # Текущая пропускная способность (бит/с)
        self.average_throughput = 0.0  # Средняя пропускная способность (бит/с)

        self.assigned_rbs = []  # Список выделенных ресурсных блоков
        self.mcs_index = 0  # Индекс MCS (0-28)

        # Статистика
        self.last_transmitted_bits = 0
        self.total_transmitted_bits = 0
        self.total_transmitted_packets = 0
        self.total_dropped_packets = 0

        # Статистика для DL
        self.current_dl_throughput = 0.0
        self.average_dl_throughput = 0.0
        self.current_dl_throughput_per_qci = {}
        self.average_dl_throughput_per_qci = {}
        self.total_dl_transmitted_bits = 0
        self.total_transmitted_dl_packets = 0
        self.total_dropped_dl_packets = 0

    def PROCESS_DCI(self, tti: int, bitmap: List[int]) -> None:
        """
        Обработка Downlink Control Information (имитация).

        Args:
            tti (int): Значение TTI.
            bitmap (List[int]): Распределение RBG для пользователя.

        """
        self.allocated_rbg = [rbg_idx for rbg_idx, bit in enumerate(bitmap) if bit == 1]
        print(f"UE{self.id} получил DCI (TTI {tti}): RBG {self.allocated_rbg}")

    def SET_MOBILITY_MODEL(self, model: str, **kwargs):
        """
        Устанавливает модель мобильности.

        Args:
            model (str): Название модели движения. Доступные модели:
                RandomWalk
                    - pause_time (float, optional): Время паузы между блужданиями.
                    - velocity_min (float): Минимальная скорость.
                    - velocity_max (float): Максимальная скорость.
                RandomWaypoint
                    - pause_time (float, optional): Время паузы между движениями.
                    - velocity_min (float): Минимальная скорость.
                    - velocity_max (float): Максимальная скорость.
                RandomDirection
                    - pause_time (float, optional): Время паузы между движениями.
                GaussMarkov
                    - alpha (float, optional): Параметр памяти модели.
                    - boundary_threshold (float, optional): Порог приближения к границе.
                DiagonalWalk
                    - pause_time (int): Время паузы.
                    - bs (BaseStation): Экземпляр базовой станции.

        Пример:
            SET_MOBILITY_MODEL('RandomWalk', pause_time=2.0, velocity_min=1.0)
        """
        from MOBILITY_MODEL import MobilityInterface

        mobility = MobilityInterface.create(model=model, ue=self, **kwargs)
        self.mobility_model = mobility

    # def SET_CH_MODEL(self, model) -> None:
    #     """
    #     Установить модель радиоканала для пользователя.

    #     Args:
    #         model (ChannelModel): Модель радиоканала.

    #     """
    #     from CHANNEL_MODEL import ChannelModel

    #     if not isinstance(model, ChannelModel):
    #         raise TypeError(f"Некорректный тип модели канала: {type(model).__name__}")

    #     self.channel_model = model

    def SET_TRAFFIC_MODEL(self, model) -> None:
        """
        Установить модель генерации трафика для пользователя.

        Args:
            model (TrafficModel): Модель генерации трафика.

        """
        if not isinstance(model, (PoissonModel, OnOffModel, MMPPModel)):
            raise TypeError(f"Некорректный тип модели трафика: {type(model).__name__}")

        self.traffic_model = model

        # @IvanNoritsin: Нужен базовый класс для моделей трафика для более
        # корректной валидации.

    def UPD_POSITION(self, update_interval: int) -> None:
        """Обновить позицию пользователя согласно модели передвижения."""

        new_pos, new_vel, new_dir = self.mobility_model.update(time_ms=update_interval)

        self.position = new_pos
        self.velocity = new_vel
        self.direction = new_dir
        self.coordinates.append(self.position)

        # Обновление 2D и 3D расстояний до базовой станции
        if self.is_indoor:
            self._calculate_distances_to_BS(self.serving_bs.position, self.serving_bs.height)

        else:
            self.dist_to_BS_2D = np.hypot(
                self.position[0] - self.serving_bs.position[0],
                self.position[1] - self.serving_bs.position[1],
            )

            self.dist_to_BS_2D_out = self.dist_to_BS_2D

            self.dist_to_BS_3D = np.hypot(
                self.dist_to_BS_2D, self.serving_bs.height - self.UE_height
            )

    def UPD_CH_QUALITY(self, channel_update_interval: int = 1) -> None:
        """
        Обновить качество канала связи согласно модели распространения сигнала.

        """
        from CHANNEL_MODEL import RMaModel, UMaModel, UMiModel

        if not self.serving_bs:
            raise ValueError("Ошибка! UE не подключен к базовой станции! {}".format(self.UE_ID))

        if not self.serving_bs.channel_model:
            raise ValueError("Ошибка! У базовой станции не инициализирована модель канала!")

        if self.is_indoor:
            self._calculate_distances_to_BS(
                self.serving_bs.position, self.serving_bs.height
            )
        else:
            self.dist_to_BS_2D = np.hypot(
                self.position[0] - self.serving_bs.position[0],
                self.position[1] - self.serving_bs.position[1],
            )
            self.dist_to_BS_2D_out = self.dist_to_BS_2D
            self.dist_to_BS_3D = np.hypot(
                self.dist_to_BS_2D,
                self.serving_bs.height - self.UE_height
            )

        # displacement = np.hypot(
        #     self.position[0] - self._last_ch_position[0],
        #     self.position[1] - self._last_ch_position[1]
        # )
        # self._last_ch_position = self.position

        STATIC_UE_VELOCITY = 0.2
        effective_velocity = max(self.velocity, STATIC_UE_VELOCITY)
        displacement = effective_velocity * (channel_update_interval / 1000.0)
        #TODO: TDL некорректно работает при displacement = 0
        # этот метод нужно хорошо протестировать и проверить
        # изучить residual movement в 3GPP. Пока решение временное
        
        if isinstance(self.serving_bs.channel_model, RMaModel):
            if self.UE_height == 0.0:
                if self.is_indoor == True:
                    self.UE_height = np.random.uniform(1, 10)
                else:
                    self.UE_height = 1.0

        if isinstance(self.serving_bs.channel_model, (UMaModel, UMiModel)):
            if self.UE_height == 0.0:
                if self.is_indoor == True:
                    N_fl = np.random.uniform(4, 8)
                    n_fl = np.random.uniform(1, N_fl)
                    self.UE_height = 3 * (n_fl - 1) + 1.5
                else:
                    self.UE_height = 1.5

        SINR_on_RB = self.serving_bs.channel_model.calculate_SINR(
            self.UE_ID,
            displacement,
            self.dist_to_BS_2D,
            self.dist_to_BS_2D_in,
            self.dist_to_BS_3D,
            self.UE_height,
            self.ue_class,
        )

        # Wideband SINR и CQI
        self.SINR = np.mean(SINR_on_RB)
        self.cqi = self.SINR_TO_CQI(self.SINR)
        if self.serving_bs.enable_tdl:
            subband_size = GLOBALS.SUBBAND_SIZE[self.serving_bs.bandwidth]
            # Subband CQI
            self.cqi_subband = []
            for i in range(0, len(SINR_on_RB), subband_size):
                sinr_subband = np.mean(SINR_on_RB[i : i + subband_size])
                self.cqi_subband.append(self.SINR_TO_CQI(sinr_subband))

        self.SINR_values.append(self.SINR)
        self.CQI_values.append(self.cqi)

    def GEN_TRFFC(self, current_time: int, update_interval: int) -> None:
        """
        Сгенерировать пакеты трафика согласно модели и добавить их в буфер.

        Args:
            current_time (int): Текущее время симуляции (мс).
            update_interval (int): Интервал обновления состояния UE (мс).

        """
        if not self.traffic_model:
            raise ValueError(
                "Ошибка! Модель генерации трафика не определена! {}".format(self.UE_ID)
            )

        if isinstance(self.traffic_model, PoissonModel):
            packets = self.traffic_model.generate_traffic(current_time, update_interval)

        if isinstance(self.traffic_model, OnOffModel) or isinstance(self.traffic_model, MMPPModel):
            packets = self.traffic_model.generate_traffic(self.UE_ID, current_time, update_interval)

        # Скорость поступления пакетов в буфер
        total_bytes = sum(p["size"] for p in packets)
        total_bits = total_bytes * 8
        interval_seconds = update_interval / 1000
        bitrate = total_bits / interval_seconds if interval_seconds > 0 else 0

        for packet in packets:
            if not self.buffer.ADD_PACKET(
                packet_size=packet["size"],
                creation_time=packet["creation_time"],
                current_time=current_time,
                priority=packet["priority"],
                ttl_ms=1000,
            ):
                self.total_dropped_packets += 1
                print(f"UE {self.UE_ID}: Пакет {packet['size']}B отброшен (буфер полный)")

        status = self.buffer.GET_STATUS(current_time)
        print(
            f"Интервал [{current_time - update_interval}-{current_time} ms]: Создано {len(packets)} пакетов"
        )
        print(f"Скорость поступления: {bitrate:.2f} бит/с")
        print(f"Статус буфера: {status}")

    def UPD_THROUGHPUT_BPS(self, bits_transmitted: int, time_interval_ms: int):
        """
        Обновить статистику пропускной способности.

        Args:
            bits_transmitted (int): Количество переданных бит.
            time_interval_ms (int): Интервал времени (мс).

        """
        # Текущая пропускная способность в бит/с
        self.current_throughput = (
            (bits_transmitted * 1000) / time_interval_ms if time_interval_ms > 0 else 0
        )

        # EWMA обновление average_throughput для Proportional Fair
        alpha = 0.002  # временный хардкод, вывести в управление.
        average_throughput_past = self.average_throughput
        self.average_throughput = (
            1 - alpha
        ) * average_throughput_past + alpha * self.current_dl_throughput

        # Текущее переданное количество бит
        self.last_transmitted_bits = bits_transmitted

        # Обновление общей статистики
        self.total_transmitted_bits += bits_transmitted

    def UPD_DL_THROUGHPUT_BPS(
        self,
        bits_dl_transmitted: int,
        bits_transmitted_per_qci: Dict[int, int],
        time_interval_ms: int,
    ):
        """
        Обновить статистику пропускной способности в DL.

        Args:
            bits_dl_transmitted (int): Количество переданных бит в DL.
            bits_transmitted_per_qci (Dict[int, int]): Переданные биты по каждому QCI.
            time_interval_ms (int): Интервал времени (мс).

        """
        def _to_bps(bits: int) -> float:
            """Конвертация переданных бит за интервал в бит/с."""
            return (bits * 1000) / time_interval_ms if time_interval_ms > 0 else 0

        # Текущая пропускная способность в бит/с
        self.current_dl_throughput = _to_bps(bits_dl_transmitted)

        # EWMA обновление average_throughput для Proportional Fair
        alpha = 0.002  # временный хардкод, вывести в управление.
        average_throughput_past = self.average_throughput
        self.average_throughput = (
            1 - alpha
        ) * average_throughput_past + alpha * self.current_dl_throughput

        # Обновляем все известные QCI на каждом TTI:
        # если передачи по QCI не было, current throughput становится 0,
        # а average throughput плавно затухает по EWMA.
        tracked_qcis = (
            set(self.current_dl_throughput_per_qci)
            | set(self.average_dl_throughput_per_qci)
            | set(bits_transmitted_per_qci)
        )
        for qci in tracked_qcis:
            bits_per_qci = bits_transmitted_per_qci.get(qci, 0)
            throughput_qci = _to_bps(bits_per_qci)
            self.current_dl_throughput_per_qci[qci] = throughput_qci

            average_throughput_qci_past = self.average_dl_throughput_per_qci.get(qci, 0)
            self.average_dl_throughput_per_qci[qci] = (
                (1 - alpha) * average_throughput_qci_past
                + alpha * throughput_qci
            )

        # Текущее переданное количество бит
        self.last_transmitted_bits = bits_dl_transmitted

        # Обновление общей статистики
        self.total_dl_transmitted_bits += bits_dl_transmitted

        #TODO: Сделать ручку для регулирования порога EWMA (alpha=) в симуляции

    def UPD_BUFFER(self, current_time: int):
        """Обновление задержки пакетов в буфере"""
        for packet in self.buffer.queue:
            packet["age"] = current_time - packet["creation_time"]

        # Удаление устаревших пакетов (пример: TTL = 1000 мс)
        self.buffer.queue = deque([p for p in self.buffer.queue if p["age"] <= 1000])

    def GET_BUFFER_STATUS(self, current_time: int) -> Dict:
        """
        Получить текущий статус буфера.

        Args:
            current_time (int): Текущее время симуляции (мс).

        Returns:
            Dict: Статус буфера.

        """
        return self.buffer.GET_STATUS(current_time)

    def GET_CH_QUALITY(self) -> Dict:
        """
        Получить текущее качество канала.

        Returns:
            Dict: Параметры качества канала.

        """
        return {
            "cqi": self.cqi,
            "SINR": self.SINR,
            "distance": self.dist_to_BS_2D,  # возможно оно тут нафиг не надо я пока не вставлял расчеты SINR
        }

    def SINR_TO_CQI(self, SINR: float) -> int:
        """
        Преобразовать SINR в CQI согласно спецификации LTE.

        Args:
            SINR (float): Отношение сигнал/шум+помехи (дБ).

        Returns:
            int: Значение CQI (1-15).

        """
        if SINR <= -6.934:
            return 1
        elif SINR >= 22.976:
            return 15
        else:
            # Линейная интерполяция
            step = (22.976 + 6.934) / 14
            return int(1 + (SINR + 6.934) / step)

    def _calculate_distances_to_BS(self) -> None:
        """
        Вычисляет расстояние от пользователя до базовой станции с учетом
        нахождения внутри здания. Разделяет расстояние на часть внутри здания
        (indoor) и снаружи (outdoor).

        """
        x_min, x_max, y_min, y_max = self.indoor_boundaries

        ue_x, ue_y = self.position
        bs_x, bs_y = self.serving_bs.position

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
        self.dist_to_BS_3D = np.hypot(self.dist_to_BS_2D, self.serving_bs.height - self.UE_height)

    def _set_scenario_parameters(self):
        """
        Установка параметров в зависимости от класса UE.

        """
        self.mean_direction = np.random.randint(0, 360)
        self.is_indoor = self.ue_class == "indoor"

        params = {
            "indoor": (0.0, 1.0, 0.5),
            "pedestrian": (0.5, 1.7, 1.2),
            "cyclist": (2.0, 5.5, 3.9),
            "car": (0.0, 16.7, 11.1),
        }

        if self.ue_class not in params:
            raise ValueError(f"Недопустимое значение типа передвижения устройства: {self.ue_class}")

        self.velocity_min, self.velocity_max, self.mean_velocity = params[self.ue_class]

        # Значение границ помещения по умолчанию, если параметр не был задан
        if self.is_indoor and self.indoor_boundaries == (0, 0, 0, 0):
            x, y = self.position
            self.indoor_boundaries = (x - 10, x + 10, y - 10, y + 10)


class UECollection:
    """
    Класс для управления коллекцией пользовательских устройств.
    Задел, чтобы не создавать их вручную и можно было регулировать количество.
    """

    def __init__(self):
        """
        Инициализация коллекции UE.

        """
        self.users = {}  # Словарь {UE_ID: UserEquipment}

    def ADD_USER(self, ue: UserEquipment) -> bool:
        """
        Добавить пользователя в коллекцию.

        Args:
            ue (UserEquipment): Объект пользовательского устройства.

        Returns:
            bool: True, если пользователь добавлен, False если уже существует.

        """
        if ue.UE_ID in self.users:
            return False

        self.users[ue.UE_ID] = ue
        return True

    def REMOVE_USER(self, UE_ID: int) -> bool:
        """
        Удалить пользователя из коллекции.

        Args:
            UE_ID (int): Идентификатор пользователя.

        Returns:
            bool: True, если пользователь удален, False если не найден.

        """
        if UE_ID in self.users:
            del self.users[UE_ID]
            return True
        return False

    def GET_USER(self, UE_ID: int) -> Optional[UserEquipment]:
        """
        Получить пользователя по ID.

        Args:
            UE_ID (int): Идентификатор пользователя.

        Returns:
            Optional[UserEquipment]: UserEquipment или None, если пользователь
            не найден.

        """
        return self.users.get(UE_ID)

    def GET_ALL_USERS(self) -> List[UserEquipment]:
        """
        Получить список всех пользователей.

        Returns:
            List[UserEquipment]: Список всех пользователей.

        """
        return list(self.users.values())

    def UPDATE_ALL_USERS(self, current_time: int,
                         update_interval: int,
                         mobility_update_interval: int = None,
                         channel_update_interval: int = None,
                         ):
        """
        Обновить состояние всех пользователей в коллекции.

        Args:
            current_time (int): Текущее время симуляции (мс).
            update_interval (int): Интервал обновления состояния UE (мс).

        """
        # Если отдельные интервалы не заданы — старое поведение
        mob_interval = mobility_update_interval or update_interval
        ch_interval = channel_update_interval or update_interval

        for ue in self.users.values():
            # Обновление позиции
            if current_time % mob_interval == 0:
                ue.UPD_POSITION(mob_interval)

            # Обновление качества канала
            if current_time % ch_interval == 0:
                ue.UPD_CH_QUALITY(ch_interval)

    def GET_ACTIVE_USERS(self) -> List[UserEquipment]:
        """
        Получить список активных пользователей (с данными в буфере).

        Returns:
            List[UserEquipment]: Список активных пользователей.

        """
        return [ue for ue in self.users.values() if ue.buffer.GET_STATUS(0)["size"] > 0]

    def GET_USERS_FOR_SCHEDULER(self) -> List:
        """
        Получить данные о пользователях для планировщика.

        Returns:
            List: Список с данными о пользователях.

        """
        users_data = []
        for ue in self.users.values():
            users_data.append(
                {
                    "UE_ID": ue.UE_ID,
                    "cqi": ue.cqi,
                    "sbb_cqi": ue.cqi_subband,
                    "ue": ue,
                }
            )

        return users_data

    def ADD_RANDOM_USERS(
        self,
        num_ue: int,
        x_min: float = None,
        x_max: float = None,
        y_min: float = None,
        y_max: float = None,
        ue_class: str = "random",
    ):
        """
        Добавить указанное количество пользовательских устройств (UE) в
        коллекцтю со случайными координатами и классом пользователя. Если класс
        UE указан как "random" то каждому UE случайно присваивается один из
        доступных классов (кроме "indoor"). В противном случае всем UE назначается
        указанный класс. Генерируемые значения могут быть зафиксированы при помощи
        указания сида в симуляции.

        Args:
            num_ue (int): Количество UE для добавления.
            x_min (float, optional): Минимальная координата по оси X.
                По умолчанию -1000.
            x_max (float, optional): Максимальная координата по оси X.
                По умолчанию 1000.
            y_min (float, optional): Минимальная координата по оси Y.
                По умолчанию -1000.
            y_max (float, optional): Максимальная координата по оси Y.
                По умолчанию 1000.
            ue_class (str, optional): Класс UE. По умолчанию "random".

        """
        rng = np.random.default_rng(GLOBALS.SEED)

        if ue_class == "random":
            available_classes = ["pedestrian", "cyclist", "car"]
            ue_classes = rng.choice(available_classes, size=num_ue, replace=True)
        else:
            ue_classes = [ue_class] * num_ue

        start_id = max(self.users.keys(), default=0) + 1

        for i in range(num_ue):
            ue_id = start_id + i
            x_position = rng.uniform(x_min, x_max)
            y_position = rng.uniform(y_min, y_max)

            self.users[ue_id] = UserEquipment(
                UE_ID=ue_id, x=x_position, y=y_position, ue_class=ue_classes[i]
            )

    def SET_MOBILITY_MODEL(self, model, ue_ids: List[int] = None, **kwargs):
        """
        Установить модель передвижения пользователей в коллекции. Если ue_ids
        задан как None, то модель применится ко всем UE в коллекции.

        Args:
            ue_ids (List[int], optional): Список ID пользователей, к которым
            необходимо применить модель. По умолчанию None.
            model (str): Название модели движения. Доступные модели:
                RandomWalk
                    - pause_time (float, optional): Время паузы между блужданиями.
                    - velocity_min (float): Минимальная скорость.
                    - velocity_max (float): Максимальная скорость.
                RandomWaypoint
                    - pause_time (float, optional): Время паузы между движениями.
                    - velocity_min (float): Минимальная скорость.
                    - velocity_max (float): Максимальная скорость.
                RandomDirection
                    - pause_time (float, optional): Время паузы между движениями.
                GaussMarkov
                    - alpha (float, optional): Параметр памяти модели.
                    - boundary_threshold (float, optional): Порог приближения к границе.
                DiagonalWalk
                    - pause_time (int): Время паузы.
                    - bs (BaseStation): Экземпляр базовой станции.

        Пример:
            SET_MOBILITY_MODEL('RandomWalk', pause_time=2.0, velocity_min=1.0)

        """
        for ue in self.users.values():
            if ue_ids is None or ue.UE_ID in ue_ids:
                ue.SET_MOBILITY_MODEL(model, **kwargs)

    def SET_TRAFFIC_MODEL(self, model, ue_ids: List[int] = None):
        """
        Установить модель генерации трафика для пользователей в коллекции. Если
        ue_ids задан как None, то модель применится ко всем UE в коллекции.

        Args:
            model (TrafficModel): Модель генерации трафика.
            ue_ids (List[int], optional): Список ID пользователей, к которым
            необходимо применить модель. По умолчанию None.

        """
        for ue in self.users.values():
            # if ue_ids is None or ue.UE_ID in ue_ids:
            ue.SET_TRAFFIC_MODEL(model)

    def REG_USERS_TO_BS(self, bs):
        """
        Регистрация всех пользователей коллекции в базовой станции.

        Args:
            bs (BaseStation): Объект базовой станции.

        """
        for ue in self.users.values():
            bs.REG_UE(ue)


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

    #TODO: Тесты выше, они вообще работают? Проверить и перенести либо в модуль
    # тестов либо в папку с тестами
