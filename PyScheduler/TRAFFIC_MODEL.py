"""
#------------------------------------------------------------------------------
# Модуль: TRAFFIC_MODEL - Модели генерации сетевого трафика
#------------------------------------------------------------------------------
# Описание:
#   Модуль содержит реализации статистических моделей генерации сетевого трафика:
#   1. Пуассоновская модель - пакеты генерируются с экспоненциальными интервалами
#   2. ON/OFF модель - устройства периодически переключаются между активными (ON)
#      и неактивными (OFF) состояниями, генерируя трафик только в активной фазе
#
#   Модели используются для имитации поведения реального сетевого трафика в симуляциях
#   и тестовых сценариях.
#
# Версия: 1.0.0
# Дата последнего изменения: 2025-04-13
# Автор: Норицин Иван
# Версия Python Kernel: 3.12.9
#------------------------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


class TrafficType(Enum):
    """
    Типы трафика согласно 3GPP.

    Каждый тип имеет соответствующий QCI (QoS Class Identifier).
    """

    VOIP = "voip"  # Голосовые звонки
    VIDEO_CALL = "video_call"  # Видеозвонки
    VIDEO_STREAM = "video_stream"  # Потоковое видео
    WEB = "web"  # Веб-браузинг
    FILE_TRANSFER = "file_transfer"  # Скачивание файлов
    GAMING = "gaming"  # Онлайн игры
    BACKGROUND = "background"  # Фоновый трафик

    def get_qci(self) -> int:
        """
        Получить QCI для типа трафика.

        Returns:
            int: QCI согласно 3GPP TS 23.203
        """
        qci_mapping = {
            TrafficType.VOIP: 1,  # GBR, 100ms delay budget
            TrafficType.VIDEO_CALL: 2,  # GBR, 150ms delay budget
            TrafficType.VIDEO_STREAM: 7,  # GBR, 100ms delay budget
            TrafficType.WEB: 9,  # Non-GBR
            TrafficType.FILE_TRANSFER: 9,  # Non-GBR
            TrafficType.GAMING: 3,  # GBR, 50ms delay budget
            TrafficType.BACKGROUND: 9,  # Non-GBR
        }
        return qci_mapping[self]

    def get_delay_budget(self) -> int:
        """
        Получить delay budget (мс) для типа трафика.

        Returns:
            int: Максимальная задержка (мс)
        """
        delay_mapping = {
            TrafficType.VOIP: 100,
            TrafficType.VIDEO_CALL: 150,
            TrafficType.VIDEO_STREAM: 300,
            TrafficType.WEB: 300,
            TrafficType.FILE_TRANSFER: 1000,
            TrafficType.GAMING: 50,
            TrafficType.BACKGROUND: 1000,
        }
        return delay_mapping[self]


@dataclass
class Packet:
    """
    Пакет данных в LTE сети.

    Attributes:
        size: Размер пакета (байты)
        ue_id: ID пользователя
        creation_time: Время создания пакета (мс)
        qci: Quality Class Identifier (1-9)
        traffic_type: Тип трафика
        priority: Приоритет (0 = highest)
        ttl_ms: Time-to-live (мс)
        deadline: Абсолютный deadline (creation_time + delay_budget)
        is_fragment: Является ли пакет фрагментом
    """

    size: int
    ue_id: int
    creation_time: float
    qci: int = 9
    traffic_type: Optional[TrafficType] = None
    priority: int = 0
    ttl_ms: int = 1000
    deadline: Optional[float] = None
    is_fragment: bool = False

    def __post_init__(self):
        """Вычисляем deadline если не задан"""
        if self.deadline is None:
            if self.traffic_type is not None:
                delay_budget = self.traffic_type.get_delay_budget()
                self.deadline = self.creation_time + delay_budget
            else:
                # Дефолтный deadline
                self.deadline = self.creation_time + self.ttl_ms

    def age(self, current_time: int) -> int:
        """
        Возраст пакета в мс относительно текущего времени симуляции.

        Args:
            current_time: Текущее время симуляции (мс)

        Returns:
            int: Возраст пакета в миллисекундах
        """
        return current_time - self.creation_time

    def to_dict(self) -> Dict:
        """
        Конвертация в dict для legacy совместимости.

        Returns:
            Dict: Словарь с полями пакета
        """
        return {
            "size": self.size,
            "creation_time": self.creation_time,
            "priority": self.priority,
            "qci": self.qci,
            "ue_id": self.ue_id,
            "ttl_ms": self.ttl_ms,
            "deadline": self.deadline,
            "is_fragment": self.is_fragment,
        }

    @staticmethod
    def from_dict(data: Dict, ue_id: int) -> "Packet":
        """
        Создать Packet из dict (для legacy кода).

        Args:
            data: Словарь с полями пакета
            ue_id: ID пользователя

        Returns:
            Packet: Созданный пакет
        """
        return Packet(
            size=data["size"],
            ue_id=ue_id,
            creation_time=data["creation_time"],
            priority=data.get("priority", 0),
            qci=data.get("qci", 9),
            ttl_ms=data.get("ttl_ms", 1000),
            is_fragment=data.get("is_fragment", False),
        )


class ITrafficModel(ABC):
    """
    Базовый интерфейс для всех моделей трафика.

    Все модели должны наследоваться от этого класса и реализовать
    абстрактные методы.

    Паттерн: Strategy
    """

    def __init__(self, min_packet_size: int = 150, max_packet_size: int = 1500):
        """
        Базовая инициализация.

        Args:
            min_packet_size: Минимальный размер пакета (байты)
            max_packet_size: Максимальный размер пакета (байты)
        """
        self.min_packet_size = min_packet_size
        self.max_packet_size = max_packet_size

    @abstractmethod
    def generate_traffic(self, ue_id: int, current_time: int, update_interval: int) -> List[Packet]:
        """
        Генерация трафика за указанный интервал.

        ⚠️ ВАЖНО: Все модели должны принимать ue_id, даже если не используют!

        Args:
            ue_id: ID пользователя
            current_time: Текущее время симуляции (мс)
            update_interval: Интервал генерации (мс)

        Returns:
            List[Packet]: Список сгенерированных пакетов
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Получить название модели.

        Returns:
            str: Название модели (например, 'Poisson', 'OnOff')
        """
        pass

    def get_model_info(self) -> Dict:
        """
        Получить информацию о модели (параметры, состояние).

        Returns:
            Dict: Словарь с информацией о модели
        """
        return {
            "name": self.get_model_name(),
            "min_packet_size": self.min_packet_size,
            "max_packet_size": self.max_packet_size,
        }


class PoissonModel:
    """
    Пуассоновская модель генерации трафика.
    Пакеты создаются с интервалами, соответствующими экспоненциальному распределению,
    моделируя среднюю скорость генерации пакетов.
    """

    def __init__(
        self, packet_rate: float, min_packet_size: float = 150, max_packet_size: float = 1500
    ):
        """
        Инициализация модели Пуассоновского трафика.

        Args:
            packet_rate: Средняя интенсивность трафика (пакетов в секунду)
            min_packet_size: Минимальный размер пакета (по умолчанию 150 байт)
            max_packet_size: Максимальный размер пакета (по умолчанию 1500 байт)
        """
        self.packet_rate = packet_rate
        self.min_packet_size = min_packet_size
        self.max_packet_size = max_packet_size

    def generate_traffic(self, current_time: int, update_interval: int):
        """
        Генерация трафика за указанный интервал времени.

        Args:
            current_time: Текущее время моделирования (мс)
            update_interval: Интервал времени для генерации трафика (мс)

        Returns:
            Список словарей с характеристиками сгенерированных пакетов:
            [{
                'size': размер пакета (байт),
                'creation_time': время создания (мс),
                'priority': приоритет пакета
            }]
        """
        packets = []
        generate_time = current_time - update_interval
        end_generate = current_time

        mean_interval_ms = 1000.0 / self.packet_rate

        while generate_time < end_generate:
            interval = np.random.exponential(mean_interval_ms)
            generate_time += interval

            if generate_time > end_generate:
                break

            packet_size = np.random.randint(self.min_packet_size, self.max_packet_size)

            packets.append({"size": packet_size, "creation_time": generate_time, "priority": 0})

        return packets


class OnOffModel:
    """
    ON/OFF модель генерации трафика.
    Устройство чередует активные (ON) и неактивные (OFF) состояния.
    В состоянии ON пакеты генерируются с заданной интенсивностью.
    """

    def __init__(
        self,
        duration_on: float,
        duration_off: float,
        packet_rate: float,
        min_packet_size: float = 150,
        max_packet_size: float = 1500,
    ):
        """
        Инициализация ON/OFF модели трафика.

        Args:
            duration_on: Средняя длительность активной фазы (сек)
            duration_off: Средняя длительность неактивной фазы (сек)
            packet_rate: Интенсивность трафика в активной фазе (пакетов/сек)
            min_packet_size: Минимальный размер пакета (по умолчанию 150 байт)
            max_packet_size: Максимальный размер пакета (по умолчанию 1500 байт)
        """
        self.duration_on = duration_on
        self.duration_off = duration_off
        self.packet_rate = packet_rate
        self.min_packet_size = min_packet_size
        self.max_packet_size = max_packet_size

        self.device_states = {}

    def generate_traffic(self, UE_ID: int, current_time: int, update_interval: int):
        """
        Генерация трафика для конкретного устройства за указанный интервал времени.

        Args:
            UE_ID: Идентификатор устройства
            current_time: Текущее время моделирования (мс)
            update_interval: Интервал времени для генерации трафика (мс)

        Returns:
            Список словарей с характеристиками сгенерированных пакетов:
            [{
                'size': размер пакета (байт),
                'creation_time': время создания (мс),
                'priority': приоритет пакета
            }]
        """
        if UE_ID not in self.device_states:
            initial_state = "ON" if np.random.rand() > 0.5 else "OFF"

            if initial_state == "ON":
                on_state_time = np.random.exponential(self.duration_on)
                on_state_time = on_state_time * 1000
                end_state_time = current_time - update_interval + on_state_time

            elif initial_state == "OFF":
                off_state_time = np.random.exponential(self.duration_off)
                off_state_time = off_state_time * 1000
                end_state_time = current_time - update_interval + off_state_time

            self.device_states[UE_ID] = {"state": initial_state, "end_state_time": end_state_time}

        state_data = self.device_states[UE_ID]
        packets = []
        t = current_time - update_interval

        while t < current_time:
            if state_data["state"] == "ON":
                if t < state_data["end_state_time"] <= current_time:
                    end_generate = state_data["end_state_time"]

                    off_state_time = np.random.exponential(self.duration_off)
                    off_state_time = off_state_time * 1000
                    end_state_time = end_generate + off_state_time

                    self.device_states[UE_ID] = {"state": "OFF", "end_state_time": end_state_time}

                else:
                    end_generate = current_time

                mean_interval_ms = 1000.0 / self.packet_rate

                while t < end_generate:
                    interval = np.random.exponential(mean_interval_ms)
                    t += interval

                    if t > end_generate:
                        break

                    packet_size = np.random.randint(self.min_packet_size, self.max_packet_size)

                    packets.append({"size": packet_size, "creation_time": t, "priority": 0})

            elif state_data["state"] == "OFF":
                if t < state_data["end_state_time"] <= current_time:
                    end_off_state = state_data["end_state_time"]

                    on_state_time = np.random.exponential(self.duration_on)
                    on_state_time = on_state_time * 1000
                    end_state_time = end_off_state + on_state_time

                    self.device_states[UE_ID] = {"state": "ON", "end_state_time": end_state_time}

                else:
                    end_off_state = current_time

                t = end_off_state

            state_data = self.device_states[UE_ID]

        return packets


class MMPPModel:
    """
    Модель трафика с марковским модулированным пуассоновским процессом (MMPP).
    Модель описывает систему, которая может находиться в нескольких состояниях,
    каждое из которых характеризуется своей интенсивностью генерации пакетов.
    Переходы между состояниями происходят согласно марковскому процессу.
    """

    def __init__(
        self, packet_rates: List[float], min_packet_size: float = 150, max_packet_size: float = 1500
    ):
        """
        Инициализация MMPP модели трафика.

        Args:
            packet_rates: Список интенсивностей трафика для каждого состояния (пакетов/сек)
            min_packet_size: Минимальный размер пакета (по умолчанию 150 байт)
            max_packet_size: Максимальный размер пакета (по умолчанию 1500 байт)
        """
        self.transition_matrix = np.array([[0, 0.07, 0.03], [0.12, 0, 0.08], [0.4, 0.1, 0]])

        self.packet_rates = packet_rates
        self.min_packet_size = min_packet_size
        self.max_packet_size = max_packet_size

        self.device_states = {}

    def _get_next_state(self, current_state: int) -> (int, float):
        """
        Определение следующего состояния и времени до перехода.

        Args:
            current_state: Текущее состояние системы

        Returns:
            next_state: следующее состояние
            time_to_transition: время до перехода (мс)
        """
        rates = self.transition_matrix[current_state]
        total_rate = sum(rates)
        if total_rate == 0:
            return current_state, float("inf")

        time_to_transition = np.random.exponential(1 / total_rate) * 1000

        probabilities = rates / total_rate
        next_state = np.random.choice(3, p=probabilities)

        return next_state, time_to_transition

    def generate_traffic(self, UE_ID: int, current_time: int, update_interval: int) -> List[Dict]:
        """
        Генерация трафика для конкретного устройства за указанный интервал времени.

        Args:
            UE_ID: Идентификатор устройства
            current_time: Текущее время моделирования (мс)
            update_interval: Интервал времени для генерации трафика (мс)

        Returns:
            Список словарей с характеристиками сгенерированных пакетов:
            [{
                'size': размер пакета (байт),
                'creation_time': время создания (мс),
                'priority': приоритет пакета
            }]
        """
        if UE_ID not in self.device_states:
            initial_state = np.random.choice(3)
            next_state, time_to_transition = self._get_next_state(initial_state)

            self.device_states[UE_ID] = {
                "current_state": initial_state,
                "transition_time": current_time - update_interval + time_to_transition,
                "next_state": next_state,
            }

        state_data = self.device_states[UE_ID]
        packets = []
        t = current_time - update_interval

        while t < current_time:
            current_state = state_data["current_state"]
            transition_time = state_data["transition_time"]

            end_time = min(transition_time, current_time)

            if self.packet_rates[current_state] > 0:
                mean_interval_ms = 1000.0 / self.packet_rates[current_state]

                while t < end_time:
                    interval = np.random.exponential(mean_interval_ms)
                    t += interval

                    if t > end_time:
                        break

                    packet_size = np.random.randint(self.min_packet_size, self.max_packet_size)
                    packets.append({"size": packet_size, "creation_time": t, "priority": 0})
            else:
                t = end_time

            # Если наступило время перехода
            if transition_time <= current_time:
                state_data["current_state"] = state_data["next_state"]
                new_next_state, time_to_transition = self._get_next_state(
                    state_data["current_state"]
                )
                state_data["next_state"] = new_next_state
                state_data["transition_time"] = transition_time + time_to_transition

        return packets


def test_traffic_models():
    """
    Тестирование и визуализация работы моделей трафика.
    """
    poisson_model = PoissonModel(packet_rate=5)
    onoff_model = OnOffModel(duration_on=2, duration_off=3, packet_rate=25)
    mmpp_model = MMPPModel(packet_rates=[5, 20, 40])

    simulation_duration = 60000
    update_interval = 250

    traffic_poisson = []
    traffic_onoff = []
    traffic_mmpp = []

    for t in range(1, simulation_duration + 1):
        if t % update_interval == 0:
            packets_poisson = poisson_model.generate_traffic(
                current_time=t, update_interval=update_interval
            )

            packets_onoff = onoff_model.generate_traffic(
                UE_ID=1, current_time=t, update_interval=update_interval
            )

            packets_mmpp = mmpp_model.generate_traffic(
                UE_ID=1, current_time=t, update_interval=update_interval
            )

            traffic_poisson.extend(packets_poisson)
            traffic_onoff.extend(packets_onoff)
            traffic_mmpp.extend(packets_mmpp)

    timestamps_poisson = [packet["creation_time"] for packet in traffic_poisson]
    sizes_poisson = [packet["size"] for packet in traffic_poisson]

    timestamps_onoff = [packet["creation_time"] for packet in traffic_onoff]
    sizes_onoff = [packet["size"] for packet in traffic_onoff]

    timestamps_mmpp = [packet["creation_time"] for packet in traffic_mmpp]
    sizes_mmpp = [packet["size"] for packet in traffic_mmpp]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.stem(timestamps_poisson, sizes_poisson, label="Пакеты")
    plt.xlabel("Время (мс)")
    plt.ylabel("Размер пакета (байты)")
    plt.title("Пуассоновская модель трафика")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.stem(timestamps_onoff, sizes_onoff, label="Пакеты")
    plt.xlabel("Время (мс)")
    plt.ylabel("Размер пакета (байты)")
    plt.title("ON/OFF модель трафика")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.stem(timestamps_mmpp, sizes_mmpp, label="Пакеты")
    plt.xlabel("Время (мс)")
    plt.ylabel("Размер пакета (байты)")
    plt.title("MMPP модель трафика")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    test_traffic_models()
