"""
#------------------------------------------------------------------------------
# Модуль: TRAFFIC_MODEL - Модели генерации сетевого трафика
#------------------------------------------------------------------------------
# Описание:
#   Модуль содержит высокоуровневые инструменты генерации сетевого трафика:
#   1. SimpleGenerator - легковесный генератор для обратной совместимости.
#   2. PacketManager - продвинутый менеджер с поддержкой Multi-bearer и QoS (QCI).
#   3. Набор статистических моделей (Poisson, On/Off, MMPP), реализующих
#      интерфейс ITrafficModel (паттерн Strategy).
#
#   Модели используются для имитации поведения реального сетевого трафика в симуляциях
#   и тестовых сценариях.
#
# Версия: 1.1.0
# Дата последнего изменения: 2026-01-23
# Автор: Норицин Иван, Дворников Андрей
# Версия Python Kernel: 3.12.9
#
# Изменения v1.1.0:
# - Глобальный рефакторинг: внедрены паттерны «Фабрика» и «Стратегия».
# - Реализована гибридная архитектура управления: добавлен SimpleGenerator
#   (Phase 1) и PacketManager (Phase 2/3) с поддержкой QoS.
# - Логика генерации и маршрутизации пакетов полностью перенесена из UE и BS
#   на уровень SimulationManager (принцип централизации управления).
# - Добавлен метод reset_ue для управления жизненным циклом контекста трафика
#   (задел под Handover и Detach сценарии).
#
#------------------------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np


class TrafficType(Enum):
    """
    Типы трафика согласно 3GPP.

    Каждый тип имеет соответствующий QCI (QoS Class Identifier).
    """

    VOIP = "voip"  # Голосовые звонки
    CONV_VIDEO = "conv_video"  # Видеозвонки
    REAL_TIME_GAMING = "real_time_gaming"  # Онлайн игры
    NON_CONV_VIDEO = "non_conv_video"  # Потоковое видео
    IMS = "ims"  # IMS сервисы
    VIDEO_TCP = "video_tcp"  # Видео через TCP
    VOICE_VIDEO_GAMING = "voice_video_gaming"  # Голос + видео + игры
    WEB_SERVICES = "web_services"  # Веб-сервисы
    DEFAULT = "default"  # Стандартный трафик

    def get_qci(self) -> int:
        """
        Получить QCI для типа трафика.

        Returns:
            int: QCI согласно 3GPP TS 23.203
        """
        qci_mapping = {
            TrafficType.VOIP: 100,
            TrafficType.CONV_VIDEO: 150,
            TrafficType.REAL_TIME_GAMING: 50,
            TrafficType.NON_CONV_VIDEO: 300,
            TrafficType.IMS: 100,
            TrafficType.VIDEO_TCP: 300,
            TrafficType.VOICE_VIDEO_GAMING: 100,
            TrafficType.WEB_SERVICES: 300,
            TrafficType.DEFAULT: 300,
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
    bearer_id: Optional[int] = None

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
            "bearer_id": self.bearer_id,
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
            bearer_id=data.get("bearer_id"),
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


class PoissonModel(ITrafficModel):
    """
    Пуассоновская модель трафика (stateless).

    Генерирует пакеты с экспоненциально распределёнными интервалами.
    Модель не имеет внутреннего состояния - каждый вызов независим.

    Паттерн: Strategy (конкретная реализация)
    """

    def __init__(self, packet_rate: float, min_packet_size: int = 150, max_packet_size: int = 1500):
        """
        Args:
            packet_rate: Интенсивность трафика (пакетов/сек)
            min_packet_size: Минимальный размер пакета (байты)
            max_packet_size: Максимальный размер пакета (байты)
        """
        super().__init__(min_packet_size, max_packet_size)
        self.packet_rate = packet_rate

    def generate_traffic(self, ue_id: int, current_time: int, update_interval: int) -> List[Packet]:
        """
        Генерация пуассоновского трафика.
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
            packet = Packet(
                size=packet_size,
                ue_id=ue_id,
                creation_time=generate_time,
                qci=9,  # default для Poisson
                priority=0,
            )
            packets.append(packet)

        return packets

    def get_model_name(self) -> str:
        return "Poisson"

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info["packet_rate"] = self.packet_rate
        return info


class OnOffModel(ITrafficModel):
    """
    ON/OFF модель трафика (stateful).

    Устройство чередует активные (ON) и неактивные (OFF) фазы.
    Хранит состояние для каждого UE отдельно.

    Паттерн: Strategy (конкретная реализация)
    """

    def __init__(
        self,
        duration_on: float,
        duration_off: float,
        packet_rate: float,
        min_packet_size: int = 150,
        max_packet_size: int = 1500,
    ):
        """
        Args:
            duration_on: Средняя длительность ON фазы (секунды)
            duration_off: Средняя длительность OFF фазы (секунды)
            packet_rate: Интенсивность в ON фазе (пакетов/сек)
        """
        super().__init__(min_packet_size, max_packet_size)
        self.duration_on = duration_on
        self.duration_off = duration_off
        self.packet_rate = packet_rate
        self._device_states: Dict[int, Dict] = {}

    def generate_traffic(self, ue_id: int, current_time: int, update_interval: int) -> List[Packet]:
        """Генерация ON/OFF трафика"""

        # Инициализация состояния для нового UE
        if ue_id not in self._device_states:
            self._initialize_state(ue_id, current_time, update_interval)

        state_data = self._device_states[ue_id]
        packets = []
        t = current_time - update_interval

        while t < current_time:
            if state_data["state"] == "ON":
                end_generate = min(state_data["end_state_time"], current_time)
                mean_interval_ms = 1000.0 / self.packet_rate

                while t < end_generate:
                    interval = np.random.exponential(mean_interval_ms)
                    t += interval
                    if t > end_generate:
                        break

                    packet_size = np.random.randint(self.min_packet_size, self.max_packet_size)
                    packet = Packet(
                        size=packet_size, ue_id=ue_id, creation_time=t, qci=9, priority=0
                    )
                    packets.append(packet)

                if state_data["end_state_time"] <= current_time:
                    self._switch_to_off(ue_id)

            elif state_data["state"] == "OFF":
                t = min(state_data["end_state_time"], current_time)

                if state_data["end_state_time"] <= current_time:
                    self._switch_to_on(ue_id)

            state_data = self._device_states[ue_id]

        return packets

    def _initialize_state(self, ue_id: int, current_time: int, update_interval: int):
        """Инициализация состояния для нового UE"""
        initial_state = "ON" if np.random.rand() > 0.5 else "OFF"

        if initial_state == "ON":
            duration = np.random.exponential(self.duration_on) * 1000
        else:
            duration = np.random.exponential(self.duration_off) * 1000

        self._device_states[ue_id] = {
            "state": initial_state,
            "end_state_time": current_time - update_interval + duration,
        }

    def _switch_to_off(self, ue_id: int):
        """Переключение в OFF состояние"""
        state = self._device_states[ue_id]
        duration = np.random.exponential(self.duration_off) * 1000
        state["state"] = "OFF"
        state["end_state_time"] = state["end_state_time"] + duration

    def _switch_to_on(self, ue_id: int):
        """Переключение в ON состояние"""
        state = self._device_states[ue_id]
        duration = np.random.exponential(self.duration_on) * 1000
        state["state"] = "ON"
        state["end_state_time"] = state["end_state_time"] + duration

    def clear_state(self, ue_id: int):
        """
        Предотвращает утечку памяти!

        Args:
            ue_id: ID пользователя для очистки
        """
        if ue_id in self._device_states:
            del self._device_states[ue_id]

    def get_model_name(self) -> str:
        return "OnOff"

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update(
            {
                "duration_on": self.duration_on,
                "duration_off": self.duration_off,
                "packet_rate": self.packet_rate,
                "active_devices": len(self._device_states),  # Сколько UE в памяти
            }
        )
        return info


class MMPPModel(ITrafficModel):
    """
    Модель трафика с марковским модулированным пуассоновским процессом (MMPP).
    Модель описывает систему, которая может находиться в нескольких состояниях,
    каждое из которых характеризуется своей интенсивностью генерации пакетов.
    Переходы между состояниями происходят согласно марковскому процессу.
    """

    def __init__(
        self,
        packet_rates: List[float],
        transition_matrix: np.ndarray,
        min_packet_size: float = 150,
        max_packet_size: float = 1500,
    ):
        """
        Инициализация MMPP модели трафика.

        Args:
            packet_rates: Список интенсивностей трафика для каждого состояния (пакетов/сек)
            min_packet_size: Минимальный размер пакета (по умолчанию 150 байт)
            max_packet_size: Максимальный размер пакета (по умолчанию 1500 байт)
        """
        super().__init__(min_packet_size, max_packet_size)
        self.transition_matrix = transition_matrix

        self.packet_rates = packet_rates
        self._device_states: Dict[int, Dict] = {}

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
        next_state = np.random.choice(len(self.packet_rates), p=probabilities)

        return next_state, time_to_transition

    def generate_traffic(self, ue_id: int, current_time: int, update_interval: int) -> List[Packet]:
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
        if ue_id not in self._device_states:
            initial_state = np.random.randint(0, len(self.packet_rates))
            next_state, time_to_transition = self._get_next_state(initial_state)

            self._device_states[ue_id] = {
                "current_state": initial_state,
                "transition_time": current_time - update_interval + time_to_transition,
                "next_state": next_state,
            }

        state_data = self._device_states[ue_id]
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
                    packet = Packet(
                        size=packet_size, ue_id=ue_id, creation_time=t, qci=9, priority=0
                    )
                    packets.append(packet)
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

    def clear_state(self, ue_id: int):
        """
        Предотвращает утечку памяти!

        Args:
            ue_id: ID пользователя для очистки
        """
        if ue_id in self._device_states:
            del self._device_states[ue_id]

    def get_model_name(self) -> str:
        return "MMPP"

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info.update(
            {
                "packet_rates": self.packet_rates,
                "num_states": len(self.packet_rates),
                "transition_matrix": self.transition_matrix.tolist(),  # для JSON-сериализации
                "active_devices": len(self._device_states),
            }
        )
        return info


class TrafficModelFactory:
    """
    Фабрика для создания моделей трафика.

    Паттерн: Factory
    """

    # TODO: Добавить валидацию параметров моделей

    @staticmethod
    def create_model(model_type: str, **kwargs) -> ITrafficModel:
        """
        Создать модель трафика по типу.

        Args:
            model_type (str): Тип создаваемой модели.
                - 'Poisson': Пакеты генерируются с постоянной интенсивностью.
                - 'OnOff': Модель с чередованием фаз активности (ON) и молчания (OFF).
                - 'MMPP': Марковская модель (2-state Markov Modulated Poisson Process).

            **kwargs: Параметры, специфичные для каждой модели:
                Для 'Poisson':
                    - packet_rate (float): Интенсивность трафика (пакетов/сек).
                    - min_packet_size (int): Минимальный размер пакета (байт).
                    - max_packet_size (int): Максимальный размер пакета (байт).

                Для 'OnOff':
                    - packet_rate (float): Интенсивность в активной фазе (пакетов/сек).
                    - duration_on (float): Средняя длительность фазы ON (мс).
                    - duration_off (float): Средняя длительность фазы OFF (мс).
                    - min_packet_size (int): Минимальный размер пакета (байт).
                    - max_packet_size (int): Максимальный размер пакета (байт).

                Для 'MMPP':
                    - lambda_1 (float): Интенсивность в состоянии 1 (пакетов/сек).
                    - lambda_2 (float): Интенсивность в состоянии 2 (пакетов/сек).
                    - q12 (float): Скорость перехода из состояния 1 в 2.
                    - q21 (float): Скорость перехода из состояния 2 в 1.
                    - min_packet_size (int): Минимальный размер пакета (байт).
                    - max_packet_size (int): Максимальный размер пакета (байт).

        Returns:
            ITrafficModel: Экземпляр созданной модели трафика.

        Raises:
            ValueError: Если передан неизвестный тип model_type.

        Example:
            >>> # Создание Poisson модели
            >>> model = TrafficModelFactory.create_model(
            ...     'Poisson',
            ...     packet_rate=100,
            ...     min_packet_size=150,
            ...     max_packet_size=1500
            ... )
        """
        if model_type == "Poisson":
            return PoissonModel(**kwargs)
        elif model_type == "OnOff":
            return OnOffModel(**kwargs)
        elif model_type == "MMPP":
            if "transition_matrix" not in kwargs:
                kwargs["transition_matrix"] = np.array(
                    [[0, 0.07, 0.03], [0.12, 0, 0.08], [0.4, 0.1, 0]]
                )
            return MMPPModel(**kwargs)
        else:
            available = TrafficModelFactory.get_available_models()
            raise ValueError(f"Unknown model type: '{model_type}'. Available: {available}")

    @staticmethod
    def get_available_models() -> List[str]:
        """Список доступных моделей"""
        return ["Poisson", "OnOff", "MMPP"]

    @staticmethod
    def create_poisson_model(packet_rate: float, **kwargs) -> PoissonModel:
        """Удобный метод для создания Poisson модели"""
        return PoissonModel(packet_rate=packet_rate, **kwargs)

    @staticmethod
    def create_onoff_model(
        duration_on: float, duration_off: float, packet_rate: float, **kwargs
    ) -> OnOffModel:
        """Удобный метод для создания OnOff модели"""
        return OnOffModel(
            duration_on=duration_on, duration_off=duration_off, packet_rate=packet_rate, **kwargs
        )

    @staticmethod
    def create_mmpp_model(
        packet_rates: List[float], transition_matrix: Optional[np.ndarray] = None, **kwargs
    ) -> MMPPModel:
        """Удобный метод для создания MMPP модели"""
        return MMPPModel(packet_rates=packet_rates, transition_matrix=transition_matrix, **kwargs)


class ITrafficGeneratorInterface(ABC):
    """
    Внешний интерфейс для модулей SIMULATOR, BS, UE.

    Паттерн: Facade
    """

    @abstractmethod
    def generate_packets(self, ue_id: int, current_time: int, update_interval: int) -> List[Packet]:
        """Генерация пакетов для UE"""
        pass

    @abstractmethod
    def set_model(self, ue_id: int, model_type: str, **params):
        """Установить модель трафика для UE"""
        pass

    @abstractmethod
    def get_statistics(self, ue_id: Optional[int] = None) -> Dict:
        """Статистика генерации"""
        pass

    @abstractmethod
    def reset_ue(self, ue_id: int):
        """Сброс состояния UE"""
        pass


class SimpleGenerator(ITrafficGeneratorInterface):
    """
    Простой генератор трафика для legacy поддержки.

    Особенности:
    - Один UE = одна модель
    - Прямое возвращение пакетов (без callback)
    - Поддержка default/random QCI

    Паттерн: Facade
    """

    def __init__(self, default_qci: int = 9, assign_random_qci: bool = False):
        """
        Args:
            default_qci: QCI по умолчанию для всех пакетов
            assign_random_qci: Если True, назначать случайный QCI
        """
        self.models: Dict[int, ITrafficModel] = {}
        self.default_qci = default_qci
        self.assign_random_qci = assign_random_qci

        # Статистика
        self._total_packets_generated = 0
        self._packets_per_ue: Dict[int, int] = {}

    def generate_packets(self, ue_id: int, current_time: int, update_interval: int) -> List[Packet]:
        """Генерация пакетов для одного UE"""
        if ue_id not in self.models:
            return []

        model = self.models[ue_id]
        packets = model.generate_traffic(ue_id, current_time, update_interval)

        # Установка QCI
        for pkt in packets:
            if self.assign_random_qci:
                import random

                pkt.qci = random.choice([1, 2, 3, 5, 7, 9])
            else:
                pkt.qci = self.default_qci

        # Статистика
        self._total_packets_generated += len(packets)
        self._packets_per_ue[ue_id] = self._packets_per_ue.get(ue_id, 0) + len(packets)

        return packets

    def set_model(self, ue_id: int, model_type: str, **params):
        """Установить модель через Factory"""
        model = TrafficModelFactory.create_model(model_type, **params)
        self.models[ue_id] = model

    def get_statistics(self, ue_id: Optional[int] = None) -> Dict:
        """Статистика генерации"""
        if ue_id is None:
            return {
                "total_packets": self._total_packets_generated,
                "active_ues": len(self.models),
                "per_ue": self._packets_per_ue.copy(),
            }
        else:
            return {
                "ue_id": ue_id,
                "packets_generated": self._packets_per_ue.get(ue_id, 0),
                "model": self.models[ue_id].get_model_name() if ue_id in self.models else None,
            }

    def reset_ue(self, ue_id: int):
        """
        Полный сброс контекста генерации трафика для указанного UE.

        TODO: Внедрить вызов этого метода при реализации:
              1. передача UE на другую БС.
              2. удаление UE из симуляции или выход за пределы покрытия.
              3. Обрыв связи.

        Что именно сбрасывается:
        - Удаляется привязанная модель трафика (TrafficModel).
        - Очищаются внутренние состояния stateful-моделей (например, текущая фаза
          в MMPP или время воспроизведения в VideoModel).
        - Предотвращаются утечки памяти при пересоздании UE с тем же ID.

        Args:
            ue_id (int): ID пользователя.
        """
        if ue_id in self.models:
            model = self.models[ue_id]

            # ✅ Автоматическая очистка состояния (если stateful)
            if hasattr(model, "clear_state"):
                model.clear_state(ue_id)

            del self.models[ue_id]

        # Очистка статистики
        if ue_id in self._packets_per_ue:
            del self._packets_per_ue[ue_id]


@dataclass
class Bearer:
    """
    Один поток трафика (bearer) внутри UE.

    В LTE один UE может иметь несколько bearers для разных сервисов:
    - Default bearer (всегда есть)
    - Dedicated bearers (для GBR трафика: VOIP, VIDEO)

    Attributes:
        bearer_id: Уникальный ID bearer внутри UE
        model: Модель генерации трафика
        qci: QoS Class Identifier (1-9)
        traffic_type: Тип трафика
        max_bitrate: Максимальный bitrate (Mbps), None = unlimited
        weight: Вес для приоритизации (0.0-1.0)
        enabled: Активен ли bearer
    """

    bearer_id: int
    model: ITrafficModel
    qci: int
    traffic_type: TrafficType
    max_bitrate: Optional[float] = None
    weight: float = 1.0
    enabled: bool = True

    def __post_init__(self):
        """Валидация"""
        if not 1 <= self.qci <= 9:
            raise ValueError(f"QCI must be 1-9, got {self.qci}")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be 0.0-1.0, got {self.weight}")
        if self.max_bitrate is not None and self.max_bitrate <= 0:
            raise ValueError(f"max_bitrate must be > 0, got {self.max_bitrate}")

    def get_info(self) -> Dict:
        """Информация о bearer"""
        return {
            "bearer_id": self.bearer_id,
            "model": self.model.get_model_name(),
            "qci": self.qci,
            "traffic_type": self.traffic_type.value,
            "max_bitrate": self.max_bitrate,
            "weight": self.weight,
            "enabled": self.enabled,
        }


class UeTrafficProfile:
    """
    Профиль трафика для одного UE (Composite pattern).

    Управляет всеми bearers одного пользователя.
    Один UE может иметь несколько потоков трафика одновременно.

    Паттерн: Composite (композиция моделей)
    """

    def __init__(self, ue_id: int):
        """
        Args:
            ue_id: ID пользователя
        """
        self.ue_id = ue_id
        self.bearers: Dict[int, Bearer] = {}
        self._next_bearer_id = 1  # Auto-increment для bearer_id

    def add_bearer(
        self,
        model: ITrafficModel,
        qci: int,
        traffic_type: TrafficType,
        max_bitrate: Optional[float] = None,
        weight: float = 1.0,
        bearer_id: Optional[int] = None,
    ) -> int:
        """
        Добавить bearer.

        Args:
            model: Модель генерации трафика
            qci: QoS Class Identifier
            traffic_type: Тип трафика
            max_bitrate: Максимальный bitrate (Mbps)
            weight: Вес для приоритизации
            bearer_id: ID bearer (если None - auto-increment)

        Returns:
            int: ID созданного bearer
        """
        if bearer_id is None:
            bearer_id = self._next_bearer_id
            self._next_bearer_id += 1

        if bearer_id in self.bearers:
            raise ValueError(f"Bearer {bearer_id} already exists for UE {self.ue_id}")

        bearer = Bearer(
            bearer_id=bearer_id,
            model=model,
            qci=qci,
            traffic_type=traffic_type,
            max_bitrate=max_bitrate,
            weight=weight,
        )

        self.bearers[bearer_id] = bearer
        return bearer_id

    def remove_bearer(self, bearer_id: int):
        """Удалить bearer"""
        if bearer_id not in self.bearers:
            raise ValueError(f"Bearer {bearer_id} not found for UE {self.ue_id}")

        bearer = self.bearers[bearer_id]

        # Очистка состояния stateful моделей
        if hasattr(bearer.model, "clear_state"):
            bearer.model.clear_state(self.ue_id)

        del self.bearers[bearer_id]

    def generate_all_traffic(self, current_time: int, update_interval: int) -> List[Packet]:
        """
        Генерация трафика со ВСЕХ активных bearers.

        Args:
            current_time: Текущее время (мс)
            update_interval: Интервал генерации (мс)

        Returns:
            List[Packet]: Пакеты со всех bearers (с проставленным bearer_id)
        """
        all_packets = []

        for bearer_id, bearer in self.bearers.items():
            if not bearer.enabled:
                continue

            # Генерация для этого bearer
            packets = bearer.model.generate_traffic(
                ue_id=self.ue_id, current_time=current_time, update_interval=update_interval
            )

            # Проставляем параметры bearer
            for pkt in packets:
                pkt.bearer_id = bearer_id
                pkt.qci = bearer.qci
                pkt.traffic_type = bearer.traffic_type

                # Обновляем deadline на основе traffic_type
                if pkt.deadline is None and bearer.traffic_type:
                    delay_budget = bearer.traffic_type.get_delay_budget()
                    pkt.deadline = pkt.creation_time + delay_budget

            all_packets.extend(packets)

        return all_packets

    def get_active_bearers(self) -> List[Bearer]:
        """Список активных bearers"""
        return [b for b in self.bearers.values() if b.enabled]

    def set_bearer_enabled(self, bearer_id: int, enabled: bool):
        """Включить/выключить bearer"""
        if bearer_id not in self.bearers:
            raise ValueError(f"Bearer {bearer_id} not found")
        self.bearers[bearer_id].enabled = enabled

    def get_total_bitrate(self, window_ms: int = 1000) -> float:
        """
        Расчёт суммарного bitrate всех bearers.

        Примерный расчёт на основе packet_rate моделей.
        Для точного расчёта нужна статистика.

        Args:
            window_ms: Окно времени для расчёта (мс)

        Returns:
            float: Bitrate (bps)
        """
        # Упрощённая оценка
        total_bitrate = 0.0

        for bearer in self.get_active_bearers():
            model = bearer.model
            # Если модель имеет packet_rate
            if hasattr(model, "packet_rate"):
                avg_packet_size = (model.min_packet_size + model.max_packet_size) / 2
                bearer_bitrate = model.packet_rate * avg_packet_size * 8  # bps
                total_bitrate += bearer_bitrate

        return total_bitrate

    def get_profile_info(self) -> Dict:
        """Информация о профиле"""
        return {
            "ue_id": self.ue_id,
            "num_bearers": len(self.bearers),
            "active_bearers": len(self.get_active_bearers()),
            "bearers": {bid: b.get_info() for bid, b in self.bearers.items()},
        }

    def clear_all(self):
        """Очистка всех bearers"""
        for bearer_id in list(self.bearers.keys()):
            self.remove_bearer(bearer_id)


class BitrateController:
    """
    Контроллер ограничения bitrate для UE.

    Использует sliding window для подсчёта текущего bitrate.
    Если превышен лимит - дропает пакеты.
    """

    def __init__(self, window_ms: int = 1000):
        """
        Args:
            window_ms: Размер окна для подсчёта bitrate (мс)
        """
        self.window_ms = window_ms

        # История пакетов: (timestamp, size_bytes)
        self._packet_history: Dict[int, Deque[Tuple[float, int]]] = {}

        # Лимиты bitrate: ue_id → max_bitrate_bps
        self._limits: Dict[int, float] = {}

        # Статистика dropped packets
        self._dropped_count: Dict[int, int] = {}
        self._dropped_bytes: Dict[int, int] = {}

    def set_limit(self, ue_id: int, max_bitrate_mbps: float):
        """
        Установить лимит bitrate для UE.

        Args:
            ue_id: ID пользователя
            max_bitrate_mbps: Максимальный bitrate (Mbps)
        """
        if max_bitrate_mbps <= 0:
            raise ValueError(f"max_bitrate_mbps must be > 0, got {max_bitrate_mbps}")

        self._limits[ue_id] = max_bitrate_mbps * 1e6  # Mbps → bps

        # Инициализация структур
        if ue_id not in self._packet_history:
            self._packet_history[ue_id] = deque()
        if ue_id not in self._dropped_count:
            self._dropped_count[ue_id] = 0
            self._dropped_bytes[ue_id] = 0

    def remove_limit(self, ue_id: int):
        """Убрать лимит для UE"""
        if ue_id in self._limits:
            del self._limits[ue_id]

    def check_and_throttle(
        self, ue_id: int, packets: List[Packet], current_time: float
    ) -> List[Packet]:
        """
        Проверить и отфильтровать пакеты согласно лимиту.

        Args:
            ue_id: ID пользователя
            packets: Пакеты для проверки
            current_time: Текущее время (мс)

        Returns:
            List[Packet]: Пакеты после throttling (может быть меньше)
        """
        # Если нет лимита - пропускаем всё
        if ue_id not in self._limits:
            return packets

        if ue_id not in self._packet_history:
            self._packet_history[ue_id] = deque()

        # Очистка старых пакетов из окна
        self._clean_old_packets(ue_id, current_time)

        # Текущий bitrate
        current_bitrate = self._get_current_bitrate(ue_id, current_time)
        limit_bps = self._limits[ue_id]

        # Если текущий bitrate уже превышен - дропаем ВСЕ новые пакеты
        if current_bitrate >= limit_bps:
            self._dropped_count[ue_id] += len(packets)
            self._dropped_bytes[ue_id] += sum(p.size for p in packets)
            return []

        # Пропускаем пакеты пока не превысим лимит
        accepted = []
        total_new_bits = 0

        for pkt in packets:
            pkt_bits = pkt.size * 8

            # Проверяем не превысим ли лимит
            if current_bitrate + total_new_bits + pkt_bits <= limit_bps:
                accepted.append(pkt)
                total_new_bits += pkt_bits

                # Добавляем в историю
                self._packet_history[ue_id].append((pkt.creation_time, pkt.size))
            else:
                # Лимит превышен - дропаем
                self._dropped_count[ue_id] += 1
                self._dropped_bytes[ue_id] += pkt.size

        return accepted

    def _clean_old_packets(self, ue_id: int, current_time: float):
        """Удалить пакеты старше окна"""
        window_start = current_time - self.window_ms
        history = self._packet_history[ue_id]

        while history and history[0][0] < window_start:
            history.popleft()

    def _get_current_bitrate(self, ue_id: int, current_time: float) -> float:
        """
        Подсчитать текущий bitrate (bps) за окно.

        Returns:
            float: Bitrate (bps)
        """
        if ue_id not in self._packet_history:
            return 0.0

        self._clean_old_packets(ue_id, current_time)

        history = self._packet_history[ue_id]
        if not history:
            return 0.0

        # Суммируем байты за окно
        total_bytes = sum(size for _, size in history)
        total_bits = total_bytes * 8

        # Bitrate = bits / (window_ms / 1000)
        bitrate_bps = total_bits / (self.window_ms / 1000.0)

        return bitrate_bps

    def get_current_bitrate(self, ue_id: int, current_time: float) -> float:
        """
        Публичный метод для получения текущего bitrate.

        Returns:
            float: Bitrate (bps)
        """
        return self._get_current_bitrate(ue_id, current_time)

    def get_current_bitrate_mbps(self, ue_id: int, current_time: float) -> float:
        """
        Текущий bitrate в Mbps.

        Returns:
            float: Bitrate (Mbps)
        """
        return self._get_current_bitrate(ue_id, current_time) / 1e6

    def get_statistics(self, ue_id: int) -> Dict:
        """Статистика throttling для UE"""
        return {
            "ue_id": ue_id,
            "limit_mbps": self._limits.get(ue_id, None) / 1e6 if ue_id in self._limits else None,
            "dropped_packets": self._dropped_count.get(ue_id, 0),
            "dropped_bytes": self._dropped_bytes.get(ue_id, 0),
            "dropped_mbits": self._dropped_bytes.get(ue_id, 0) * 8 / 1e6,
        }

    def reset(self, ue_id: int):
        """Сброс статистики и истории для UE"""
        if ue_id in self._packet_history:
            self._packet_history[ue_id].clear()
        if ue_id in self._dropped_count:
            self._dropped_count[ue_id] = 0
            self._dropped_bytes[ue_id] = 0


class TrafficStatistics:
    """
    Сбор детальной статистики по генерации трафика.

    Поддерживает метрики:
    - По UE (общий bitrate, packets/sec, dropped)
    - По QCI (bitrate per QCI, latency distribution)
    - По bearer (индивидуальная статистика)
    - Глобальные (total bitrate, total packets)
    """

    def __init__(self, window_ms: int = 1000):
        """
        Args:
            window_ms: Окно для расчёта bitrate (мс)
        """
        self.window_ms = window_ms
        self.start_time = None

        # Per-UE статистика
        self._ue_packets: Dict[int, int] = defaultdict(int)
        self._ue_bytes: Dict[int, int] = defaultdict(int)
        self._ue_first_time: Dict[int, float] = {}
        self._ue_last_time: Dict[int, float] = {}

        # Per-QCI статистика
        self._qci_packets: Dict[int, int] = defaultdict(int)
        self._qci_bytes: Dict[int, int] = defaultdict(int)

        # Per-bearer статистика
        self._bearer_packets: Dict[tuple, int] = defaultdict(int)  # (ue_id, bearer_id) → count
        self._bearer_bytes: Dict[tuple, int] = defaultdict(int)

        # Глобальная
        self._total_packets = 0
        self._total_bytes = 0

    def update(self, packets: List[Packet]):
        """
        Обновить статистику на основе сгенерированных пакетов.

        Args:
            packets: Список пакетов для учёта
        """
        if not packets:
            return

        if self.start_time is None:
            self.start_time = min(p.creation_time for p in packets)

        for pkt in packets:
            ue_id = pkt.ue_id
            qci = pkt.qci
            bearer_id = pkt.bearer_id
            size = pkt.size

            # Per-UE
            self._ue_packets[ue_id] += 1
            self._ue_bytes[ue_id] += size

            if ue_id not in self._ue_first_time:
                self._ue_first_time[ue_id] = pkt.creation_time
            self._ue_last_time[ue_id] = pkt.creation_time

            # Per-QCI
            self._qci_packets[qci] += 1
            self._qci_bytes[qci] += size

            # Per-bearer
            if bearer_id is not None:
                key = (ue_id, bearer_id)
                self._bearer_packets[key] += 1
                self._bearer_bytes[key] += size

            # Global
            self._total_packets += 1
            self._total_bytes += size

    def get_ue_stats(self, ue_id: int) -> Dict:
        """
        Статистика по конкретному UE.

        Returns:
            Dict: {
                'packets': int,
                'bytes': int,
                'avg_packet_size': float,
                'bitrate_bps': float,
                'bitrate_mbps': float,
                'duration_ms': float
            }
        """
        packets = self._ue_packets.get(ue_id, 0)
        total_bytes = self._ue_bytes.get(ue_id, 0)

        if packets == 0:
            return {
                "packets": 0,
                "bytes": 0,
                "avg_packet_size": 0.0,
                "bitrate_bps": 0.0,
                "bitrate_mbps": 0.0,
                "duration_ms": 0.0,
            }

        # Длительность
        first_time = self._ue_first_time[ue_id]
        last_time = self._ue_last_time[ue_id]
        duration_ms = last_time - first_time

        # Bitrate
        if duration_ms > 0:
            bitrate_bps = (total_bytes * 8) / (duration_ms / 1000.0)
        else:
            bitrate_bps = 0.0

        return {
            "packets": packets,
            "bytes": total_bytes,
            "avg_packet_size": total_bytes / packets,
            "bitrate_bps": bitrate_bps,
            "bitrate_mbps": bitrate_bps / 1e6,
            "duration_ms": duration_ms,
        }

    def get_qci_stats(self, qci: int) -> Dict:
        """Статистика по конкретному QCI"""
        packets = self._qci_packets.get(qci, 0)
        total_bytes = self._qci_bytes.get(qci, 0)

        return {
            "qci": qci,
            "packets": packets,
            "bytes": total_bytes,
            "avg_packet_size": total_bytes / packets if packets > 0 else 0.0,
        }

    def get_bearer_stats(self, ue_id: int, bearer_id: int) -> Dict:
        """Статистика по конкретному bearer"""
        key = (ue_id, bearer_id)
        packets = self._bearer_packets.get(key, 0)
        total_bytes = self._bearer_bytes.get(key, 0)

        return {
            "ue_id": ue_id,
            "bearer_id": bearer_id,
            "packets": packets,
            "bytes": total_bytes,
            "avg_packet_size": total_bytes / packets if packets > 0 else 0.0,
        }

    def get_global_stats(self) -> Dict:
        """Глобальная статистика"""
        return {
            "total_packets": self._total_packets,
            "total_bytes": self._total_bytes,
            "total_mbits": self._total_bytes * 8 / 1e6,
            "num_ues": len(self._ue_packets),
            "num_qcis": len(self._qci_packets),
            "avg_packet_size": self._total_bytes / self._total_packets
            if self._total_packets > 0
            else 0.0,
        }

    def get_qci_distribution(self) -> Dict[int, float]:
        """
        Распределение трафика по QCI (%).

        Returns:
            Dict[int, float]: qci → процент байтов
        """
        if self._total_bytes == 0:
            return {}

        return {
            qci: (bytes_val / self._total_bytes) * 100 for qci, bytes_val in self._qci_bytes.items()
        }

    def reset(self):
        """Сброс всей статистики"""
        self.start_time = None
        self._ue_packets.clear()
        self._ue_bytes.clear()
        self._ue_first_time.clear()
        self._ue_last_time.clear()
        self._qci_packets.clear()
        self._qci_bytes.clear()
        self._bearer_packets.clear()
        self._bearer_bytes.clear()
        self._total_packets = 0
        self._total_bytes = 0


class PacketManager(ITrafficGeneratorInterface):
    """
    Advanced генератор трафика с поддержкой:
    - Multi-bearer per UE
    - QoS aware traffic generation
    - Bitrate control
    - Callback система для буферов
    - Детальная статистика

    Паттерн: Facade + Observer (callback)
    """

    def __init__(
        self,
        packet_handler: Optional[Callable[[List[Packet]], None]] = None,
        enable_bitrate_control: bool = True,
        bitrate_window_ms: int = 1000,
    ):
        """
        Args:
            packet_handler: Callback для обработки пакетов (отправка в буферы)
            enable_bitrate_control: Включить контроль bitrate
            bitrate_window_ms: Окно для расчёта bitrate (мс)
        """
        # UE profiles (multi-bearer support)
        self.ue_profiles: Dict[int, UeTrafficProfile] = {}

        # Callback для пакетов
        self.packet_handler = packet_handler

        # Bitrate controller
        self.enable_bitrate_control = enable_bitrate_control
        if enable_bitrate_control:
            self.bitrate_controller = BitrateController(window_ms=bitrate_window_ms)
        else:
            self.bitrate_controller = None

        # Статистика
        self.statistics = TrafficStatistics(window_ms=bitrate_window_ms)

    def generate_packets(self, ue_id: int, current_time: int, update_interval: int) -> List[Packet]:
        """
        Генерация пакетов для UE (со всех его bearers).

        Args:
            ue_id: ID пользователя
            current_time: Текущее время (мс)
            update_interval: Интервал генерации (мс)

        Returns:
            List[Packet]: Сгенерированные пакеты (после throttling)
        """
        if ue_id not in self.ue_profiles:
            return []

        profile = self.ue_profiles[ue_id]

        # Генерация со всех bearers
        packets = profile.generate_all_traffic(current_time, update_interval)

        # Bitrate throttling (если включен)
        if self.enable_bitrate_control and self.bitrate_controller:
            packets = self.bitrate_controller.check_and_throttle(ue_id, packets, current_time)

        # Статистика
        self.statistics.update(packets)

        # Callback (отправка в буферы)
        if self.packet_handler and packets:
            self.packet_handler(packets)

        return packets

    def set_model(self, ue_id: int, model_type: str, **params):
        """
        Установить ОДНУ модель для UE (legacy support).

        Создаёт default bearer с этой моделью.
        Для multi-bearer используйте add_bearer()!

        Args:
            ue_id: ID пользователя
            model_type: Тип модели ('Poisson', 'OnOff', 'MMPP')
            **params: Параметры модели
        """
        # Создаём профиль если нет
        if ue_id not in self.ue_profiles:
            self.ue_profiles[ue_id] = UeTrafficProfile(ue_id)

        # Создаём модель через Factory
        model = TrafficModelFactory.create_model(model_type, **params)

        # Добавляем default bearer
        profile = self.ue_profiles[ue_id]
        profile.add_bearer(
            model=model,
            qci=params.get("qci", 9),
            traffic_type=params.get("traffic_type", TrafficType.BACKGROUND),
            bearer_id=0,  # Default bearer
        )

    def add_bearer(
        self,
        ue_id: int,
        model_type: str,
        qci: int,
        traffic_type: TrafficType,
        max_bitrate: Optional[float] = None,
        weight: float = 1.0,
        bearer_id: Optional[int] = None,
        **model_params,
    ) -> int:
        """
        Добавить bearer для UE.

        Args:
            ue_id: ID пользователя
            model_type: Тип модели ('Poisson', 'OnOff', 'MMPP')
            qci: QoS Class Identifier
            traffic_type: Тип трафика
            max_bitrate: Максимальный bitrate bearer (Mbps)
            weight: Вес для приоритизации
            bearer_id: ID bearer (если None - auto)
            **model_params: Параметры модели (packet_rate, ...)

        Returns:
            int: ID созданного bearer

        Example:
            >>> manager.add_bearer(
            ...     ue_id=1,
            ...     model_type='Poisson',
            ...     qci=1,
            ...     traffic_type=TrafficType.VOIP,
            ...     packet_rate=50,
            ...     max_bitrate=0.064  # 64 Kbps для VOIP
            ... )
        """
        # Создаём профиль если нет
        if ue_id not in self.ue_profiles:
            self.ue_profiles[ue_id] = UeTrafficProfile(ue_id)

        # Создаём модель
        model = TrafficModelFactory.create_model(model_type, **model_params)

        # Добавляем bearer
        profile = self.ue_profiles[ue_id]
        bid = profile.add_bearer(
            model=model,
            qci=qci,
            traffic_type=traffic_type,
            max_bitrate=max_bitrate,
            weight=weight,
            bearer_id=bearer_id,
        )

        return bid

    def remove_bearer(self, ue_id: int, bearer_id: int):
        """Удалить bearer"""
        if ue_id not in self.ue_profiles:
            raise ValueError(f"UE {ue_id} not found")

        self.ue_profiles[ue_id].remove_bearer(bearer_id)

    def set_bitrate_limit(self, ue_id: int, max_bitrate_mbps: float):
        """
        Установить лимит bitrate для UE (общий для всех bearers).

        Args:
            ue_id: ID пользователя
            max_bitrate_mbps: Максимальный bitrate (Mbps)
        """
        if not self.enable_bitrate_control or not self.bitrate_controller:
            raise RuntimeError("Bitrate control is disabled")

        self.bitrate_controller.set_limit(ue_id, max_bitrate_mbps)

    def get_statistics(self, ue_id: Optional[int] = None) -> Dict:
        """
        Получить статистику.

        Args:
            ue_id: ID пользователя (если None - глобальная статистика)

        Returns:
            Dict: Статистика
        """
        if ue_id is None:
            return self.statistics.get_global_stats()
        else:
            return self.statistics.get_ue_stats(ue_id)

    def get_qos_statistics(self, ue_id: int, qci: Optional[int] = None) -> Dict:
        """
        Статистика QoS для UE.

        Args:
            ue_id: ID пользователя
            qci: Конкретный QCI (если None - по всем QCI)

        Returns:
            Dict: QoS статистика
        """
        if qci is not None:
            return self.statistics.get_qci_stats(qci)
        else:
            # Статистика по всем QCI для этого UE
            ue_stats = self.statistics.get_ue_stats(ue_id)
            qci_dist = self.statistics.get_qci_distribution()

            return {"ue_id": ue_id, "ue_stats": ue_stats, "qci_distribution": qci_dist}

    def reset_ue(self, ue_id: int):
        """
        Полный сброс контекста генерации трафика для указанного UE.

        TODO: Внедрить вызов этого метода при реализации:
              1. передача UE на другую БС.
              2. удаление UE из симуляции или выход за пределы покрытия.
              3. Обрыв связи.

        Что именно сбрасывается:
        - Удаляется привязанная модель трафика (TrafficModel).
        - Очищаются внутренние состояния stateful-моделей (например, текущая фаза
          в MMPP или время воспроизведения в VideoModel).
        - Предотвращаются утечки памяти при пересоздании UE с тем же ID.

        Args:
            ue_id (int): ID пользователя.
        """
        if ue_id in self.ue_profiles:
            profile = self.ue_profiles[ue_id]
            profile.clear_all()
            del self.ue_profiles[ue_id]

        # Сброс bitrate controller
        if self.bitrate_controller:
            self.bitrate_controller.reset(ue_id)

    def get_bearer_info(self, ue_id: int, bearer_id: Optional[int] = None) -> Dict:
        """
        Информация о bearers UE.

        Args:
            ue_id: ID пользователя
            bearer_id: ID bearer (если None - все bearers)

        Returns:
            Dict: Информация о bearers
        """
        if ue_id not in self.ue_profiles:
            return {}

        profile = self.ue_profiles[ue_id]

        if bearer_id is not None:
            if bearer_id not in profile.bearers:
                return {}
            return profile.bearers[bearer_id].get_info()
        else:
            return profile.get_profile_info()


def test_traffic_models():
    """
    Тестирование и визуализация работы моделей трафика.
    """
    poisson_model = PoissonModel(packet_rate=5)
    # onoff_model = OnOffModel(duration_on=2, duration_off=3, packet_rate=25)
    # mmpp_model = MMPPModel(packet_rates=[5, 20, 40])

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

            # packets_onoff = onoff_model.generate_traffic(
            #     UE_ID=1, current_time=t, update_interval=update_interval
            # )

            # packets_mmpp = mmpp_model.generate_traffic(
            #     UE_ID=1, current_time=t, update_interval=update_interval
            # )

            traffic_poisson.extend(packets_poisson)
            # traffic_onoff.extend(packets_onoff)
            # traffic_mmpp.extend(packets_mmpp)

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
