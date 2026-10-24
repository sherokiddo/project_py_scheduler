"""
#------------------------------------------------------------------------------
# Модуль: MOBILITY_MODEL - Модели мобильности пользовательских устройств (UE)
#------------------------------------------------------------------------------
# Описание:
#   Реализация различных моделей мобильности для имитации перемещения пользовательских
#   устройств в системах беспроводной связи. Включает как простые, так и сложные
#   модели движения с поддержкой граничных условий.
#
# Версия: 1.1
# Дата последнего изменения: 2025-11-05
# Автор: Норицин Иван, Дворников Андрей
# Версия Python Kernel: 3.12.9
#
# Изменения v1.0.1:
#   - Добавлен фактори-паттерн. Теперь модели вызываются через MobilityInterface.
#   - Добавлено наследование от MobilityInterface для всех моделей.
#
# Версия: 1.2
# Дата последнего изменения: 2026-08-05
# Автор: Норицин Иван, Дворников Андрей, Шаимов Богдан
# Версия Python Kernel: 3.12.9
#
# Изменения v1.1:
#   - Исправлен фактори-паттерн.
#   - Добавлен синглтон-паттерн для установки единых границ карты для всех моделей.
#   - Исправлены модели. Теперь в UE_MODULE передаются только new_position, new_velocity и
#   new_direction. Остальные аргументы остаются внутри моделей движения.
#   - Аргументы с границами карты(x_min/max, y_min/max) теперь берутся из синглтона MapBorders.
#   - Аргументы скоростей и позиции пользователя теперь берутся из UserEquipment.
#
# Изменения v1.2 (Архитектурный рефакторинг):
#   - Полностью переработана генерация случайных чисел (RNG).
#   - Инициализация генератора `RandomGenerator` в конструкторе `MobilityInterface`.
#   - Из всех моделей удален избыточный код по ручному управлению потоками `RngStream`.
#   - Внедрен механизм `stream_offset` для автоматического распределения независимых потоков
#     внутри сложных моделей.
#   - Обеспечена полная математическая независимость траекторий абонентов на базе уникального `base_idx`.
#------------------------------------------------------------------------------
"""

import inspect
from typing import Tuple
import math
import numpy as np
from BS_MODULE import BaseStation
from UE_MODULE import UserEquipment
from RNG import RandomGenerator


class MapBorders:
    """
    Синглтон для установки границ карты
    Args:
        x_min: Минимальная граница по оси X
        x_max: Максимальная граница по оси X
        y_min: Минимальная граница по оси Y
        y_max: Максимальная граница по оси Y
    """

    _instance = None

    def __new__(cls, x_min=None, x_max=None, y_min=None, y_max=None):
        """
        Создает единственный объект границ.

        Args:
            x_min: Минимальная граница по оси X
            x_max: Максимальная граница по оси X
            y_min: Минимальная граница по оси Y
            y_max: Максимальная граница по оси Y
        """
        if x_min is not None and x_max is not None and x_max <= x_min:
            raise ValueError("x_max должен быть больше x_min")
        if y_min is not None and y_max is not None and y_max <= y_min:
            raise ValueError("y_max должен быть больше y_min")
        for val in [x_min, x_max, y_min, y_max]:
            if val is not None and not isinstance(val, (int, float)):
                raise TypeError("Границы карты должны быть числами")
        if x_min is not None:
            if x_max <= x_min:
                raise ValueError("x_max должен быть больше x_min")

        if cls._instance is None:
            if any(v is None for v in (x_min, x_max, y_min, y_max)):
                raise RuntimeError(
                    "MapBorders не инициализированы. "
                    "Вызовите MapBorders(x_min, x_max, y_min, y_max) перед созданием модели."
                )
            cls._instance = super().__new__(cls)
            cls._instance.x_min = x_min
            cls._instance.x_max = x_max
            cls._instance.y_min = y_min
            cls._instance.y_max = y_max

        return cls._instance

    def get_borders(self):
        """
        Возвращает значения границ карты.

        Returns:
            x_min: Минимальная граница по оси X
            x_max: Максимальная граница по оси X
            y_min: Минимальная граница по оси Y
            y_max: Максимальная граница по оси Y
        """
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    @classmethod
    def reset(cls):
        """Сброс синглтона (для тестов и повторных запусков)"""
        cls._instance = None

class MobilityInterface:
    """
    Класс-фабрика для моделей мобильности
    """

    def __init__(self, ue: UserEquipment, **kwargs):
        """
        Инициализация фабрики для моделей мобильности.
        Получает границы карты и назначает объект UserEquipment,  инициализирует генератор случайных чисел.

        Args:
            ue: Объект пользователя.
            **kwargs: Дополнительные параметры (seed, base_idx, run и др.)
        """
        self.x_min, self.x_max, self.y_min, self.y_max = MapBorders().get_borders()
        self.ue = ue

        seed = kwargs.get('seed', 42)
        base_idx = kwargs.get('base_idx', 0)
        run = kwargs.get('run', 1)
        self.rng = RandomGenerator(seed=seed, base_stream_idx=base_idx, run_idx=run)

        self.current_velocity = 0.0
        self.current_direction = 0.0

    @staticmethod
    def validate_params(cls, kwargs):
        """
        Проверяет наличие всех обязательных параметров при создании модели.
        """
        sig = inspect.signature(cls.__init__)
        missing = []
        for name, param in sig.parameters.items():
            if name == "self" or name == "kwargs":
                continue
            if param.default == inspect.Parameter.empty and name not in kwargs:
                missing.append(name)
        if missing:
            raise ValueError(f"Отсутствуют обязательные параметры: {', '.join(missing)}")

    def update(self, time_ms: int, **kwargs) -> tuple:
        """
        Обязательный метод для всех моделей.
        """
        raise NotImplementedError("Модель обязательно должна иметь метод update()")

    def _apply_boundary_reflection(
        self, x: float, y: float
    ) -> Tuple[float, float]:
        """
        Физически корректная рефлексия для произвольного смещения.
        Работает при любом time_ms — обрабатывает многократное
        пересечение границ через зигзаг-нормализацию.
        """
        w = self.x_max - self.x_min
        h = self.y_max - self.y_min

        x_fold = (x - self.x_min) % (2 * w)
        if x_fold > w:
            x_fold = 2 * w - x_fold
        x = self.x_min + x_fold

        y_fold = (y - self.y_min) % (2 * h)
        if y_fold > h:
            y_fold = 2 * h - y_fold
        y = self.y_min + y_fold

        return x, y

    @staticmethod
    def get_models():
        """
        Возвращает словарь доступных моделей.
        """
        models = {
            "RandomWalk": RandomWalkModel,
            "RandomWaypoint": RandomWaypointModel,
            "RandomDirection": RandomDirectionModel,
            "GaussMarkov": GaussMarkovModel,
            "DiagonalWalk": DiagonalWalkModel,
        }
        return models

    @staticmethod
    def create(model: str, **kwargs):
        """
        Метод для создания фабрики.
        Выбирает модель по названию и проверяет параметры.
        Args:
            model (str): Название модели.
            Иные аргументы для модели.
        """
        models = MobilityInterface.get_models()
        if model not in models:
            raise ValueError(
                f"Unknown model '{model}'. "
                f"Valid: {', '.join(MobilityInterface.get_models().keys())}"
            )

        model = models[model]
        MobilityInterface.validate_params(model, kwargs)
        return model(**kwargs)

    @staticmethod
    def get_available_models():
        """
        Выводит на экран список названий доступных моделей.
        """
        models = list(MobilityInterface.get_models().keys())
        print("Доступные модели:", models)


class RandomWalkModel(MobilityInterface):
    """
    Модель передвижения Random Walk для пользовательского устройства (UE).
    Устройство движется в случайном направлении в пределах заданных границ.
    """

    def __init__(self, ue, **kwargs):
        """
        Инициализация модели случайного блуждания.

        Args:
            ue: Объект пользователя.
        """
        super().__init__(ue=ue, **kwargs)
        self.mode = kwargs.get("mode", "Time")
        self.mode_time = kwargs.get("mode_time", 1.0)
        self.mode_distance = kwargs.get("mode_distance", 1.0)

        self.time_until_next_change = 0.0
        self.current_vel_vector = np.array([0.0, 0.0])
        self.is_initialized = False

    def _draw_random_velocity(self, current_position=None):
        """
        Выбирает новую случайную скорость и направление движения.

        Args:
            current_position: Текущие координаты устройства (x, y) для корректной проверки границ.
        """
        if current_position is None:
            current_position = self.ue.position

        speed = self.rng.uniform(self.ue.velocity_min, self.ue.velocity_max, stream_offset=0)
        direction = self.rng.uniform(0.0, 6.283185, stream_offset=1)

        vx = np.cos(direction) * speed
        vy = np.sin(direction) * speed

        px, py = current_position
        eps = 1e-7
        if px >= self.x_max - eps:
            vx = -abs(vx)
        elif px <= self.x_min + eps:
            vx = abs(vx)
        if py >= self.y_max - eps:
            vy = -abs(vy)
        elif py <= self.y_min + eps:
            vy = abs(vy)

        self.current_vel_vector = np.array([vx, vy])
        match self.mode:
            case "Time":
                self.time_until_next_change = self.mode_time
            case "Distance":
                self.time_until_next_change = self.mode_distance / speed if speed > 0 else float('inf')
            case _:
                raise ValueError(f"Неизвестный режим RandomWalk: {self.mode}")

    def update(self, time_ms: int, **kwargs) -> Tuple[Tuple[float, float], float, float]:
        """
        Обновляет позицию, скорость и направление устройства на основе модели случайного блуждания.

        Args:
            time_ms: Время, прошедшее с последнего обновления (миллисекунды).
        Returns:
            new_position: Новые координаты устройства (x, y).
            new_velocity: Новая скорость устройства (м/с).
            new_direction: Новое направление движения (радианы).
        """
        dt = time_ms / 1000.0
        remaining_dt = dt
        current_position = np.array(self.ue.position[:2])

        if not self.is_initialized:
            self._draw_random_velocity(current_position)
            self.is_initialized = True

        while remaining_dt > 1e-11:
            t_to_change = self.time_until_next_change
            t_to_bound = float('inf')
            vx, vy = self.current_vel_vector

            if vx > 1e-12:
                t_to_bound = min(t_to_bound, (self.x_max - current_position[0]) / vx)
            elif vx < -1e-12:
                t_to_bound = min(t_to_bound, (self.x_min - current_position[0]) / vx)
            if vy > 1e-12:
                t_to_bound = min(t_to_bound, (self.y_max - current_position[1]) / vy)
            elif vy < -1e-12:
                t_to_bound = min(t_to_bound, (self.y_min - current_position[1]) / vy)

            step = min(remaining_dt, t_to_change, t_to_bound)
            current_position += self.current_vel_vector * step
            remaining_dt -= step
            self.time_until_next_change -= step

            if step == t_to_bound and t_to_bound < t_to_change:
                eps = 1e-7
                if current_position[0] >= self.x_max - eps:
                    self.current_vel_vector[0] = -abs(self.current_vel_vector[0])
                    current_position[0] = self.x_max - eps
                elif current_position[0] <= self.x_min + eps:
                    self.current_vel_vector[0] = abs(self.current_vel_vector[0])
                    current_position[0] = self.x_min + eps

                if current_position[1] >= self.y_max - eps:
                    self.current_vel_vector[1] = -abs(self.current_vel_vector[1])
                    current_position[1] = self.y_max - eps
                elif current_position[1] <= self.y_min + eps:
                    self.current_vel_vector[1] = abs(self.current_vel_vector[1])
                    current_position[1] = self.y_min + eps

            if self.time_until_next_change <= 1e-11:
                self._draw_random_velocity(current_position)

        new_position = (float(current_position[0]), float(current_position[1]))
        new_velocity = float(np.linalg.norm(self.current_vel_vector))
        new_direction = float(np.arctan2(self.current_vel_vector[1], self.current_vel_vector[0]))

        return new_position, new_velocity, new_direction


class RandomWaypointModel(MobilityInterface):
    """
    Модель передвижения Random Waypoint для пользовательского устройства (UE).
    Устройство движется к случайным пунктам назначения с паузами между движениями.
    """

    def __init__(self, ue, **kwargs):
        """
        Инициализация модели Random Waypoint.

        Args:
            ue: Объект пользователя.
            rng_stream: Поток генератора случайных чисел.
            **kwargs: Дополнительные параметры (pause_time, seed, run).
        """
        super().__init__(ue=ue, **kwargs)
        self.pause_val = kwargs.get("pause_time", 2.0)
        self.destination = np.array([0.0, 0.0])
        self.is_paused = True
        self.time_until_next_state = 0.0
        self.current_vel_vector = np.array([0.0, 0.0])
        self.is_initialized = False

    def _begin_walk(self, current_position):
        """
        Выбор новой точки назначения и скорости движения.

        Args:
            сurrent_position: Текущие координаты устройства.
        """
        dest_x = self.rng.uniform(self.x_min, self.x_max, stream_offset=2)
        dest_y = self.rng.uniform(self.y_min, self.y_max, stream_offset=3)
        self.destination = np.array([dest_x, dest_y])

        speed = self.rng.uniform(self.ue.velocity_min, self.ue.velocity_max, stream_offset=0)
        delta = self.destination - current_position
        distance = np.linalg.norm(delta)

        if distance > 1e-9:
            self.current_vel_vector = (delta / distance) * speed
            self.time_until_next_state = distance / speed
        else:
            self.current_vel_vector = np.array([0.0, 0.0])
            self.time_until_next_state = 0.0

        self.is_paused = False

    def _start_pause(self):
        """
        Устройство останавливается на заданное время.
        """
        self.current_vel_vector = np.array([0.0, 0.0])
        self.time_until_next_state = self.rng.uniform(self.pause_val, self.pause_val, stream_offset=1)
        self.is_paused = True

    def update(self, time_ms: int, **kwargs) -> Tuple[Tuple[float, float], float, float]:
        """
        Обновляет позицию, скорость, направление и состояние устройства на основе модели Random Waypoint.

        Args:
            time_ms: Время, прошедшее с последнего обновления (миллисекунды)

        Returns:
            new_position: Новые координаты устройства (x, y)
            new_velocity: Новая скорость устройства (м/с)
            new_direction: Новое направление движения (радианы)
        """
        dt = time_ms / 1000.0
        remaining_dt = dt
        current_position = np.array(self.ue.position[:2])

        if not self.is_initialized:
            self._start_pause()
            self.is_initialized = True

        while remaining_dt > 1e-11:
            step = min(remaining_dt, self.time_until_next_state)
            current_position += self.current_vel_vector * step
            remaining_dt -= step
            self.time_until_next_state -= step

            if self.time_until_next_state <= 1e-11:
                if self.is_paused:
                    self._begin_walk(current_position)
                else:
                    self._start_pause()

        new_position = (float(current_position[0]), float(current_position[1]))
        new_velocity = float(np.linalg.norm(self.current_vel_vector))
        new_direction = float(np.arctan2(self.current_vel_vector[1], self.current_vel_vector[0]))

        return new_position, new_velocity, new_direction


class RandomDirectionModel(MobilityInterface):
    """
    Модель передвижения Random Direction для пользовательского устройства (UE).
    Устройство движется к границе области моделирования в случайном направлении,
    делает паузу, а затем выбирает новое направление.
    """

    def __init__(self, ue, **kwargs):
        """
        Инициализация модели Random Waypoint.

        Args:
            ue: Объект пользователя.
            rng_stream: Поток генератора случайных чисел.
            **kwargs: Дополнительные параметры (pause_time, seed, run).
        """
        super().__init__(ue=ue, **kwargs)
        self.pause_val = kwargs.get("pause_time", 2.0)
        self.is_paused = False
        self.pause_timer = 0.0
        self.current_velocity = 0.0
        self.current_direction = 0.0
        self.destination = np.array([0.0, 0.0])
        self.is_initialized = False

    def _get_direction_in_range(self, min_val, max_val):
        """
        GetValue(min, max), который меняет границы.

        Args:
            min_val: Минимально допустимый угол направления (в радианах).
            max_val: Максимально допустимый угол направления (в радианах).

        Returns:
            Случайно сгенерированное направление в заданных пределах.
        """
        return self.rng.uniform(min_val, max_val, stream_offset=0)

    def _reset_direction_and_speed(self, current_position):
        """
        Вызывается после достижения границы для выбора нового направления отскока.

        Args:
            current_position: Текущие координаты устройства на границе.
        """
        px, py = current_position
        eps = 1e-7
        on_right = (px >= self.x_max - eps)
        on_left = (px <= self.x_min + eps)
        on_top = (py >= self.y_max - eps)
        on_bot = (py <= self.y_min + eps)

        if on_right and on_top:
            dir_val = self._get_direction_in_range(math.pi, math.pi + math.pi / 2)
        elif on_left and on_top:
            dir_val = self._get_direction_in_range(math.pi + math.pi / 2, 2 * math.pi)
        elif on_right and on_bot:
            dir_val = self._get_direction_in_range(0.0, math.pi / 2) + math.pi / 2
        elif on_left and on_bot:
            dir_val = self._get_direction_in_range(math.pi / 2, math.pi)
        elif on_right:
            dir_val = self._get_direction_in_range(math.pi / 2, math.pi + math.pi / 2)
        elif on_left:
            dir_val = self._get_direction_in_range(-math.pi / 2, math.pi - math.pi / 2)
        elif on_top:
            dir_val = self._get_direction_in_range(math.pi, 2 * math.pi)
        elif on_bot:
            dir_val = self._get_direction_in_range(0.0, math.pi)
        else:
            dir_val = self._get_direction_in_range(0.0, 2 * math.pi)

        self.current_direction = dir_val
        self.current_velocity = self.rng.uniform(self.ue.velocity_min, self.ue.velocity_max, stream_offset=1)

        vx = self.current_velocity * math.cos(self.current_direction)
        vy = self.current_velocity * math.sin(self.current_direction)
        self.current_vel_vector = np.array([vx, vy])

        t_to_bound = float('inf')
        if vx > 1e-12:
            t_to_bound = min(t_to_bound, (self.x_max - px) / vx)
        elif vx < -1e-12:
            t_to_bound = min(t_to_bound, (self.x_min - px) / vx)
        if vy > 1e-12:
            t_to_bound = min(t_to_bound, (self.y_max - py) / vy)
        elif vy < -1e-12:
            t_to_bound = min(t_to_bound, (self.y_min - py) / vy)

        self.pause_timer = max(0.0, t_to_bound)
        self.is_paused = False

    def _begin_pause(self):
        """
        Устройство останавливается у границы карты на заданное время.
        """
        self.current_velocity = 0.0
        self.current_vel_vector = np.array([0.0, 0.0])
        self.pause_timer = self.rng.uniform(self.pause_val, self.pause_val, stream_offset=2)
        self.is_paused = True

    def update(self, time_ms: int, **kwargs) -> Tuple[Tuple[float, float], float, float]:
        """
        Обновляет позицию, скорость, направление и состояние устройства на основе модели Random Direction.

        Args:
            time_ms: Время, прошедшее с последнего обновления (миллисекунды)

        Returns:
            new_position: Новые координаты устройства (x, y)
            new_velocity: Новая скорость устройства (м/с)
            new_direction: Новое направление движения (радианы)
        """
        dt = time_ms / 1000.0
        remaining_dt = dt
        current_position = np.array(self.ue.position[:2])

        if not self.is_initialized:
            self.current_direction = self._get_direction_in_range(0.0, 2 * math.pi)
            self.current_velocity = self.rng.uniform(self.ue.velocity_min, self.ue.velocity_max, stream_offset=1)
            vx = self.current_velocity * math.cos(self.current_direction)
            vy = self.current_velocity * math.sin(self.current_direction)
            self.current_vel_vector = np.array([vx, vy])

            t_to_bound = float('inf')
            if vx > 1e-12:
                t_to_bound = min(t_to_bound, (self.x_max - current_position[0]) / vx)
            elif vx < -1e-12:
                t_to_bound = min(t_to_bound, (self.x_min - current_position[0]) / vx)
            if vy > 1e-12:
                t_to_bound = min(t_to_bound, (self.y_max - current_position[1]) / vy)
            elif vy < -1e-12:
                t_to_bound = min(t_to_bound, (self.y_min - current_position[1]) / vy)

            self.pause_timer = max(0.0, t_to_bound)
            self.destination = np.array([current_position[0] + vx * t_to_bound, current_position[1] + vy * t_to_bound])
            self.is_paused = False
            self.is_initialized = True

        while remaining_dt > 1e-11:
            step = min(remaining_dt, self.pause_timer)
            current_position += self.current_vel_vector * step
            remaining_dt -= step
            self.pause_timer -= step

            if self.pause_timer <= 1e-11:
                eps = 1e-7
                if current_position[0] > self.x_max - eps:
                    current_position[0] = self.x_max
                elif current_position[0] < self.x_min + eps:
                    current_position[0] = self.x_min
                if current_position[1] > self.y_max - eps:
                    current_position[1] = self.y_max
                elif current_position[1] < self.y_min + eps:
                    current_position[1] = self.y_min

                if self.is_paused:
                    self._reset_direction_and_speed(current_position)
                else:
                    self._begin_pause()

        new_position = (float(current_position[0]), float(current_position[1]))
        return new_position, float(self.current_velocity), float(self.current_direction)


class GaussMarkovModel(MobilityInterface):
    """
    Модель передвижения Gauss-Markov для пользовательского устройства (UE).
    Устройство движется в соответствии с моделью Гаусса-Маркова, где скорость и направление
    изменяются на основе предыдущих значений и случайных отклонений.
    Эта модель является 3D-моделью и использует 6 потоков генератора.
    """

    def __init__(
        self, ue: UserEquipment, alpha: float = 0.75, boundary_threshold: float = 5.0, **kwargs
    ):
        """
        Инициализация модели Gauss-Markov.

        Args:
            ue: Объект пользователя
            alpha: Параметр памяти модели (влияет на зависимость текущих значений от предыдущих).
            **kwargs: Дополнительные параметры (alpha, mode_time, pitch_min/max, z_min/max).
        """
        super().__init__(ue=ue, **kwargs)
        self.alpha = kwargs.get('alpha', 0.8)
        self.time_step = kwargs.get("mode_time", 1.0)
        self.boundary_threshold = kwargs.get("boundary_threshold", 0.0)
        self.z_min = kwargs.get('z_min', 0.0)
        self.z_max = kwargs.get('z_max', 0.0)
        self.pitch_min = kwargs.get('pitch_min', 0.0)
        self.pitch_max = kwargs.get('pitch_max', 0.0)
        self.norm_bound = 10.0

        self.mean_velocity = 0.0
        self.mean_direction = 0.0
        self.mean_pitch = 0.0

        self.velocity = 0.0
        self.direction = 0.0
        self.pitch = 0.0

        self.current_vel_vector = np.array([0.0, 0.0, 0.0])
        self.time_until_next_change = 0.0
        self.is_initialized = False
        self._last_z = 0.0

    def _start(self):
        """
        Высчитывает новые векторы движения по цепи Маркова с применением нормального распределения.
        Отвечает за предиктивную проверку отскока от границ карты (3D).
        """
        if self.mean_velocity == 0.0:
            self.mean_velocity = self.rng.uniform(self.ue.velocity_min, self.ue.velocity_max, stream_offset=0)
            self.mean_direction = self.rng.uniform(0.0, 6.283185, stream_offset=2)
            self.mean_pitch = self.rng.uniform(self.pitch_min, self.pitch_max, stream_offset=4)

            self.velocity = self.mean_velocity
            self.direction = self.mean_direction
            self.pitch = self.mean_pitch

        rv = self.rng.normal(0.0, 1.0, self.norm_bound, stream_offset=1)
        rd = self.rng.normal(0.0, 1.0, self.norm_bound, stream_offset=3)
        rp = self.rng.normal(0.0, 1.0, self.norm_bound, stream_offset=5)

        one_minus_alpha = 1.0 - self.alpha
        sqrt_alpha = math.sqrt(1.0 - self.alpha * self.alpha)

        self.velocity = self.alpha * self.velocity + one_minus_alpha * self.mean_velocity + sqrt_alpha * rv
        self.direction = self.alpha * self.direction + one_minus_alpha * self.mean_direction + sqrt_alpha * rd
        self.pitch = self.alpha * self.pitch + one_minus_alpha * self.mean_pitch + sqrt_alpha * rp

        cosDir = math.cos(self.direction)
        cosPit = math.cos(self.pitch)
        sinDir = math.sin(self.direction)
        sinPit = math.sin(self.pitch)

        vx = self.velocity * cosDir * cosPit
        vy = self.velocity * sinDir * cosPit
        vz = self.velocity * sinPit

        next_x = self.current_pos[0] + vx * self.time_step
        next_y = self.current_pos[1] + vy * self.time_step
        next_z = self.current_pos[2] + vz * self.time_step

        if not (self.x_min <= next_x <= self.x_max and
                self.y_min <= next_y <= self.y_max and
                self.z_min <= next_z <= self.z_max):

            if next_x > self.x_max or next_x < self.x_min:
                vx = -vx
                self.mean_direction = math.pi - self.mean_direction
            if next_y > self.y_max or next_y < self.y_min:
                vy = -vy
                self.mean_direction = -self.mean_direction
            if next_z > self.z_max or next_z < self.z_min:
                vz = -vz
                self.mean_pitch = -self.mean_pitch

            self.direction = self.mean_direction
            self.pitch = self.mean_pitch

        self.current_vel_vector = np.array([vx, vy, vz])
        self.time_until_next_change = self.time_step

    def update(self, time_ms: int, **kwargs) -> Tuple[Tuple[float, float], float, float]:
        """
        Обновляет позицию, скорость и направление устройства на основе модели Gauss-Markov.

        Args:
            time_ms: Время, прошедшее с последнего обновления (миллисекунды).

        Returns:
            new_position: Новые 2D координаты устройства (x, y).
            current_velocity_2d: Новая скорость устройства (м/с).
            current_direction_2d: Новое направление движения (радианы).
        """
        dt = time_ms / 1000.0
        remaining_dt = dt

        if not self.is_initialized:
            ue_pos = self.ue.position
            z_val = ue_pos[2] if len(ue_pos) > 2 else 0.0
            self.current_pos = np.array([ue_pos[0], ue_pos[1], z_val], dtype=float)
            self._last_z = z_val
            self._start()
            self.is_initialized = True

        if isinstance(self.current_pos, (tuple, list)) or len(self.current_pos) < 3:
            z_val = self.current_pos[2] if len(self.current_pos) > 2 else getattr(self, '_last_z', 0.0)
            self.current_pos = np.array([self.current_pos[0], self.current_pos[1], z_val], dtype=float)

        if getattr(self, 'boundary_threshold', 0.0) > 0:
            x, y = self.current_pos[0], self.current_pos[1]
            if x < self.x_min + self.boundary_threshold:
                self.mean_direction = math.pi / 4 if y < self.y_min + self.boundary_threshold else (
                    7 * math.pi / 4 if y > self.y_max - self.boundary_threshold else 0.0)
            elif x > self.x_max - self.boundary_threshold:
                self.mean_direction = 3 * math.pi / 4 if y < self.y_min + self.boundary_threshold else (
                    5 * math.pi / 4 if y > self.y_max - self.boundary_threshold else math.pi)
            elif y < self.y_min + self.boundary_threshold:
                self.mean_direction = math.pi / 2
            elif y > self.y_max - self.boundary_threshold:
                self.mean_direction = 3 * math.pi / 2

        while remaining_dt > 1e-11:
            step = min(remaining_dt, self.time_until_next_change)
            self.current_pos += self.current_vel_vector * step
            remaining_dt -= step
            self.time_until_next_change -= step

            if self.time_until_next_change <= 1e-11:
                self._start()

        self._last_z = float(self.current_pos[2])

        new_position = (float(self.current_pos[0]), float(self.current_pos[1]))
        current_velocity_2d = float(np.hypot(self.current_vel_vector[0], self.current_vel_vector[1]))
        current_direction_2d = float(np.arctan2(self.current_vel_vector[1], self.current_vel_vector[0]))

        return new_position, current_velocity_2d, current_direction_2d


class DiagonalWalkModel(MobilityInterface):
    """
    Модель передвижения Diagonal Walk для пользовательского устройства (UE).
    Устройство движется к точке, противоположной его текущей позиции относительно базовой
    станции (BS).
    """

    def __init__(self, ue: UserEquipment, bs: BaseStation, pause_time: int, **kwargs):
        """
        Инициализация модели DiagonalWalk.

        Args:
            ue: Объект пользователя
            bs: Объект базовой станции
            pause_time: Время паузы между движениями (в мс)
        """

        super().__init__(ue=ue, **kwargs)
        self.bs_x = bs.position[0]
        self.bs_y = bs.position[1]
        self.pause_time = pause_time
        self.is_paused = False
        self.pause_timer = 0.0
        self._diagonal_destination = None
        self.is_first_move = True

    def _choose_new_destination(
        self,
        current_position: Tuple[float, float],
        velocity_min: float,
        velocity_max: float,
        is_first_move: bool,
    ):
        """
        Вычисление точки назначения, противоположной начальной позиции пользователя, относительно
        базовой станции.

        Args:
            current_position: Текущие координаты устройства (x, y)
            velocity_min: Минимальная скорость устройства (м/с)
            velocity_max: Максимальная скорость устройства (м/с)
            is_first_move: Флаг, указывающий, является ли это первым движением устройства

        Returns:
            new_destination: Новые координаты точки назначения (x, y)
            new_velocity: Новая скорость устройства (м/с)
            new_direction: Новое направление движения (радианы)
            is_paused: Флаг, указывающий, находится ли устройство в режиме паузы
            pause_timer: Текущее время, прошедшее в режиме паузы (мс)
            is_first_move: Обновлённый флаг, указывающий, завершено ли первое движение
        """
        if is_first_move or self._diagonal_destination is None:
            delta_x = current_position[0] - self.bs_x
            delta_y = current_position[1] - self.bs_y
            dist = np.hypot(delta_x, delta_y)

            if dist < 1e-6:
                out_angle = np.arctan2(delta_y, delta_x)
                out_distance = dist
                opposite_angle = out_angle + np.pi
                new_destination = (
                    self.bs_x + out_distance * np.cos(opposite_angle),
                    self.bs_y + out_distance * np.sin(opposite_angle),
                )
            else:
                new_destination = (
                    2 * self.bs_x - current_position[0],
                    2 * self.bs_y - current_position[1],
                )
            self._diagonal_destination = new_destination
        return self._diagonal_destination, velocity_max, 0.0, False, 0.0, 0.0

    def update(self, time_ms: int, **kwargs) -> Tuple[Tuple[float, float], float, float]:
        """
        Обновляет позицию, скорость, направление и состояние устройства на основе модели DiagonalWalk.

        Args:
            time_ms: Время, прошедшее с последнего обновления (миллисекунды)

        Returns:
            new_position: Новые координаты устройства (x, y)
            new_velocity: Новая скорость устройства (м/с)
            new_direction: Новое направление движения (радианы)
        """
        current_position = self.ue.position
        velocity_min = self.ue.velocity_min
        velocity_max = self.ue.velocity_max
        time_s = time_ms / 1000.0

        if self.is_paused:
            self.pause_timer += time_ms
            if self.pause_timer >= self.pause_time:
                self._diagonal_destination = None
                self.is_paused = False
                self.pause_timer = 0.0
            else:
                return current_position, 0.0, self.current_direction

        if self._diagonal_destination is None:
            (
                self._diagonal_destination,
                self.current_velocity,
                self.current_direction,
                self.is_paused,
                self.pause_timer,
                self.is_first_move,
            ) = self._choose_new_destination(
                current_position, velocity_min, velocity_max, self.is_first_move
            )

        delta_x = self._diagonal_destination[0] - current_position[0]
        delta_y = self._diagonal_destination[1] - current_position[1]
        distance = np.hypot(delta_x, delta_y)

        self.current_direction = np.arctan2(delta_y, delta_x)

        if distance <= self.current_velocity * time_s:
            self.is_paused = True
            new_position = self._diagonal_destination
            self.current_velocity = 0.0
            return new_position, 0.0, self.current_direction
        else:
            new_x = (
                current_position[0]
                + self.current_velocity * np.cos(self.current_direction) * time_s
            )
            new_y = (
                current_position[1]
                + self.current_velocity * np.sin(self.current_direction) * time_s
            )
            new_position = (new_x, new_y)
            return new_position, self.current_velocity, self.current_direction
