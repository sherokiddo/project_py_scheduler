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
# Изменения v1.1:
#   - Исправлен фактори-паттерн.
#   - Добавлен синглтон-паттерн для установки единых границ карты для всех моделей.
#   - Исправлены модели. Теперь в UE_MODULE передаются только new_position, new_velocity и
#   new_direction. Остальные аргументы остаются внутри моделей движения.
#   - Аргументы с границами карты(x_min/max, y_min/max) теперь берутся из синглтона MapBorders.
#   - Аргументы скоростей и позиции пользователя теперь берутся из UserEquipment.
#------------------------------------------------------------------------------
"""

import inspect
from typing import Tuple

import numpy as np
from BS_MODULE import BaseStation
from UE_MODULE import UserEquipment


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
        if cls._instance is None:
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


class MobilityInterface:
    """
    Класс-фабрика для моделей мобильности
    """

    def __init__(self, ue: UserEquipment, **kwargs):
        """
        Инициализация фабрики для моделей мобильности.
        Получает границы карты и назначает объект UserEquipment.

        Args:
            ue: Объект пользователя.
            Иные аргументы для моделей.
        """
        self.x_min, self.x_max, self.y_min, self.y_max = MapBorders().get_borders()
        self.ue = ue
        self.current_velocity = np.random.uniform(self.ue.velocity_min, self.ue.velocity_max)
        self.current_direction = np.random.uniform(0, 2 * np.pi)

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
                f"Unknown model '{model}'. Valid: {', '.join(MobilityInterface.models.keys())}"
            )

        model = models[model]
        MobilityInterface.validate_params(model, kwargs)
        return model(**kwargs)

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

    def __init__(self, ue: UserEquipment, **kwargs):
        """
        Инициализация модели случайного блуждания.

        Args:
            ue: Объект пользователя.
        """
        super().__init__(ue=ue, **kwargs)
        self.is_first_move = True

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
        current_position = self.ue.position
        current_direction = self.ue.direction
        velocity_min, velocity_max = self.ue.velocity_min, self.ue.velocity_max
        is_first_move = kwargs.get("is_first_move", False)
        time_s = time_ms / 1000.0

        if is_first_move:
            is_first_move = False
        else:
            current_velocity = self.ue.velocity

        delta_x = current_velocity * np.cos(current_direction) * time_s
        delta_y = current_velocity * np.sin(current_direction) * time_s

        new_x = current_position[0] + delta_x
        new_y = current_position[1] + delta_y
        new_direction = current_direction

        if new_x < self.x_min or new_x > self.x_max:
            new_direction = np.pi - current_direction
            new_x = current_position[0] + np.cos(new_direction) * current_velocity * time_s

        if new_y < self.y_min or new_y > self.y_max:
            new_direction = -current_direction
            new_y = current_position[1] + np.sin(new_direction) * current_velocity * time_s

        if self.x_min <= new_x <= self.x_max and self.y_min <= new_y <= self.y_max:
            new_direction = np.random.uniform(0, 2 * np.pi)

        new_velocity = np.random.uniform(velocity_min, velocity_max)
        new_position = (new_x, new_y)

        return new_position, new_velocity, new_direction


class RandomWaypointModel(MobilityInterface):
    """
    Модель передвижения Random Waypoint для пользовательского устройства (UE).
    Устройство движется к случайным пунктам назначения с паузами между движениями.
    """

    def __init__(self, ue: UserEquipment, **kwargs):
        """
        Инициализация модели Random Waypoint.

        Args:
            ue: Объект пользователя.
            pause_time: Время паузы между движениями (в мс)
        """
        super().__init__(ue=ue, **kwargs)
        self.pause_time = kwargs.get("pause_time", 5.0)
        self.is_paused = False
        self.pause_timer = 0.0
        self.destination, _, _, _ = self._choose_new_destination(
            self.ue.position, self.ue.velocity_min, self.ue.velocity_max
        )

    def _choose_new_destination(
        self, current_position: Tuple[float, float], velocity_min: float, velocity_max: float
    ) -> Tuple[Tuple[float, float], float, float, bool]:
        """
        Выбирает новую точку назначения, скорость и направление для устройства.

        Args:
            current_position: Текущие координаты устройства (x, y)
            velocity_min: Минимальная скорость устройства (м/с)
            velocity_max: Максимальная скорость устройства (м/с)

        Returns:
            new_destination: Новые координаты точки назначения (x, y)
            new_velocity: Новая скорость устройства (м/с)
            new_direction: Новое направление движения (радианы)
            is_paused: Флаг, указывающий, находится ли устройство в режиме паузы
        """
        new_destination = (
            np.random.uniform(self.x_min, self.x_max),
            np.random.uniform(self.y_min, self.y_max),
        )

        new_velocity = np.random.uniform(velocity_min, velocity_max)

        delta_x = new_destination[0] - current_position[0]
        delta_y = new_destination[1] - current_position[1]
        new_direction = np.arctan2(delta_y, delta_x)

        return new_destination, new_velocity, new_direction, False

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
        current_position = self.ue.position
        velocity_min, velocity_max = self.ue.velocity_min, self.ue.velocity_max
        time_s = time_ms / 1000.0

        if self.is_paused:
            self.pause_timer += time_ms
            if self.pause_timer >= self.pause_time:
                new_destination, new_velocity, new_direction, is_paused = (
                    self._choose_new_destination(current_position, velocity_min, velocity_max)
                )
                self.destination = new_destination
                self.current_velocity = new_velocity
                self.current_direction = new_direction
                self.is_paused = False
                self.pause_timer = 0.0

                return current_position, new_velocity, new_direction
            else:
                return current_position, 0.0, self.current_direction

        delta_x = self.destination[0] - current_position[0]
        delta_y = self.destination[1] - current_position[1]
        distance = np.sqrt(delta_x**2 + delta_y**2)

        if distance <= self.current_velocity * time_s:
            new_position = self.destination
            self.is_paused = True
            self.current_velocity = 0.0
            self.current_position = new_position

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
            self.current_position = new_position

            return new_position, self.current_velocity, self.current_direction


class RandomDirectionModel(MobilityInterface):
    """
    Модель передвижения Random Direction для пользовательского устройства (UE).
    Устройство движется к границе области моделирования в случайном направлении,
    делает паузу, а затем выбирает новое направление.
    """

    def __init__(self, ue: UserEquipment, pause_time=5.0, **kwargs):
        """
        Инициализация модели Random Waypoint.

        Args:
            ue: Объект пользователя
            pause_time: Время паузы между движениями (в мс)
        """
        super().__init__(ue=ue, **kwargs)
        self.pause_time = pause_time
        self.is_paused = False
        self.pause_timer = 0.0
        self.is_first_move = True
        self.destination, self.current_velocity, self.current_direction, _, self.is_first_move = (
            self._choose_new_direction(
                self.ue.position, self.ue.velocity_min, self.ue.velocity_max, self.is_first_move
            )
        )

    def _choose_new_direction(
        self,
        current_position: Tuple[float, float],
        velocity_min: float,
        velocity_max: float,
        is_first_move: bool,
    ) -> Tuple[Tuple[float, float], float, float, bool, bool]:
        """
        Выбирает новое случайное направление и вычисляет точку на границе области моделирования.

        Args:
            current_position: Текущие координаты устройства (x, y).
            velocity_min: Минимальная скорость устройства (м/с).
            velocity_max: Максимальная скорость устройства (м/с).
            is_first_move: Флаг, указывающий, является ли это первым движением устройства.

        Returns:
            new_destination: Координаты точки на границе (x, y).
            new_velocity: Новая скорость устройства (м/с).
            new_direction: Новое направление движения (радианы).
            is_paused: Флаг, указывающий, находится ли устройство в режиме паузы.
            is_first_move: Обновленный флаг, указывающий, завершено ли первое движение.
        """
        if is_first_move:
            new_direction = np.random.uniform(0, 2 * np.pi)
            is_first_move = False
        else:
            new_direction = np.random.uniform(0, np.pi)

        new_destination = self._calculate_boundary_point(current_position, new_direction)

        # Жесточайший костыль, но что поделать, пока будет так
        while not (
            self.x_min <= new_destination[0] <= self.x_max
            and self.y_min <= new_destination[1] <= self.y_max
        ):
            new_direction = np.random.uniform(0, 2 * np.pi)
            new_destination = self._calculate_boundary_point(current_position, new_direction)

        new_velocity = np.random.uniform(velocity_min, velocity_max)

        return new_destination, new_velocity, new_direction, False, is_first_move

    def _calculate_boundary_point(
        self, current_position: Tuple[float, float], direction: float
    ) -> Tuple[float, float]:
        """
        Вычисляет точку на границе области моделирования, в которую движется устройство.

        Args:
            current_position: Текущие координаты устройства (x, y)
            direction: Направление движения в радианах

        Returns:
            Координаты точки на границе (x, y)
        """
        x, y = current_position

        distances = []
        if np.cos(direction) != 0:
            distances.append((self.x_max - x) / np.cos(direction))
            distances.append((self.x_min - x) / np.cos(direction))
        if np.sin(direction) != 0:
            distances.append((self.y_max - y) / np.sin(direction))
            distances.append((self.y_min - y) / np.sin(direction))

        positive_distances = [d for d in distances if d > 0]

        if not positive_distances:
            return x, y

        min_distance = min(positive_distances)

        boundary_x = x + min_distance * np.cos(direction)
        boundary_y = y + min_distance * np.sin(direction)

        return boundary_x, boundary_y

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
        time_s = time_ms / 1000.0
        current_position = self.ue.position

        if self.is_paused:
            self.pause_timer += time_ms
            if self.pause_timer >= self.pause_time:
                self.current_direction, _, self.is_first_move = self._choose_new_direction(
                    current_position, self.ue.velocity_min, self.ue.velocity_max, self.is_first_move
                )
                self.is_paused = False
                self.pause_timer = 0.0
            else:
                return current_position, 0.0, self.current_direction

        delta_x = self.destination[0] - current_position[0]
        delta_y = self.destination[1] - current_position[1]
        distance = np.sqrt(delta_x**2 + delta_y**2)

        if distance <= self.current_velocity * time_s:
            self.is_paused = True
            new_position = self.destination
            self.current_position = new_position
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
            self.current_position = new_position
            return new_position, self.current_velocity, self.current_direction


class GaussMarkovModel(MobilityInterface):
    """
    Модель передвижения Gauss-Markov для пользовательского устройства (UE).
    Устройство движется в соответствии с моделью Гаусса-Маркова, где скорость и направление
    изменяются на основе предыдущих значений и случайных отклонений. При приближении к границам
    области моделирования направление корректируется для предотвращения выхода за пределы.
    """

    def __init__(
        self, ue: UserEquipment, alpha: float = 0.75, boundary_threshold: float = 5.0, **kwargs
    ):
        """
        Инициализация модели Gauss-Markov.

        Args:
            ue: Объект пользователя
            alpha: Параметр памяти модели (влияет на зависимость текущих значений от предыдущих).
            boundary_threshold: Расстояние до границы, при котором начинается корректировка направления.
        """
        super().__init__(ue=ue, **kwargs)
        self.alpha = alpha
        self.boundary_threshold = boundary_threshold
        self.mean_velocity = self.current_velocity
        self.mean_direction = self.current_direction

    def update(self, time_ms: int, **kwargs) -> Tuple[Tuple[float, float], float, float]:
        """
        Обновляет позицию, скорость и направление устройства на основе модели Gauss-Markov.

        Args:
            time_ms: Время, прошедшее с последнего обновления (миллисекунды).

        Returns:
            new_position: Новые координаты устройства (x, y).
            new_velocity: Новая скорость устройства (м/с).
            new_direction: Новое направление движения (радианы).
        """
        time_s = time_ms / 1000.0
        current_position = self.ue.position
        current_velocity = getattr(self, "current_velocity", 0)
        current_direction = getattr(self, "current_direction", 0)

        mean_velocity = kwargs.get(
            "mean_velocity", getattr(self, "mean_velocity", current_velocity)
        )
        mean_direction = kwargs.get(
            "mean_direction", getattr(self, "mean_direction", current_direction)
        )

        x, y = current_position

        # При пересечении установленной "защитной" границы меняем среднее направление
        if x < self.x_min + self.boundary_threshold:
            if y < self.y_min + self.boundary_threshold:
                mean_direction = np.deg2rad(45)
            elif y > self.y_max - self.boundary_threshold:
                mean_direction = np.deg2rad(315)
            else:
                mean_direction = np.deg2rad(0)
        elif x > self.x_max - self.boundary_threshold:
            if y < self.y_min + self.boundary_threshold:
                mean_direction = np.deg2rad(135)
            elif y > self.y_max - self.boundary_threshold:
                mean_direction = np.deg2rad(225)
            else:
                mean_direction = np.deg2rad(180)
        elif y < self.y_min + self.boundary_threshold:
            mean_direction = np.deg2rad(90)
        elif y > self.y_max - self.boundary_threshold:
            mean_direction = np.deg2rad(270)

        new_velocity = (
            self.alpha * current_velocity
            + (1 - self.alpha) * mean_velocity
            + np.sqrt(1 - self.alpha**2) * np.random.normal(0, 1)
        )

        new_direction = (
            self.alpha * current_direction
            + (1 - self.alpha) * mean_direction
            + np.sqrt(1 - self.alpha**2) * np.random.normal(0, 1)
        )

        new_x = x + new_velocity * np.cos(new_direction) * time_s
        new_y = y + new_velocity * np.sin(new_direction) * time_s

        # Если всё же пользователь залез за область симуляции - делаем отскок
        if new_x < self.x_min or new_x > self.x_max:
            new_direction = np.pi - current_direction
            new_x = x + np.cos(new_direction) * current_velocity * time_s

        if new_y < self.y_min or new_y > self.y_max:
            new_direction = -current_direction
            new_y = y + np.sin(new_direction) * current_velocity * time_s

        new_position = (new_x, new_y)

        # Обновляем состояние
        self.current_velocity = new_velocity
        self.current_direction = new_direction
        self.mean_velocity = mean_velocity
        self.mean_direction = mean_direction
        self.current_position = new_position

        return new_position, new_velocity, new_direction


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
        self.current_velocity = np.random.uniform(ue.velocity_min, ue.velocity_max)
        self.current_direction = np.random.uniform(0, 2 * np.pi)
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
            self.current_position = new_position
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
            self.current_position = new_position
            return new_position, self.current_velocity, self.current_direction
