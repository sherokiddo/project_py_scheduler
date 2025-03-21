import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from UE_MODULE import UserEquipment, Buffer, UECollection

class RandomWalkModel:
    """
    Модель передвижения Random Walk для пользовательского устройства (UE).
    Устройство движется в случайном направлении в пределах заданных границ.
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """
        Инициализация модели случайного блуждания.

        Args:
            x_min: Минимальная граница по оси X
            x_max: Максимальная граница по оси X
            y_min: Минимальная граница по оси Y
            y_max: Максимальная граница по оси Y
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
    def update(self, current_position: Tuple[float, float], current_velocity: float,
               velocity_min: float, velocity_max: float, current_direction: float, 
               is_first_move: bool, time_ms: int) -> Tuple[Tuple[float, float], float, float]:
        """
        Обновляет позицию, скорость и направление устройства на основе модели случайного блуждания.

        Args:
            current_position: Текущие координаты устройства (x, y).
            current_velocity: Текущая скорость устройства (м/с).
            velocity_min: Минимальная скорость устройства (м/с).
            velocity_max: Максимальная скорость устройства (м/с).
            current_direction: Текущее направление движения (радианы).
            is_first_move: Флаг, указывающий, является ли это первым движением устройства.
            time_ms: Время, прошедшее с последнего обновления (миллисекунды).

        Returns:
            new_position: Новые координаты устройства (x, y).
            new_velocity: Новая скорость устройства (м/с).
            new_direction: Новое направление движения (радианы).
            is_first_move: Обновленный флаг, указывающий, завершено ли первое движение.
        """
        time_s = time_ms / 1000.0
        
        if is_first_move:
            current_velocity = np.random.uniform(velocity_min, velocity_max)
            current_direction = np.random.uniform(0, 2 * np.pi)
            is_first_move = False
        
        delta_x = current_velocity * np.cos(current_direction) * time_s
        delta_y = current_velocity * np.sin(current_direction) * time_s
        
        new_x = current_position[0] + delta_x
        new_y = current_position[1] + delta_y
        
        # Здесь сделан отскок при выходе за границу. Вроде работает, а вроде и нет.
        # Нужно подумать и немного переделать
        if new_x < self.x_min or new_x > self.x_max:
            self.direction = np.pi - self.direction
            new_x = current_position[0] + delta_x * np.cos(self.direction)

        if new_y < self.y_min or new_y > self.y_max:
            self.direction = -self.direction
            new_y = current_position[1] + delta_y * np.sin(self.direction)
            
        self.velocity = np.random.uniform(velocity_min, velocity_max)
        self.direction = np.random.uniform(0, 2 * np.pi)
        
        new_position = (new_x, new_y)
        new_velocity = self.velocity
        new_direction = self.direction

        return new_position, new_velocity, new_direction, is_first_move
    
class RandomWaypointModel:
    """
    Модель передвижения Random Waypoint для пользовательского устройства (UE).
    Устройство движется к случайным пунктам назначения с паузами между движениями.
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, pause_time: float):
        """
        Инициализация модели Random Waypoint.

        Args:
            x_min: Минимальная граница по оси X
            x_max: Максимальная граница по оси X
            y_min: Минимальная граница по оси Y
            y_max: Максимальная граница по оси Y
            pause_time: Время паузы между движениями (в мс)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.pause_time = pause_time 
    
    def _choose_new_destination(self, current_position: Tuple[float, float], 
                                velocity_min: float, velocity_max: float):
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
            np.random.uniform(self.y_min, self.y_max)
        )
        
        new_velocity = np.random.uniform(velocity_min, velocity_max)
        
        delta_x = new_destination[0] - current_position[0]
        delta_y = new_destination[1] - current_position[1]
        new_direction = np.arctan2(delta_y, delta_x)
        
        return new_destination, new_velocity, new_direction, False
        
    
    def update(self, current_position: Tuple[float, float], current_velocity: float,
               velocity_min: float, velocity_max: float, current_direction: float, 
               destination: Tuple[float, float], is_paused: bool, pause_timer: float, 
               time_ms: int):
        
        """
        Обновляет позицию, скорость, направление и состояние устройства на основе модели Random Waypoint.

        Args:
            current_position: Текущие координаты устройства (x, y)
            current_velocity: Текущая скорость устройства (м/с)
            velocity_min: Минимальная скорость устройства (м/с)
            velocity_max: Максимальная скорость устройства (м/с)
            current_direction: Текущее направление движения (радианы)
            destination: Текущая точка назначения (x, y)
            is_paused: Флаг, указывающий, находится ли устройство в режиме паузы
            pause_timer: Текущее время, прошедшее в режиме паузы (мс)
            time_ms: Время, прошедшее с последнего обновления (миллисекунды)

        Returns:
            new_position: Новые координаты устройства (x, y)
            new_velocity: Новая скорость устройства (м/с)
            new_direction: Новое направление движения (радианы)
            destination: Текущая точка назначения (x, y)
            is_paused: Флаг, указывающий, находится ли устройство в режиме паузы
            pause_timer: Обновленное время, прошедшее в режиме паузы (мс)
        """
        time_s = time_ms / 1000.0
        
        if is_paused:
            pause_timer += time_ms
            if pause_timer >= self.pause_time:
                new_destination, new_velocity, new_direction, is_paused = self._choose_new_destination(
                    current_position, velocity_min, velocity_max)
                pause_timer = 0.0
                return current_position, new_velocity, new_direction, new_destination, is_paused, pause_timer
            return current_position, 0.0, current_direction, destination, is_paused, pause_timer
        
        delta_x = destination[0] - current_position[0]
        delta_y = destination[1] - current_position[1]
        distance = np.sqrt(delta_x**2 + delta_y**2)
        
        if distance <= current_velocity * time_s:
            new_position = destination
            is_paused = True
            return new_position, 0.0, current_direction, destination, is_paused, pause_timer
        else:
            new_x = current_position[0] + current_velocity * np.cos(current_direction) * time_s
            new_y = current_position[1] + current_velocity * np.sin(current_direction) * time_s
            new_position = (new_x, new_y)
            return new_position, current_velocity, current_direction, destination, is_paused, pause_timer
    
def test_random_walk():
    random_walk_model = RandomWalkModel(x_min=-100, x_max=100, y_min=-100, y_max=100)
    
    ue = UserEquipment(UE_ID=1, x=0.0, y=0.0, velocity_min=1.0, velocity_max=5.0)
    ue.SET_MOBILITY_MODEL(random_walk_model)
    
    simulation_duration = 100000
    update_interval = 500
    
    x_coords = [ue.position[0]]
    y_coords = [ue.position[1]]
    
    for t in range(simulation_duration):
        if t % 500 == 0:
            ue.UPD_POSITION(time_ms=update_interval, bs_position=(0, 0))
            
            x_coords.append(ue.position[0])
            y_coords.append(ue.position[1])
    
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, linestyle='-', color='b', label='Траектория')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.title("Траектория движения пользователя")
    plt.xlabel("Ось X (м)")
    plt.ylabel("Ось Y (м)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
def test_random_waypoint():
    random_waypoint_model = RandomWaypointModel(x_min=-100, x_max=100, y_min=-100, y_max=100, pause_time=50)
    
    ue = UserEquipment(UE_ID=1, x=0.0, y=0.0, velocity_min=1.0, velocity_max=5.0)
    ue.SET_MOBILITY_MODEL(random_waypoint_model)
    
    simulation_duration = 1000000
    update_interval = 500
    
    x_coords = [ue.position[0]]
    y_coords = [ue.position[1]]
    
    for t in range(simulation_duration):
        if t % 500 == 0:
            ue.UPD_POSITION(time_ms=update_interval, bs_position=(0, 0))
            
            x_coords.append(ue.position[0])
            y_coords.append(ue.position[1])
            
    plt.figure(figsize=(8, 8))
    plt.plot(x_coords, y_coords, linestyle='-', color='b', label='Траектория')
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.title("Траектория движения пользователя")
    plt.xlabel("Ось X (м)")
    plt.ylabel("Ось Y (м)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    #test_random_walk()
    test_random_waypoint()