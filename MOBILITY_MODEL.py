import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from UE_MODULE import UserEquipment, UECollection

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
                                velocity_min: float, velocity_max: float) -> Tuple[Tuple[float, float], 
                                                                                   float, float, bool]:
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
               time_ms: int) -> Tuple[Tuple[float, float], float, float, 
                                      Tuple[float, float], bool, float]:
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

class RandomDirectionModel:
    """
    Модель передвижения Random Direction для пользовательского устройства (UE).
    Устройство движется к границе области моделирования в случайном направлении,
    делает паузу, а затем выбирает новое направление.
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, pause_time: float):
        """
        Инициализация модели Random Direction.

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
        
    def _choose_new_direction(self, current_position: Tuple[float, float], velocity_min: float, 
                              velocity_max: float, is_first_move: bool) -> Tuple[Tuple[float, float], 
                                                                                 float, float, bool, bool]:
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
        while not (self.x_min <= new_destination[0] <= self.x_max and self.y_min <= new_destination[1] <= self.y_max):
            new_direction = np.random.uniform(0, 2 * np.pi)
            new_destination = self._calculate_boundary_point(current_position, new_direction)
        
        new_velocity = np.random.uniform(velocity_min, velocity_max)
        
        return new_destination, new_velocity, new_direction, False, is_first_move
    
    def _calculate_boundary_point(self, current_position: Tuple[float, float], 
                              direction: float) -> Tuple[float, float]:
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
        
    def update(self, current_position: Tuple[float, float], current_velocity: float,
               velocity_min: float, velocity_max: float, current_direction: float, 
               destination: Tuple[float, float], is_paused: bool, pause_timer: float, 
               is_first_move: bool, time_ms: int) -> Tuple[Tuple[float, float], float, float, 
                                                           Tuple[float, float], bool, float, bool]:
        """
        Обновляет позицию, скорость, направление и состояние устройства на основе модели Random Direction.
    
        Args:
            current_position: Текущие координаты устройства (x, y).
            current_velocity: Текущая скорость устройства (м/с).
            velocity_min: Минимальная скорость устройства (м/с).
            velocity_max: Максимальная скорость устройства (м/с).
            current_direction: Текущее направление движения (радианы).
            destination: Текущая точка назначения (x, y).
            is_paused: Флаг, указывающий, находится ли устройство в режиме паузы.
            pause_timer: Текущее время, прошедшее в режиме паузы (мс).
            is_first_move: Флаг, указывающий, является ли это первым движением устройства.
            time_ms: Время, прошедшее с последнего обновления (миллисекунды).
    
        Returns:
            new_position: Новые координаты устройства (x, y).
            new_velocity: Новая скорость устройства (м/с).
            new_direction: Новое направление движения (радианы).
            destination: Текущая точка назначения (x, y).
            is_paused: Флаг, указывающий, находится ли устройство в режиме паузы.
            pause_timer: Обновленное время, прошедшее в режиме паузы (мс).
            is_first_move: Обновленный флаг, указывающий, завершено ли первое движение.
        """
        time_s = time_ms / 1000.0
        
        if is_paused:
            pause_timer += time_ms
            if pause_timer >= self.pause_time:
                new_destination, new_velocity, new_direction, is_paused, is_first_move = self._choose_new_direction(
                    current_position, velocity_min, velocity_max, is_first_move)
                pause_timer = 0.0
                return current_position, new_velocity, new_direction, new_destination, is_paused, pause_timer, is_first_move
            return current_position, 0.0, current_direction, destination, is_paused, is_first_move
        
        delta_x = destination[0] - current_position[0]
        delta_y = destination[1] - current_position[1]
        distance = np.sqrt(delta_x**2 + delta_y**2)
        
        if distance <= current_velocity * time_s:
            new_position = destination
            is_paused = True
            return new_position, 0.0, current_direction, destination, is_paused, pause_timer, is_first_move
        else:
            new_x = current_position[0] + current_velocity * np.cos(current_direction) * time_s
            new_y = current_position[1] + current_velocity * np.sin(current_direction) * time_s
            new_position = (new_x, new_y)
            return new_position, current_velocity, current_direction, destination, is_paused, pause_timer, is_first_move

def visualize_user_mobility(ue_collection: UECollection, x_min: float, 
                            x_max: float, y_min: float, y_max: float):
    """
    Функция для построения карты передвижения пользователей.

    Args:
        ue_collection: Коллекция пользователей (объект UECollection)
        x_min: Минимальная граница по оси X
        x_max: Максимальная граница по оси X
        y_min: Минимальная граница по оси Y
        y_max: Максимальная граница по оси Y
    """
    plt.figure(figsize=(10, 6))
    plt.title("Карта передвижения пользователей")
    plt.xlabel("X координата (м)")
    plt.ylabel("Y координата (м)")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    for ue in ue_collection.GET_ALL_USERS():
        x_coords = ue.x_coordinates
        y_coords = ue.y_coordinates
        plt.plot(x_coords, y_coords, label=f"UE {ue.UE_ID}")
    
    plt.legend()
    plt.grid(True)
    plt.show()
    
def example_usage():
    """Пример использования модели передвижения пользователей"""
    ue_collection = UECollection()
    
    ue1 = UserEquipment(UE_ID=1, x=0, y=0, velocity_min=5, velocity_max=20)
                                          
    random_direction_model = RandomDirectionModel(x_min=-30, x_max=30, y_min=-30, 
                                                y_max=30, pause_time=50)                            
    
    ue1.SET_MOBILITY_MODEL(random_direction_model)
    
    ue_collection.ADD_USER(ue1)
    
    simulation_duration = 200000
    update_interval = 500
    
    for t in range(simulation_duration):
        if t % update_interval == 0:
            ue_collection.UPDATE_ALL_USERS(time_ms=update_interval)
        
    visualize_user_mobility(ue_collection=ue_collection, x_min=-50, x_max=50, 
                            y_min=-50, y_max=50)  
    
if __name__ == "__main__":
    example_usage()