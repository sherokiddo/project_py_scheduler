import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from UE_MODULE import UserEquipment, Buffer, UECollection

class RandomWalkModel:
    """
    Модель передвижения Random Walk для пользовательского устройства (UE).
    Устройство движется в случайном направлении в пределах заданных границ.
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, 
                 velocity_min: float, velocity_max: float, time_interval: float):
        """
        Инициализация модели случайного блуждания.

        Args:
            x_min: Минимальная граница по оси X
            x_max: Максимальная граница по оси X
            y_min: Минимальная граница по оси Y
            y_max: Максимальная граница по оси Y
            velocity_min: Минимальная скорость движения (м/с)
            velocity_max: Максимальная скорость движения (м/с)
            time_interval: Интервал времени для каждого шага (секунды)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.velocity_min = velocity_min
        self.velocity_max = velocity_max
        self.time_interval = time_interval
        
        self.velocity = np.random.uniform(self.velocity_min, self.velocity_max)
        self.direction = np.random.uniform(0, 2 * np.pi)
        
    def update(self, current_position: Tuple[float, float], current_velocity: float,
               current_direction: float, time_ms: int) -> Tuple[Tuple[float, float], float, float]:
        """
        Обновляет позицию, скорость и направление устройства на основе модели случайного блуждания.

        Args:
            current_position: Текущие координаты устройства (x, y)
            current_velocity: Текущая скорость устройства (м/с)
            current_direction: Текущее направление движения (радианы)
            time_ms: Время, прошедшее с последнего обновления (миллисекунды)

        Returns:
            new_position: Новые координаты устройства (x, y)
            new_velocity: Новая скорость устройства (м/с)
            new_direction: Новое направление движения (радианы)
        """
        time_s = time_ms / 1000.0
        
        delta_x = self.velocity * np.cos(self.direction) * time_s
        delta_y = self.velocity * np.sin(self.direction) * time_s
        
        new_x = current_position[0] + delta_x
        new_y = current_position[1] + delta_y
        
        if new_x < self.x_min or new_x > self.x_max:
            self.direction = np.pi - self.direction
            new_x = current_position[0] + delta_x * np.cos(self.direction)

        if new_y < self.y_min or new_y > self.y_max:
            self.direction = -self.direction
            new_y = current_position[1] + delta_y * np.sin(self.direction)
            
        self.velocity = np.random.uniform(self.velocity_min, self.velocity_max)
        self.direction = np.random.uniform(0, 2 * np.pi)
        
        new_position = (new_x, new_y)
        new_velocity = self.velocity
        new_direction = self.direction

        return new_position, new_velocity, new_direction
    
def test_random_walk():
    random_walk_model = RandomWalkModel(x_min=-100, x_max=100, y_min=-100, y_max=100, 
                                        velocity_min=1.0, velocity_max=5.0, time_interval=1.0)
    
    ue = UserEquipment(UE_ID=1, x=0.0, y=0.0)
    ue.SET_MOVEMENT_MODEL(random_walk_model)
    
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
    
if __name__ == "__main__":
    test_random_walk()