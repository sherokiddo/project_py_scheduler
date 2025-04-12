import numpy as np
import matplotlib.pyplot as plt
from UE_MODULE import UserEquipment, UECollection
from MOBILITY_MODEL import RandomWalkModel, RandomWaypointModel, RandomDirectionModel, GaussMarkovModel
from CHANNEL_MODEL import RMaModel, UMaModel, UMiModel
from TRAFFIC_MODEL import PoissonModel
from BS_MODULE import BaseStation

def visualize_user_mobility(ue_collection: UECollection, bs: BaseStation,
                            x_min: float, x_max: float, y_min: float, y_max: float):
    """
    Функция для построения карты передвижения пользователей.

    Args:
        ue_collection: Коллекция пользователей (объект UECollection)
        x_min: Минимальная граница по оси X
        x_max: Максимальная граница по оси X
        y_min: Минимальная граница по оси Y
        y_max: Максимальная граница по оси Y
    """
    x_bs, y_bs = bs.position
    
    plt.figure(figsize=(10, 6))
    plt.title("Карта передвижения пользователей")
    plt.xlabel("X координата (м)")
    plt.ylabel("Y координата (м)")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.plot(x_bs, y_bs, marker = 'o', label='Base Station')
    
    for ue in ue_collection.GET_ALL_USERS():
        x_coords = ue.x_coordinates
        y_coords = ue.y_coordinates
        plt.plot(x_coords, y_coords, label=f"UE {ue.UE_ID}")
    
    plt.legend()
    plt.grid(True)
    plt.show()
    
def visualize_sinr_cqi_user(ue_collection: UECollection, simulation_duration: float,
                            update_interval: float):
    
    time = np.arange(0, simulation_duration, update_interval)
    
    for ue in ue_collection.GET_ALL_USERS():
        plt.figure(figsize=(10, 6))
        plt.title(f"График SINR во времени для UE {ue.UE_ID}")
        plt.xlabel("Время (мс)")
        plt.ylabel("SINR (dB)")
        sinr_values = ue.SINR_values
        plt.plot(time, sinr_values)
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.title(f"График CQI во времени для UE {ue.UE_ID}")
        plt.xlabel("Время (мс)")
        plt.ylabel("CQI")
        cqi_values = ue.CQI_values
        plt.step(time, cqi_values)
        plt.grid(True)
        plt.show()

def example_usage_modules():
    """Пример использования разработанных модулей"""
    
    # Создание коллекции пользовательских устройств
    ue_collection = UECollection()
    
    # Создание объекта базовой станции
    bs = BaseStation(x=3000, y=3000)
    
    # Создание объектов пользовательских устройств
    ue1 = UserEquipment(UE_ID=1, x=2300, y=2300, ue_class="indoor")
    #ue2 = UserEquipment(UE_ID=2, x=2000, y=2000, ue_class="car")
    
    # Границы симуляции
    x_min_global = -5000
    y_min_global = -5000
    x_max_global = 5000
    y_max_global = 5000    

    x_min_h = 2295
    y_min_h = 2295
    x_max_h = 2305
    y_max_h = 2305                      

    # Создание объектов моделей передвижения пользователей
    gauss_markov_model = GaussMarkovModel(x_min=x_min_global, x_max=x_max_global, 
                                          y_min=y_min_global, y_max=y_max_global) 
    
    random_waypoint_model = RandomWaypointModel(x_min=x_min_h, x_max=x_max_h, 
                                                y_min=y_min_h, y_max=y_max_h, 
                                                pause_time=500)
  
    # Создание объекта модели канала  
    umi_channel_model = UMiModel(bs)  

    # Создание объекта модели генерации трафика
    poisson_traffic_model = PoissonModel(packet_rate=500)         
    
    # Установка модели передвижения и модели канала для устройства
    ue1.SET_MOBILITY_MODEL(random_waypoint_model)
    #ue2.SET_MOBILITY_MODEL(gauss_markov_model)
    
    ue1.SET_CH_MODEL(umi_channel_model)
    #ue2.SET_CH_MODEL(umi_channel_model)
    
    ue1.SET_TRAFFIC_MODEL(poisson_traffic_model)
    #ue2.SET_TRAFFIC_MODEL(poisson_traffic_model)
    
    # Добавление пользовательских устройств в коллекцию
    ue_collection.ADD_USER(ue1)
    #ue_collection.ADD_USER(ue2)
    
    # Временные параметры симуляции
    simulation_duration = 50000
    update_interval = 250
    
    # Симуляция и визуализация результатов
    for t in range(1, simulation_duration + 1):
        if t % update_interval == 0:
            ue_collection.UPDATE_ALL_USERS(time_ms=t, update_interval=update_interval, 
                                           bs_position=bs.position, bs_height=bs.height, 
                                           indoor_boundaries=(x_min_h, y_min_h, x_min_h, x_max_h))
        
    visualize_user_mobility(ue_collection=ue_collection, bs=bs, 
                            x_min=1500, x_max=3500, y_min=1500, y_max=3500)
    
    visualize_sinr_cqi_user(ue_collection, simulation_duration, update_interval)
    
if __name__ == "__main__":
    example_usage_modules()