import numpy as np
import matplotlib.pyplot as plt
from UE_MODULE import UserEquipment, UECollection
from MOBILITY_MODEL import RandomWalkModel, RandomWaypointModel, RandomDirectionModel, GaussMarkovModel
from CHANNEL_MODEL import RMaModel
from BS_MODULE import BaseStation

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

def example_usage_mobility_model():
    """Пример использования модели передвижения пользователей"""
    ue_collection = UECollection()
    
    bs = BaseStation(x=3000, y=3000)
    
    ue1 = UserEquipment(UE_ID=1, x=0, y=0, ue_class="car") 

    x_min_h = -1000
    y_min_h = -1000
    x_max_h = 1000
    y_max_h = 1000                      

    gauss_markov_model = GaussMarkovModel(x_min=x_min_h, x_max=x_max_h, y_min=y_min_h, y_max=y_max_h)  

    rma_channel_model = RMaModel(bs)                       
    
    ue1.SET_MOBILITY_MODEL(gauss_markov_model)
    ue1.SET_CH_MODEL(rma_channel_model)
    
    ue_collection.ADD_USER(ue1)
    
    simulation_duration = 100000
    update_interval = 250
    
    for t in range(simulation_duration):
        if t % update_interval == 0:
            ue_collection.UPDATE_ALL_USERS(time_ms=update_interval, bs_position=bs.position, 
                                           bs_height=bs.height)
        
    visualize_user_mobility(ue_collection=ue_collection, x_min=-1000, x_max=1000, 
                            y_min=-1000, y_max=1000)
    
    visualize_sinr_cqi_user(ue_collection, simulation_duration, update_interval)
    
if __name__ == "__main__":
    example_usage_mobility_model()