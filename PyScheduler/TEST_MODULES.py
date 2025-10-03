import numpy as np
import math
import matplotlib.pyplot as plt
from UE_MODULE import UserEquipment, UECollection
from BS_MODULE import BaseStation, Packet
from RES_GRID import RES_GRID_LTE
from SCHEDULER import RoundRobinScheduler, BestCQIScheduler, ProportionalFairScheduler
from MOBILITY_MODEL import RandomWalkModel, RandomWaypointModel, RandomDirectionModel, GaussMarkovModel
from CHANNEL_MODEL import RMaModel, UMaModel, UMiModel

def visualize_users_mobility(ue_collection: UECollection, bs: BaseStation,
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
        x_coords = [x for x, _ in ue.coordinates]
        y_coords = [y for _, y in ue.coordinates]
        plt.plot(x_coords, y_coords, label=f"UE {ue.UE_ID}")
    
    plt.legend()
    plt.grid(True)
    plt.show()
    
def visualize_users_sinr(ue_collection: UECollection, sim_duration: float,
                            update_interval: float):
    
    tti_range = np.arange(0, sim_duration, update_interval)
    
    plt.figure(figsize=(10, 6))
    plt.title("График SINR во времени для всех пользователей")
    plt.xlabel("TTI")
    plt.ylabel("SINR (dB)")
    for ue in ue_collection.GET_ALL_USERS():
        sinr_values = ue.SINR_values
        plt.plot(tti_range, sinr_values, label=f'UE{ue.UE_ID}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def print_users_stats(ue_collection: UECollection, tti: int, bs: BaseStation, 
                      sched_result: dict):
    
    print(f"\n[TTI {tti}]")
    print("=" * 40)
    
    allocation = sched_result["allocation"]
    
    for ue in ue_collection.GET_ALL_USERS():
        ue_id = ue.UE_ID
        num_rbs = len(allocation.get(ue_id, []))
    
        displacement = np.hypot(
            ue.position[0] - ue.coordinates[-2][0],
            ue.position[1] - ue.coordinates[-2][1]
        )
    
        print(f"UE {ue_id}:")
        print(f"\tRBs выделено        : {num_rbs}")
        print(f"\tТекущая скорость    : {ue.current_dl_throughput} bit/s")
        print(f"\tСредняя скорость    : {ue.average_throughput:.2f} bit/s")
        print("\t+" + "-" * 35)
        print(f"\tSINR                : {ue.SINR:.2f} dB")
        print(f"\tCQI                 : {ue.cqi}")
        print(f"\tРазмер буфера       : {bs.ue_buffers[1].sizes[1]} B")
        print("\t+" + "-" * 35)
        print(f"\tПред. позиция       : {ue.coordinates[-2]}")
        print(f"\tТекущая позиция     : {ue.position}")
        print(f"\tСмещение            : {displacement} m")
        print("-" * 40)

def debug_simulation():
    """Пример использования разработанных модулей"""
    
    sim_duration = 20 # Время симуляции (в мс)
    update_interval = 1 # Интервал обновления параметров пользователя (в мс)
    num_frames = int(np.ceil(sim_duration / 10)) # Кол-во кадров (для ресурсной сетки)
    bandwidth = 10 # Ширина полосы (в МГц)
    inf = math.inf 
    
    # Создание и настройка базовой станции
    bs = BaseStation(x=0, y=0, bandwidth=bandwidth, global_max=inf, per_ue_max=inf)
       
    # Создание коллекции пользовательских устройств
    ue_collection = UECollection()
    
    # Создание и настройка пользовательских устройств
    ue1 = UserEquipment(UE_ID=1, x=800, y=800, ue_class="pedestrian")
    ue2 = UserEquipment(UE_ID=2, x=400, y=-200, ue_class="cyclist")
    ue3 = UserEquipment(UE_ID=3, x=-500, y=-500, ue_class="car")
    
    # Создание модели передвижения пользователей
    random_waypoint = RandomWaypointModel(x_min=-1000, 
                                          x_max=1000, 
                                          y_min=-1000, 
                                          y_max=1000, 
                                          pause_time=0)
  
    # Назначение пользователям модели передвижения
    ue1.SET_MOBILITY_MODEL(random_waypoint)
    ue2.SET_MOBILITY_MODEL(random_waypoint)
    ue3.SET_MOBILITY_MODEL(random_waypoint)
    
    # Создание модели радиоканала  
    uma = UMaModel(bs)
    
    # Назначение пользователям модели радиоканала
    ue1.SET_CH_MODEL(uma)
    ue2.SET_CH_MODEL(uma)
    ue3.SET_CH_MODEL(uma)
    
    # Регистрация пользователей в базовой станции
    bs.REG_UE(ue1)
    bs.REG_UE(ue2)
    bs.REG_UE(ue3)
    
    # Добавление пользовательских устройств в коллекцию
    ue_collection.ADD_USER(ue1)
    ue_collection.ADD_USER(ue2)
    ue_collection.ADD_USER(ue3)
    
    # Имитация Full Buffer 
    bs.ue_buffers[1].ADD_PACKET(Packet(size=inf, ue_id=1, creation_time=0), current_time=0)
    bs.ue_buffers[2].ADD_PACKET(Packet(size=inf, ue_id=2, creation_time=0), current_time=0)
    bs.ue_buffers[3].ADD_PACKET(Packet(size=inf, ue_id=3, creation_time=0), current_time=0)

    # Создание ресурсной сетки
    lte_grid = RES_GRID_LTE(bandwidth=bandwidth, num_frames=num_frames)
    
    # Создание планировщика
    scheduler = ProportionalFairScheduler(lte_grid, bs)

    # Основной цикл симуляции
    for current_time in range(update_interval, sim_duration + 1, update_interval):
        
        # Обновление состояния пользователей
        ue_collection.UPDATE_ALL_USERS(time_ms=current_time, 
                                       update_interval=update_interval, 
                                       bs_position=bs.position, 
                                       bs_height=bs.height)
    
        # Цикл для планирования ресурсов (по TTI)
        for tti in range(current_time - update_interval, current_time):
            
            # Подготовка данных для планировщика     
            users = ue_collection.GET_USERS_FOR_SCHEDULER()
            
            # Планирование ресурсов 
            sched_result = scheduler.schedule(tti, users)
            
            # Вывод статистики для каждого пользователя
            print_users_stats(ue_collection=ue_collection, 
                              tti=tti, 
                              bs=bs,
                              sched_result=sched_result)
            
            
    # Визуализация передвижения пользователей
    visualize_users_mobility(ue_collection=ue_collection, 
                            bs=bs, 
                            x_min=-1000, 
                            x_max=1000, 
                            y_min=-1000, 
                            y_max=1000)
    
    # Визуализация SINR пользователей во времени     
    visualize_users_sinr(ue_collection=ue_collection,
                            sim_duration=sim_duration, 
                            update_interval=update_interval)
    
if __name__ == "__main__":
    debug_simulation()