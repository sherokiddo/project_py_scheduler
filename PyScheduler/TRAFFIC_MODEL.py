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
import numpy as np

class PoissonModel:
    """
    Пуассоновская модель генерации трафика.
    Пакеты создаются с интервалами, соответствующими экспоненциальному распределению, 
    моделируя среднюю скорость генерации пакетов.
    """
    def __init__(self, packet_rate: float, min_packet_size: float = 150, 
                 max_packet_size: float = 1500):
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
            
            packets.append({
                'size': packet_size,
                'creation_time': generate_time,
                'priority': 0
                })
            
        return packets
    
class OnOffModel:
    """
    ON/OFF модель генерации трафика.
    Устройство чередует активные (ON) и неактивные (OFF) состояния. 
    В состоянии ON пакеты генерируются с заданной интенсивностью.
    """
    def __init__(self, duration_on: float, duration_off: float, packet_rate: float,
                 min_packet_size: float = 150, max_packet_size: float = 1500):
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
                
            self.device_states[UE_ID] = {
                "state": initial_state,
                "end_state_time": end_state_time
            }
        
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
                    
                    self.device_states[UE_ID] = {
                        "state": "OFF",
                        "end_state_time": end_state_time
                    }
                    
                else:
                    end_generate = current_time
                    
                mean_interval_ms = 1000.0 / self.packet_rate
                
                while t < end_generate:
                    interval = np.random.exponential(mean_interval_ms)
                    t += interval
                    
                    if t > end_generate:
                        break
                    
                    packet_size = np.random.randint(self.min_packet_size, self.max_packet_size)
                    
                    packets.append({
                        'size': packet_size,
                        'creation_time': t,
                        'priority': 0
                    })
                    
            elif state_data["state"] == "OFF":
                if t < state_data["end_state_time"] <= current_time:
                    end_off_state = state_data["end_state_time"]
                    
                    on_state_time = np.random.exponential(self.duration_on)
                    on_state_time = on_state_time * 1000
                    end_state_time = end_off_state + on_state_time
                    
                    self.device_states[UE_ID] = {
                        "state": "ON",
                        "end_state_time": end_state_time
                    }
                    
                else:
                    end_off_state = current_time
                    
                t = end_off_state
                
            state_data = self.device_states[UE_ID]
        
        return packets
            
def test_traffic_models():
    """
    Тестирование и визуализация работы моделей трафика.
    """
    poisson_model = PoissonModel(packet_rate=5)      
    onoff_model = OnOffModel(duration_on=2, duration_off=3, packet_rate=25)

    simulation_duration = 60000
    update_interval = 250
    
    traffic_poisson = []
    traffic_onoff = []
    
    for t in range(1, simulation_duration + 1):
        if t % update_interval == 0:
            
            packets_poisson = poisson_model.generate_traffic(current_time=t,
                                                             update_interval=update_interval)
            
            packets_onoff = onoff_model.generate_traffic(UE_ID=1, 
                                                         current_time=t, 
                                                         update_interval=update_interval)

            traffic_poisson.extend(packets_poisson)
            traffic_onoff.extend(packets_onoff)

    timestamps_poisson = [packet["creation_time"] for packet in traffic_poisson]
    sizes_poisson = [packet["size"] for packet in traffic_poisson]
    
    timestamps_onoff = [packet["creation_time"] for packet in traffic_onoff]
    sizes_onoff = [packet["size"] for packet in traffic_onoff]

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.stem(timestamps_poisson, sizes_poisson, label='Пакеты')
    plt.xlabel("Время (мс)")
    plt.ylabel("Размер пакета (байты)")
    plt.title("Пуассоновская модель трафика")
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.stem(timestamps_onoff, sizes_onoff, label='Пакеты')
    plt.xlabel("Время (мс)")
    plt.ylabel("Размер пакета (байты)")
    plt.title("ON/OFF модель трафика")
    plt.legend()
    plt.grid()
    plt.show()
        
if __name__ == "__main__":
    test_traffic_models()