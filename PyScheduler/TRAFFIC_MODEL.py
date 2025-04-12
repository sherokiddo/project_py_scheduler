import numpy as np

class PoissonModel:
    
    def __init__(self, packet_rate: float, min_packet_size: float = 150, 
                 max_packet_size: float = 1500):
        
        self.packet_rate = packet_rate
        self.min_packet_size = min_packet_size
        self.max_packet_size = max_packet_size
        
    def generate_traffic(self, current_time: int, update_interval: int):
        
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
        
        
        