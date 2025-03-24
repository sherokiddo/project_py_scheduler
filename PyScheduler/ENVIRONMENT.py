"""
#------------------------------------------------------------------------------
# Модуль: ENVIRONMENT - Среда симуляции для системы LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет среду для симуляции работы системы LTE, включая взаимодействие
# пользовательских устройств, планировщиков ресурсов и визуализацию результатов.

# Версия: 1.0.0
# Дата последнего изменения: 2025-03-18
# Версия Python Kernel: 3.12.9

# Зависимости:
# - UE_MODULE.py (модели пользовательского оборудования)
# - LTE_GRID_ver_alpha.py (модель ресурсной сетки LTE)
# - SCHEDULER.py (планировщики ресурсов)
#------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
from UE_MODULE import UserEquipment, UECollection
from LTE_GRID_ver_alpha import RES_GRID_LTE
from SCHEDULER import RoundRobinScheduler

class SimulationEnvironment:
    """
    Класс, представляющий среду симуляции для системы LTE.
    """
    
    def __init__(self, bandwidth_mhz: float = 10, num_frames: int = 10, 
                 bs_position: tuple = (500, 500)):
        """
        Инициализация среды симуляции.
        
        Args:
            bandwidth_mhz: Полоса частот в МГц (1.4, 3, 5, 10, 15, 20)
            num_frames: Количество кадров для симуляции
            bs_position: Координаты базовой станции (x, y) в метрах
        """
        self.lte_grid = RES_GRID_LTE(bandwidth=bandwidth_mhz, num_frames=num_frames)
        self.ue_collection = UECollection(bs_position=bs_position)
        self.scheduler = None
        self.simulation_time = 0  # Текущее время симуляции в мс
        self.results = {
            'throughput_by_ue': {},
            'allocated_rbs_by_tti': {},
            'total_allocated_rbs': 0
        }
    
    def set_scheduler(self, scheduler_type: str = "round_robin"):
        """
        Установить планировщик ресурсов.
        
        Args:
            scheduler_type: Тип планировщика ("round_robin", "proportional_fair", и т.д.)
        """
        if scheduler_type == "round_robin":
            self.scheduler = RoundRobinScheduler(self.lte_grid)
        else:
            raise ValueError(f"Неизвестный тип планировщика: {scheduler_type}")
    
    def add_user(self, ue: UserEquipment):
        """
        Добавить пользователя в симуляцию.
        
        Args:
            ue: Объект пользовательского устройства
        """
        self.ue_collection.ADD_USER(ue)
        self.results['throughput_by_ue'][ue.UE_ID] = []
    
    def run_simulation(self, duration_ms: int, time_step_ms: int = 1):
        """
        Запустить симуляцию на заданную длительность.
        
        Args:
            duration_ms: Длительность симуляции в миллисекундах
            time_step_ms: Шаг времени симуляции в миллисекундах
        """
        if not self.scheduler:
            self.set_scheduler("round_robin")
        
        self.simulation_time = 0
        
        while self.simulation_time < duration_ms:
            # Определяем текущий TTI
            current_tti = self.simulation_time % (10 * self.lte_grid.num_frames)
            
            # Обновляем состояние пользователей
            self.ue_collection.UPDATE_ALL_USERS(self.simulation_time)
            
            # Вызываем планировщик для распределения ресурсов
            result = self.scheduler.schedule_with_ue_collection(current_tti, self.ue_collection)
            
            # Обновляем пропускную способность для пользователей
            self.scheduler.update_throughput(self.ue_collection, result['allocation'], time_step_ms)
            
            # Сохраняем результаты
            self.results['allocated_rbs_by_tti'][self.simulation_time] = result['statistics']['allocated_rbs']
            self.results['total_allocated_rbs'] += result['statistics']['allocated_rbs']
            
            for ue in self.ue_collection.GET_ALL_USERS():
                self.results['throughput_by_ue'][ue.UE_ID].append(ue.current_throughput)
            
            # Увеличиваем время
            self.simulation_time += time_step_ms
    
    def visualize_resource_grid(self, tti_start: int = 0, tti_end: Optional[int] = None,
                              show_UE_IDs: bool = True, save_path: Optional[str] = None):
        """
        Визуализировать ресурсную сетку LTE.
        
        Args:
            tti_start: Начальный TTI для визуализации
            tti_end: Конечный TTI для визуализации (не включительно)
            show_UE_IDs: Показывать ли ID пользователей на графике
            save_path: Путь для сохранения изображения (если None, то изображение отображается)
        """
        self.lte_grid.visualize_grid(tti_start, tti_end, show_UE_IDs, save_path)
    
    def visualize_throughput(self, save_path: Optional[str] = None):
        """
        Визуализировать пропускную способность пользователей.
        
        Args:
            save_path: Путь для сохранения изображения (если None, то изображение отображается)
        """
        plt.figure(figsize=(12, 8))
        
        for ue_id, throughput_values in self.results['throughput_by_ue'].items():
            plt.plot(range(len(throughput_values)), throughput_values, label=f'UE {ue_id}')
        
        plt.xlabel('Время (мс)')
        plt.ylabel('Пропускная способность (бит/с)')
        plt.title('Пропускная способность пользователей')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_resource_allocation(self, save_path: Optional[str] = None):
        """
        Визуализировать распределение ресурсных блоков по времени.
        
        Args:
            save_path: Путь для сохранения изображения (если None, то изображение отображается)
        """
        plt.figure(figsize=(12, 8))
        
        times = list(self.results['allocated_rbs_by_tti'].keys())
        values = list(self.results['allocated_rbs_by_tti'].values())
        
        plt.bar(times, values, width=1.0)
        
        plt.xlabel('Время (мс)')
        plt.ylabel('Количество распределенных RB')
        plt.title('Распределение ресурсных блоков')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def print_statistics(self):
        """
        Вывести статистику симуляции.
        """
        print(f"=== Статистика симуляции ===")
        print(f"Длительность симуляции: {self.simulation_time} мс")
        print(f"Количество пользователей: {len(self.ue_collection.GET_ALL_USERS())}")
        print(f"Всего распределено RB: {self.results['total_allocated_rbs']}")
        
        print("\nСредняя пропускная способность пользователей:")
        for ue_id, throughput_values in self.results['throughput_by_ue'].items():
            avg_throughput = sum(throughput_values) / len(throughput_values) if throughput_values else 0
            print(f"UE {ue_id}: {avg_throughput:.2f} бит/с")

    def test_round_robin():
        """
        Тестирование алгоритма Round Robin.
        """
        # Создаем среду симуляции
        env = SimulationEnvironment(bandwidth_mhz=10, num_frames=1)
        
        # Добавляем пользователей
        for i in range(1, 6):
            x = np.random.randint(400, 600)
            y = np.random.randint(400, 600)
            ue = UserEquipment(UE_ID=i, x=x, y=y)
            env.add_user(ue)
        
        # Устанавливаем планировщик Round Robin
        env.set_scheduler("round_robin")
        
        # Запускаем симуляцию
        env.run_simulation(duration_ms=10)
        
        # Визуализируем результаты
        env.visualize_resource_grid(tti_start=0, tti_end=10)
        env.visualize_throughput()
        env.visualize_resource_allocation()
        env.print_statistics()
    
    if __name__ == "__main__":
        test_round_robin()