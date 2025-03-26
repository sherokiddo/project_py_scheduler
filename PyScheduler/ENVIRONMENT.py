"""
#------------------------------------------------------------------------------
# Модуль: ENVIRONMENT - Среда симуляции для системы LTE
#------------------------------------------------------------------------------
# Описание:
# Предоставляет среду для симуляции работы системы LTE, включая взаимодействие
# пользовательских устройств, планировщиков ресурсов и визуализацию результатов.
#------------------------------------------------------------------------------

# Версия: 1.0.0
# Дата последнего изменения: 2025-03-18
# Версия Python Kernel: 3.12.9
# Автор: Брагин Кирилл

# Зависимости:
# - UE_MODULE.py (модели пользовательского оборудования)
# - LTE_GRID_ver_alpha.py (модель ресурсной сетки LTE)
# - SCHEDULER.py (модель планировщика)
#------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
from UE_MODULE import UserEquipment, UECollection
from LTE_GRID_ver_alpha import RES_GRID_LTE
from SCHEDULER import RoundRobinScheduler
%matplotlib

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
        self.scheduler = RoundRobinScheduler(self.lte_grid)
        self.simulation_time = 0  # Текущее время симуляции в мс
        self.results = {
            'throughput_by_ue': {},
            'allocated_rbs_by_tti': {},
            'total_allocated_rbs': 0
        }
    
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
        self.simulation_time = 0
        
        # Предварительно проверяем, что у пользователей есть данные в буфере
        for ue in self.ue_collection.GET_ALL_USERS():
            if ue.buffer.GET_STATUS(0)['size'] == 0:
                # Если буфер пуст, добавляем в него данные
                ue.buffer.ADD_PACKET(10000, 0, 5)  # Добавляем пакет размером 10000 байт
        
        while self.simulation_time < duration_ms:
            # Определяем текущий TTI
            current_tti = self.simulation_time % (10 * self.lte_grid.num_frames)
            
            # Обновляем состояние пользователей
            self.ue_collection.UPDATE_ALL_USERS(self.simulation_time)
            
            # Получаем данные о пользователях для планировщика
            users_data = []
            for ue in self.ue_collection.GET_ACTIVE_USERS():
                buffer_status = ue.GET_BUFFER_STATUS(self.simulation_time)
                channel_quality = ue.GET_CH_QUALITY()
                
                users_data.append({
                    'UE_ID': ue.UE_ID,
                    'buffer_size': buffer_status['size'],
                    'packet_count': buffer_status['packet_count'],
                    'cqi': channel_quality['cqi'],
                    'SINR': channel_quality['SINR']
                })
            
            # Вызываем планировщик для распределения ресурсов
            result = self.scheduler.schedule(current_tti, users_data)
            
            # После назначения ресурсных блоков в методе run_simulation
            for ue_id, rb_indices in result['allocation'].items():
                ue = self.ue_collection.GET_USER(ue_id)
                if ue and rb_indices:
                    # Используем метод из планировщика для расчёта количества бит
                    bits_transmitted = self.scheduler._calculate_bits_transmitted(ue.cqi, len(rb_indices))
                    
                    # Обновляем статистику пропускной способности пользователя
                    ue.UPD_THROUGHPUT(bits_transmitted, time_step_ms)
                        
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
        num_rb = self.lte_grid.num_rb  # Количество RB
        total_tti = self.lte_grid.total_tti  # Общее количество TTI

        if tti_end is None:
            tti_end = total_tti

        if tti_end > total_tti:
            tti_end = total_tti

        # Создаем матрицу для визуализации
        grid_matrix = np.zeros((num_rb, tti_end - tti_start), dtype=int)

        # Заполняем матрицу данными о пользователях
        for tti in range(tti_start, tti_end):
            for freq_idx in range(num_rb):
                rb = self.lte_grid.GET_RB(tti, freq_idx)
                if rb and rb.UE_ID is not None:
                    grid_matrix[freq_idx, tti - tti_start] = rb.UE_ID
                else:
                    grid_matrix[freq_idx, tti - tti_start] = -1  # Используем -1 для пустых RB

        # Создаем Figure и Axes
        fig, ax = plt.subplots(figsize=(15, 7))

        # Подготовка данных для pcolormesh
        x = np.arange(0, tti_end - tti_start + 1)
        y = np.arange(0, num_rb + 1)
        
        # Используем pcolormesh для визуализации сетки
        cmap = plt.cm.get_cmap('viridis', len(self.ue_collection.GET_ALL_USERS()) + 1)
        cmap.set_under('lightgray')  # Цвет для пустых ячеек
        
        mesh = ax.pcolormesh(x, y, grid_matrix, cmap=cmap, vmin=0)

        # Настраиваем оси
        ax.set_xlabel('TTI')
        ax.set_ylabel('Resource Block Index')
        ax.set_xticks(np.arange(0.5, tti_end - tti_start + 0.5))
        ax.set_yticks(np.arange(0.5, num_rb + 0.5))
        ax.set_xticklabels(np.arange(tti_start, tti_end))
        ax.set_yticklabels(np.arange(0, num_rb))
        
        # Добавляем colorbar
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label('UE ID')

        # Добавляем аннотации с ID пользователей
        if show_UE_IDs:
            for tti_idx in range(tti_end - tti_start):
                for rb_idx in range(num_rb):
                    ue_id = grid_matrix[rb_idx, tti_idx]
                    if ue_id != -1:  # Если RB не пустой
                        ax.text(tti_idx + 0.5, rb_idx + 0.5, str(ue_id),
                               ha='center', va='center', color='white', fontweight='bold')

        # Добавляем заголовок
        ax.set_title('Resource Grid Allocation')

        # Сохраняем изображение или отображаем его
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
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
    env = SimulationEnvironment(bandwidth_mhz=15, num_frames=1)
    
    # Добавляем пользователей
    for i in range(1, 100):
        x = np.random.randint(400, 600)
        y = np.random.randint(400, 600)
        ue = UserEquipment(UE_ID=i, x=x, y=y)
        
        # Важно: инициализируем буфер пользователя
        ue.buffer.ADD_PACKET(10000, 0, 5)  # Добавляем большой пакет, чтобы пользователь был активен
        
        env.add_user(ue)
    
    # Запускаем симуляцию
    env.run_simulation(duration_ms=10)
    
    # Визуализируем результаты
    env.visualize_resource_grid(tti_start=0, tti_end=20)
    env.visualize_throughput()
    env.visualize_resource_allocation()
    env.print_statistics()

if __name__ == "__main__":
    test_round_robin()
