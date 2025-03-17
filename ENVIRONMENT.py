from LTE_GRID_ver_alpha import RES_GRID_LTE
from SCHEDULER import RoundRobinScheduler, EnhancedRoundRobinScheduler
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
#upd

def test_round_robin_scheduler():
    # Создаем ресурсную сетку: 3 МГц (15 RB), 1 кадр (10 TTI)
    lte_grid = RES_GRID_LTE(bandwidth=3, num_frames=1)
    
    # Создаем планировщик Round Robin
    scheduler = RoundRobinScheduler(lte_grid)
    
    # Добавляем 5 пользователей
    for user_id in range(1, 6):
        scheduler.add_user(user_id)
    
    print("Запуск планировщика Round Robin для 5 пользователей...")
    
    # Планируем ресурсы для всех TTI
    total_allocated = scheduler.schedule_all_ttis()
    
    print(f"Всего назначено {total_allocated} ресурсных блоков.")
    
    # Визуализация результатов работы планировщика
    print("Ресурсная сетка после работы планировщика Round Robin:")
    visualize_grid(lte_grid)
    
    return lte_grid, scheduler

def test_enhanced_round_robin_scheduler():
    # Создаем ресурсную сетку: 10 МГц (50 RB), 1 кадр (10 TTI)
    lte_grid = RES_GRID_LTE(bandwidth=10, num_frames=1)
    
    # Создаем улучшенный планировщик Round Robin
    scheduler = EnhancedRoundRobinScheduler(lte_grid, min_rb_per_user=3, max_rb_per_user=7)
    
    # Добавляем 8 пользователей
    for user_id in range(1, 9):
        scheduler.add_user(user_id)
    
    print(f"Запуск улучшенного планировщика Round Robin для 8 пользователей...")
    print(f"Настройки: мин. RB на пользователя = {scheduler.min_rb_per_user}, "
          f"макс. RB на пользователя = {scheduler.max_rb_per_user}")
    
    # Планируем ресурсы для всех TTI
    total_allocated = scheduler.schedule_all_ttis()
    
    print(f"Всего назначено {total_allocated} ресурсных блоков.")
    
    # Визуализация результатов работы планировщика
    print("Ресурсная сетка после работы улучшенного планировщика Round Robin:")
    visualize_grid(lte_grid)
    
    return lte_grid, scheduler

def visualize_grid(lte_grid, tti_start: int = 0, tti_end: Optional[int] = None,
                       show_UE_IDs: bool = True, save_path: Optional[str] = None):
    """
    Визуализация ресурсной сетки.
    Args:
        lte_grid: Обьект класса RES_GRID_LTE
        tti_start: Начальный TTI для визуализации
        tti_end: Конечный TTI для визуализации (не включительно)
        show_UE_IDs: Показывать ли ID пользователей на графике
        save_path: Путь для сохранения изображения (если None, то изображение отображается)
    """
    if tti_end is None:
        tti_end = min(tti_start + 10, lte_grid.total_tti)
    
    # Создаем матрицу для отображения статуса RB
    # 0 для свободных, >0 для занятых блоков (значение = UE_ID)
    grid = np.zeros((lte_grid.num_rb, tti_end - tti_start))
    
    for tti in range(tti_start, tti_end):
        for freq_idx in range(lte_grid.num_rb):
            rb = lte_grid.GET_RB(tti, freq_idx)
            if rb:
                if rb.CHCK_RB():
                    grid[freq_idx, tti - tti_start] = 0
                else:
                    # Для визуализации используем UE_ID как цвет
                    grid[freq_idx, tti - tti_start] = rb.UE_ID if rb.UE_ID is not None else 0
    
    plt.figure(figsize=(15, 10))
    
    # Создаем маску для свободных блоков
    mask_free = (grid == 0)
    
    # Создаем кастомную цветовую карту
    cmap = plt.cm.jet
    cmap.set_bad('white', 1.0)  # Свободные блоки будут белыми
    
    # Создаем матрицу с NaN для свободных блоков
    grid_masked = np.ma.array(grid, mask=mask_free)
    
    # Отображаем матрицу
    plt.imshow(grid_masked, aspect='auto', cmap=cmap, interpolation='nearest')
    plt.colorbar(label='User ID')
    plt.xlabel('TTI')
    plt.ylabel('Frequency (RB index)')
    plt.title(f'LTE Resource Grid (Bandwidth: {lte_grid.bandwidth} MHz, TTI: {tti_start}-{tti_end-1})')
    
    # Настройка осей
    plt.yticks(np.arange(0, lte_grid.num_rb, 5))
    plt.xticks(np.arange(0, tti_end - tti_start, 1), np.arange(tti_start, tti_end, 1))
    plt.gca().invert_yaxis()
    
    # Добавление линий для разделения кадров
    for i in range(tti_start, tti_end, 10):
        rel_i = i - tti_start
        if 0 <= rel_i < (tti_end - tti_start):
            plt.axvline(x=rel_i, color='red', linestyle='-', linewidth=1, label='Frame boundary' if i == tti_start else None)
    
    # Добавление линий для разделения подкадров
    for i in range(tti_start, tti_end):
        rel_i = i - tti_start
        if 0 <= rel_i < (tti_end - tti_start):
            plt.axvline(x=rel_i, color='gray', linestyle='--', linewidth=0.5)
    
    # Добавление текста с ID пользователей, если требуется
    if show_UE_IDs:
        for tti in range(tti_start, tti_end):
            for freq_idx in range(lte_grid.num_rb):
                rb = lte_grid.GET_RB(tti, freq_idx)
                if rb and not rb.CHCK_RB():
                    plt.text(tti - tti_start, freq_idx, str(rb.UE_ID),
                             ha='center', va='center', color='black', fontsize=8)
    
    plt.grid(True, color='black', linestyle='-', linewidth=0.2)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Тестирование базового планировщика Round Robin
    grid1, scheduler1 = test_round_robin_scheduler()
    
    # Тестирование улучшенного планировщика Round Robin
    grid2, scheduler2 = test_enhanced_round_robin_scheduler()
    
import matplotlib.animation as animation

def animate_allocation(lte_grid, scheduler, num_frames, save_path=None):
    """
    Создает анимацию выделения ресурсов.

    Args:
        lte_grid: Объект ресурсной сетки LTE.
        scheduler: Объект планировщика.
        num_frames: Количество кадров анимации.
        save_path: Путь для сохранения анимации (если None, анимация отображается).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = None

    def init():
        """Инициализация графика."""
        nonlocal im
        grid = np.zeros((lte_grid.num_rb, 1))
        cmap = plt.cm.jet
        cmap.set_bad('white', 1.0)
        im = ax.imshow(grid, aspect='auto', cmap=cmap, interpolation='nearest')
        ax.invert_yaxis()
        ax.set_xlabel('TTI')
        ax.set_ylabel('Frequency (RB index)')
        ax.set_title('LTE Resource Grid Animation')
        ax.set_yticks(np.arange(0, lte_grid.num_rb, 5))
        ax.set_xticks([0])
        return [im]

    def update(frame):
        """Обновляет состояние графика для каждого кадра."""
        nonlocal im
        tti = frame % lte_grid.total_tti
        grid = np.zeros((lte_grid.num_rb, 1))

        for freq_idx in range(lte_grid.num_rb):
            rb = lte_grid.GET_RB(tti, freq_idx)
            if rb and not rb.CHCK_RB():
                grid[freq_idx, 0] = rb.UE_ID

        # Обновляем данные изображения
        im.set_array(grid)
        ax.set_title(f'LTE Resource Grid (TTI: {tti})')
        return [im]

    # Создаем анимацию
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, repeat=False)

    # Сохраняем анимацию в файл (если указано)
    if save_path:
        ani.save(save_path, writer='ffmpeg', fps=10)
    else:
        plt.show()

    plt.close(fig)
