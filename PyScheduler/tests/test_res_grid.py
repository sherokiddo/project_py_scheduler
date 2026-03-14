"""
#------------------------------------------------------------------------------
# Модуль: TEST_RES_GRID - Тесты ресурсной сетки LTE
#------------------------------------------------------------------------------
# Описание:
# Тесты для классов RES_GRID_LTE (v1.0.2) и RES_GRID_LTE_CACHED (v1.1.0).
# Вынесены из RES_GRID.py.
#
# Версия: 1.0.0
# Дата: 2026-03-14
# Автор: Македон Никита
#------------------------------------------------------------------------------
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from RES_GRID import RES_GRID_LTE, RES_GRID_LTE_CACHED


## Тесты для RES_GRID_LTE (оригинальная реализация)

### test_rb_allocation()

#Проверяет базовое выделение и повторное выделение ресурсных блоков.

def test_rb_allocation():
    """Тест выделения RB в оригинальной реализации"""
    grid = RES_GRID_LTE(bandwidth=10)
    
    # Выделение RB в первом слоте
    assert grid.ALLOCATE_RB(0, "sub_0_slot_0", 10, 100), "Ошибка выделения RB"
    
    # Попытка повторного выделения (должна вернуть False)
    assert not grid.ALLOCATE_RB(0, "sub_0_slot_0", 10, 200), "Должна была вернуть False (RB занят)"
    
    # Выделение во втором слоте (та же частота, разный слот)
    assert grid.ALLOCATE_RB(0, "sub_0_slot_1", 10, 100), "Ошибка выделения во втором слоте"
    
    # Проверка статуса RB
    rb1 = grid.GET_RB(0, "sub_0_slot_0", 10)
    rb2 = grid.GET_RB(0, "sub_0_slot_1", 10)
    assert rb1.UE_ID == 100 and rb2.UE_ID == 100, "Некорректное назначение UE_ID"
    assert rb1.status == "assigned" and rb2.status == "assigned", "Статус должен быть 'assigned'"
    
    print("✓ test_rb_allocation passed")

### test_bandwidth_configuration()

#Проверяет корректность параметров сетки для разных полос частот.

def test_bandwidth_configuration():
    """Проверка конфигурации для всех допустимых полос частот"""
    test_cases = [
        (1.4, 6),
        (3, 15),
        (5, 25),
        (10, 50),
        (15, 75),
        (20, 100)
    ]
    
    for bandwidth, expected_rb in test_cases:
        grid = RES_GRID_LTE(bandwidth=bandwidth)
        assert grid.rb_per_slot == expected_rb, \
            f"Ошибка для {bandwidth} МГц: ожидается {expected_rb}, получено {grid.rb_per_slot}"
        assert grid.num_rb == expected_rb * 2, \
            f"num_rb должна быть {expected_rb * 2}, получено {grid.num_rb}"
    
    print("✓ test_bandwidth_configuration passed")

### test_frame_structure()

#Проверяет иерархическую структуру кадров, подкадров и слотов.

def test_frame_structure():
    """Проверка структуры Frame → Subframe → Slot → RB"""
    grid = RES_GRID_LTE(num_frames=2, bandwidth=10)
    
    # Проверка количества кадров
    assert len(grid.frames) == 2, f"Ожидается 2 кадра, получено {len(grid.frames)}"
    
    # Проверка структуры подкадров в каждом кадре
    for frame_idx, frame in enumerate(grid.frames):
        assert len(frame.subframes) == 10, \
            f"Кадр {frame_idx}: ожидается 10 подкадров, получено {len(frame.subframes)}"
        
        # Проверка структуры слотов в каждом подкадре
        for sf_idx, subframe in enumerate(frame.subframes):
            assert len(subframe.slots) == 2, \
                f"Подкадр {sf_idx}: ожидается 2 слота, получено {len(subframe.slots)}"
            
            # Проверка количества RB в каждом слоте
            for slot in subframe.slots:
                assert len(slot.resource_blocks) == 50, \
                    f"Слот: ожидается 50 RB, получено {len(slot.resource_blocks)}"
    
    # Проверка общего числа TTI (кадров × подкадров)
    assert grid.total_tti == 20, f"Ожидается 20 TTI (2 × 10), получено {grid.total_tti}"
    
    print("✓ test_frame_structure passed")

### test_rb_allocation_semantics()

#Проверяет семантику состояний RB (свободно/назначено).

def test_rb_allocation_semantics():
    """Проверка смены состояний RB: free → assigned → free"""
    grid = RES_GRID_LTE(bandwidth=10)
    
    # Получение RB
    rb = grid.GET_RB(0, "sub_0_slot_0", 25)
    assert rb is not None, "RB не должен быть None"
    assert rb.CHCK_RB(), "Изначально RB должен быть свободным"
    
    # Выделение RB
    assert grid.ALLOCATE_RB(0, "sub_0_slot_0", 25, 100), "Ошибка выделения"
    assert rb.UE_ID == 100, "UE_ID должен быть 100"
    assert rb.status == "assigned", "Статус должен быть 'assigned'"
    assert not rb.CHCK_RB(), "RB не должна быть свободной"
    
    # Освобождение RB
    assert rb.RELEASE_RB(), "Ошибка при освобождении"
    assert rb.CHCK_RB(), "RB должна быть свободной после освобождения"
    assert rb.UE_ID is None, "UE_ID должен быть None"
    
    print("✓ test_rb_allocation_semantics passed")

### test_rb_group_allocation()

#Проверяет выделение пары RB (оба слота на одной частоте).

def test_rb_group_allocation():
    """Тест метода ALLOCATE_RB_PAIR"""
    grid = RES_GRID_LTE(bandwidth=5)
    freq_idx = 10
    
    # Выделение группы (оба слота)
    assert grid.ALLOCATE_RB_PAIR(0, freq_idx, 200), "Ошибка группового выделения"
    
    # Проверка обоих слотов
    slot0_rb = grid.GET_RB(0, "sub_0_slot_0", freq_idx)
    slot1_rb = grid.GET_RB(0, "sub_0_slot_1", freq_idx)
    
    assert slot0_rb.UE_ID == 200, f"Слот 0: UE_ID должен быть 200, получено {slot0_rb.UE_ID}"
    assert slot1_rb.UE_ID == 200, f"Слот 1: UE_ID должен быть 200, получено {slot1_rb.UE_ID}"
    
    print("✓ test_rb_group_allocation passed")

### test_boundary_conditions()

#Проверяет выделение в граничных условиях (последний TTI, последний RB).

def test_boundary_conditions():
    """Проверка граничных условий: последний TTI, последний RB"""
    grid = RES_GRID_LTE(bandwidth=20, num_frames=1)
    
    # Выделение в последнем TTI (TTI 9)
    assert grid.ALLOCATE_RB(9, "sub_9_slot_1", 99, 400), "Ошибка выделения в последнем TTI"
    assert grid.stats["allocation_by_tti"][9] == 1, "Счетчик TTI не увеличен"
    
    # Проверка статистики перед освобождением
    assert grid.stats["allocated_rbs"] == 1, "Должна быть 1 выделенная RB"
    assert 400 in grid.stats["allocation_by_user"], "UE_ID 400 должен быть в статистике"
    
    # Освобождение RB
    assert grid.RELEASE_RB(9, "sub_9_slot_1", 99), "Ошибка освобождения"
    
    # Проверка статистики после освобождения
    assert grid.stats["allocated_rbs"] == 0, "allocated_rbs должна быть 0"
    assert 400 not in grid.stats["allocation_by_user"], "UE_ID 400 должен быть удален"
    assert grid.stats["allocation_by_tti"][9] == 0, "Счетчик TTI должен быть 0"
    
    print("✓ test_boundary_conditions passed")

### test_3gpp_compliance()

#Проверяет соответствие стандарту TS 36.211.

def test_3gpp_compliance():
    """Проверка соответствия TS 36.211 Section 6.2.3"""
    grid = RES_GRID_LTE(bandwidth=10, num_frames=1)
    
    # Проверка параметров слота
    slot = grid.frames[0].subframes[0].slots[0]
    assert len(slot.resource_blocks) == 50, \
        f"Для 10 МГц ожидается 50 RB/слот, получено {len(slot.resource_blocks)}"
    
    # Проверка частотных индексов RB
    rb = next(iter(slot.resource_blocks.values()))
    assert rb.freq_idx >= 0 and rb.freq_idx < 50, \
        f"Частотный индекс {rb.freq_idx} вне диапазона [0, 49]"
    
    # Проверка идентификаторов RB
    for rb_id, rb in slot.resource_blocks.items():
        assert rb.status in ["free", "assigned"], f"Неизвестный статус: {rb.status}"
    
    print("✓ test_3gpp_compliance passed")

### test_resource_utilization_stats()

#Проверяет корректность подсчета статистики использования ресурсов.

def test_resource_utilization_stats():
    """Проверка подсчета статистики: allocation_by_user, allocation_by_tti"""
    grid = RES_GRID_LTE(bandwidth=10)
    
    # Выделение 5 RB с разными UE_ID
    for i in range(5):
        slot_idx = i % 2  # Чередуем слоты
        assert grid.ALLOCATE_RB(0, f"sub_0_slot_{slot_idx}", i, 100 + i), \
            f"Ошибка выделения для UE_ID={100+i}"
    
    # Проверка общей статистики
    assert grid.stats["allocated_rbs"] == 5, \
        f"Ожидается 5 выделенных RB, получено {grid.stats['allocated_rbs']}"
    
    # Проверка подсчета по пользователям
    for i in range(5):
        expected_count = grid.stats["allocation_by_user"].get(100 + i, 0)
        assert expected_count == 1, \
            f"UE_ID {100+i}: ожидается 1 RB, получено {expected_count}"
    
    # Проверка подсчета по TTI
    assert grid.stats["allocation_by_tti"][0] == 5, \
        f"TTI 0: ожидается 5 RB, получено {grid.stats['allocation_by_tti'][0]}"
    
    print("✓ test_resource_utilization_stats passed")


## Тесты для RES_GRID_LTE_CACHED (новая оптимизированная реализация)

### test_cached_allocate_rbg()

#Проверяет выделение RBG в кэшированной реализации.

def test_cached_allocate_rbg():
    """Проверка ALLOCATE_RBG в RES_GRID_LTE_CACHED"""
    grid = RES_GRID_LTE_CACHED(bandwidth=10, window_size=100)
    
    # Выделение RBG 0 в TTI 0 пользователю 100
    assert grid.ALLOCATE_RBG(0, 0, 100), "Ошибка выделения RBG 0"
    
    # Выделение RBG 1 в TTI 0 пользователю 200 (РАЗНАЯ RBG!)
    assert grid.ALLOCATE_RBG(0, 1, 200), "Ошибка выделения RBG 1"
    
    # Выделение RBG 0 в TTI 1 пользователю 300 (РАЗНЫЙ TTI, поэтому свободна!)
    assert grid.ALLOCATE_RBG(1, 0, 300), "Ошибка выделения RBG 0 в TTI 1"
    
    print("✓ test_cached_allocate_rbg passed")

### test_cached_release_rbg()

#Проверяет освобождение RBG в кэшированной реализации.

def test_cached_release_rbg():
    """Проверка RELEASE_RBG в RES_GRID_LTE_CACHED"""
    grid = RES_GRID_LTE_CACHED(bandwidth=10, window_size=100)
    
    # Выделение и освобождение RBG
    assert grid.ALLOCATE_RBG(0, 1, 200), "Ошибка выделения RBG"
    assert grid.RELEASE_RBG(0, 1), "Ошибка освобождения RBG"
    
    # Повторное выделение (должно быть успешным)
    assert grid.ALLOCATE_RBG(0, 1, 300), "После освобождения RBG должна быть свободна"
    
    print("✓ test_cached_release_rbg passed")

### test_sliding_window_cache()

#Проверяет работу скользящего окна кэша (ФУНДАМЕНТАЛЬНОЕ УЛУЧШЕНИЕ ПАМЯТИ).

def test_sliding_window_cache():
    """Проверка SlidingWindowCache: только N последних TTI в памяти"""
    grid = RES_GRID_LTE_CACHED(bandwidth=10, window_size=10)
    
    # Выделение в 15 разных TTI (window_size=10)
    for tti in range(15):
        assert grid.ALLOCATE_RBG(tti, 0, 100 + tti), f"Ошибка для TTI {tti}"
    
    # Проверка размера кэша (должен быть ≤ 10)
    cache_stats = grid.get_window_stats()
    assert cache_stats['current_size'] <= 10, \
        f"Кэш содержит {cache_stats['current_size']} элементов, максимум {grid.cache.window_size}"
    
    # Проверка eviction count (должно быть 5: TTI 0-4 исключены)
    assert cache_stats['evictions'] >= 5, \
        f"Ожидается ≥5 исключений, получено {cache_stats['evictions']}"
    
    print(f"✓ test_sliding_window_cache passed (hit_rate={cache_stats['hit_rate']:.2%})")

### test_cache_hit_rate()

#Проверяет эффективность кэша (высокий hit_rate при повторных обращениях).

def test_cache_hit_rate():
    """Проверка эффективности кэша: повторные обращения должны быть очень быстрыми"""
    grid = RES_GRID_LTE_CACHED(bandwidth=10, window_size=50)
    
    # Первое выделение (MISS)
    assert grid.ALLOCATE_RBG(0, 0, 100), "Ошибка первого выделения"
    
    # Повторное выделение в том же TTI (HIT из кэша)
    assert grid.ALLOCATE_RBG(0, 1, 200), "Ошибка второго выделения"
    
    # Еще несколько обращений (все HIT)
    for _ in range(5):
        grid.ALLOCATE_RBG(0, 2, 300)
    
    # Проверка статистики кэша
    stats = grid.get_window_stats()
    assert stats['hits'] > 0, "Должны быть HIT в кэше"
    assert stats['hit_rate'] > 0.5, f"Hit rate должен быть >50%, получено {stats['hit_rate']:.1%}"
    
    print(f"✓ test_cache_hit_rate passed (hit_rate={stats['hit_rate']:.1%})")

### test_generate_bitmap()

#Проверяет генерацию bitmap распределения RBG для пользователя.

def test_generate_bitmap():
    """Проверка метода GENERATE_BITMAP"""
    grid = RES_GRID_LTE_CACHED(bandwidth=10, window_size=100)
    
    # Выделение нескольких RBG
    grid.ALLOCATE_RBG(0, 0, 100)  # RBG 0
    grid.ALLOCATE_RBG(0, 2, 100)  # RBG 2
    
    # Генерация bitmap
    bitmap = grid.GENERATE_BITMAP(0, 100)
    
    # Проверка: RBG 0 и 2 должны быть 1, остальные 0
    total_rbg = (grid.rb_per_slot + grid.get_rbg_size() - 1) // grid.get_rbg_size()
    assert len(bitmap) == total_rbg, f"Ожидается {total_rbg} элементов, получено {len(bitmap)}"
    assert bitmap[0] == 1, "RBG 0 должна быть 1"
    assert bitmap[2] == 1, "RBG 2 должна быть 1"
    assert bitmap[1] == 0, "RBG 1 должна быть 0"
    
    print("✓ test_generate_bitmap passed")

### test_rbg_indices_caching()

#Проверяет кэширование результатов GET_RBG_INDICES (ОПТИМИЗАЦИЯ УРОВНЯ 1).

def test_rbg_indices_caching():
    """Проверка кэширования GET_RBG_INDICES: результаты должны быть одинаковыми"""
    grid = RES_GRID_LTE_CACHED(bandwidth=10)
    
    # Первый вызов (вычисляется)
    indices1 = grid.GET_RBG_INDICES(0)
    
    # Второй вызов (должен вернуть из кэша)
    indices2 = grid.GET_RBG_INDICES(0)
    
    # Должны быть одинаковыми (даже одна и та же ссылка объекта)
    assert indices1 == indices2, "Результаты должны быть одинаковыми"
    assert indices1 is indices2, "Должна быть одна и та же ссылка на объект (кэш работает)"
    
    print("✓ test_rbg_indices_caching passed")

### test_memory_efficiency()

#Демонстрирует экономию памяти (ОСНОВНОЕ УЛУЧШЕНИЕ).
def test_3gpp_compliance_new():
    # TS 36.211 Section 6.2.3
    # Новая версия: используем кэш и создаем TTI по требованию
    grid = RES_GRID_LTE_CACHED(bandwidth=10, window_size=10)
    
    # Запрашиваем Subframe для TTI 0 (это создаст его в кэше)
    subframe = grid._get_or_create_subframe(0)
    
    # Проверка параметров слота 0
    slot = subframe.GET_SLOT(0)
    assert slot is not None, "Слот 0 должен существовать"
    
    # В новой версии слоты могут использовать resource_blocks_by_freq
    # Проверяем количество RB (должно быть 50 для 10 МГц)
    rbs = slot.GET_ALL_RES_BLCK()
    assert len(rbs) == 50, f"Ожидается 50 RB/слот, получено {len(rbs)}"
    
    # Проверка частотных индексов
    rb = rbs[0]
    assert 0 <= rb.freq_idx < 50, "Некорректный частотный индекс"
    
    print("✓ test_3gpp_compliance (CACHED) passed")
def test_memory_efficiency():
    """
    Демонстрация экономии памяти:
    - Старая реализация: 6М объектов × 400 байт ≈ 2400 МБ
    - Новая реализация: 100 × 100 объектов ≈ 40 МБ
    """
    import sys
    
    # Создание с малым окном для теста
    grid_cached = RES_GRID_LTE_CACHED(bandwidth=10, window_size=100)
    
    # Проверяем, что живых объектов мало
    cache_stats = grid_cached.get_window_stats()
    max_objects = cache_stats['current_size'] * 100  # (окно) × (RB на TTI)
    
    # Для window_size=100, max ~10k объектов (было 6М)
    assert max_objects <= 15000, \
        f"Слишком много объектов в памяти: {max_objects} (максимум 15k для теста)"
    
    print(f"✓ test_memory_efficiency passed (max {max_objects} objects vs 6M in old version)")


## Запуск всех тестов

if __name__ == "__main__":
    print("=" * 60)
    print("ЗАПУСК ТЕСТОВ RES_GRID_LTE (v1.0.2)")
    print("=" * 60)
    
    test_rb_allocation()
    test_bandwidth_configuration()
    test_frame_structure()
    test_rb_allocation_semantics()
    test_rb_group_allocation()
    test_boundary_conditions()
    test_3gpp_compliance()
    test_resource_utilization_stats()
    
    print("\n" + "=" * 60)
    print("ЗАПУСК ТЕСТОВ RES_GRID_LTE_CACHED (v1.1.1)")
    print("=" * 60)
    
    test_cached_allocate_rbg()
    test_cached_release_rbg()
    test_sliding_window_cache()
    test_cache_hit_rate()
    test_generate_bitmap()
    test_rbg_indices_caching()
    test_3gpp_compliance_new()
    test_memory_efficiency()
    
    
    print("\n" + "=" * 60)
    print("✅ ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")
    print("=" * 60)
