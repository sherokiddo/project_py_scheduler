"""
ИСПРАВЛЕННЫЙ ТЕСТ SHADOW FADING - БЕЗ ОШИБОК

Исправлены все проблемы из предыдущих тестов:
1. НЕ сбрасывается состояние канала каждый раз
2. Правильно отслеживается движение UE
3. Корректная логика корреляции
"""

import numpy as np
import matplotlib.pyplot as plt

class TestChannelState:
    def __init__(self):
        self.last_position = None
        self.last_shadow_fading = 0.0
        self.first_call = True

    def reset(self):
        self.__init__()

def calculate_correlated_shadow_fading(channel_state, current_position, 
                                     sigma_sf, correlation_distance):
    """
    Правильная реализация корреляционной модели shadow fading.
    """

    if channel_state.first_call or channel_state.last_position is None:
        # Первый вызов - инициализация
        channel_state.last_shadow_fading = np.random.normal(0, sigma_sf)
        channel_state.last_position = current_position.copy()
        channel_state.first_call = False
        print(f"Инициализация: SF = {channel_state.last_shadow_fading:.2f}")
        return channel_state.last_shadow_fading

    # Расчет расстояния от предыдущей позиции
    delta_d = np.linalg.norm(current_position - channel_state.last_position)

    # Коэффициент корреляции (модель Гудмундсона)
    R = np.exp(-delta_d / correlation_distance)

    # Новое коррелированное значение
    new_sf = (R * channel_state.last_shadow_fading + 
             np.sqrt(1 - R**2) * np.random.normal(0, sigma_sf))

    # Отладочный вывод каждые 50 итераций
    if len(str(current_position)) % 50 == 0:
        print(f"Pos: {current_position}, Δd: {delta_d:.2f}м, R: {R:.3f}, "
              f"SF: {channel_state.last_shadow_fading:.2f} → {new_sf:.2f}")

    # Обновление состояния
    channel_state.last_shadow_fading = new_sf
    channel_state.last_position = current_position.copy()

    return new_sf

def test_corrected_shadow_fading():
    """
    ИСПРАВЛЕННЫЙ тест корреляционной модели shadow fading.
    """
    print("ИСПРАВЛЕННЫЙ ТЕСТ КОРРЕЛЯЦИОННОЙ МОДЕЛИ")
    print("="*50)

    # Параметры модели UMi
    sf_correlation_distance_los = 10.0  # метры для UMi LOS
    sf_correlation_distance_nlos = 13.0  # метры для UMi NLOS  
    sigma_sf_los = 4.0  # стандартное отклонение для LOS
    sigma_sf_nlos = 7.82  # стандартное отклонение для NLOS

    # ОДНО состояние канала для всего теста (НЕ сбрасывается!)
    channel_state_los = TestChannelState()
    channel_state_nlos = TestChannelState()

    # Параметры теста
    num_points = 200  # уменьшено для наглядности
    movement_step = 1.0  # метр за шаг

    # Массивы результатов
    shadow_fading_los = []
    shadow_fading_nlos = []
    correlation_coeff_los = []
    correlation_coeff_nlos = []
    distances = []
    delta_d_values = []

    print(f"Параметры: {num_points} точек, шаг {movement_step}м")
    print(f"LOS: σ={sigma_sf_los} дБ, d_corr={sf_correlation_distance_los}м")
    print(f"NLOS: σ={sigma_sf_nlos} дБ, d_corr={sf_correlation_distance_nlos}м")
    print()

    # Начальная позиция
    start_x, start_y = 0.0, 0.0

    for i in range(num_points):
        # Текущая позиция (движение по диагонали)
        current_x = start_x + i * movement_step
        current_y = start_y + i * movement_step * 0.3  # небольшой угол
        current_pos = np.array([current_x, current_y])

        # Расчет расстояния движения
        if i > 0:
            prev_pos = np.array([distances[i-1][0], distances[i-1][1]])
            delta_d = np.linalg.norm(current_pos - prev_pos)
        else:
            delta_d = 0.0

        delta_d_values.append(delta_d)

        # Тест для LOS условий (БЕЗ СБРОСА!)
        sf_los = calculate_correlated_shadow_fading(
            channel_state_los, current_pos, sigma_sf_los, 
            sf_correlation_distance_los
        )
        shadow_fading_los.append(sf_los)

        # Тест для NLOS условий (БЕЗ СБРОСА!) 
        sf_nlos = calculate_correlated_shadow_fading(
            channel_state_nlos, current_pos, sigma_sf_nlos,
            sf_correlation_distance_nlos
        )
        shadow_fading_nlos.append(sf_nlos)

        # Расчет коэффициентов корреляции
        R_los = np.exp(-delta_d / sf_correlation_distance_los) if delta_d > 0 else 1.0
        R_nlos = np.exp(-delta_d / sf_correlation_distance_nlos) if delta_d > 0 else 1.0

        correlation_coeff_los.append(R_los)
        correlation_coeff_nlos.append(R_nlos)

        distances.append([current_x, current_y])

    print("\n✅ Расчеты завершены!")

    # Создание графиков
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ИСПРАВЛЕННЫЙ Тест корреляционной модели Shadow Fading', fontsize=14)

    iterations = np.arange(num_points)

    # График 1: Shadow Fading
    ax1.plot(iterations, shadow_fading_los, 'b-', alpha=0.8, linewidth=1.5, label='LOS')
    ax1.plot(iterations, shadow_fading_nlos, 'r-', alpha=0.8, linewidth=1.5, label='NLOS')
    ax1.set_title('Shadow Fading (должен быть ПЛАВНЫМ!)')
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('SF (дБ)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Коэффициенты корреляции
    ax2.plot(iterations, correlation_coeff_los, 'b-', alpha=0.8, label='LOS R')
    ax2.plot(iterations, correlation_coeff_nlos, 'r-', alpha=0.8, label='NLOS R')
    ax2.set_title('Коэффициенты корреляции R (должны ИЗМЕНЯТЬСЯ!)')
    ax2.set_xlabel('Итерация')
    ax2.set_ylabel('R')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # График 3: Изменения Shadow Fading между точками
    sf_los_diff = np.abs(np.diff(shadow_fading_los))
    sf_nlos_diff = np.abs(np.diff(shadow_fading_nlos))

    ax3.plot(iterations[1:], sf_los_diff, 'b-', alpha=0.7, label='LOS изменения')
    ax3.plot(iterations[1:], sf_nlos_diff, 'r-', alpha=0.7, label='NLOS изменения')
    ax3.axhline(y=3, color='orange', linestyle='--', alpha=0.7, label='Норма (<3 дБ)')
    ax3.set_title('Изменения Shadow Fading между точками')
    ax3.set_xlabel('Итерация')
    ax3.set_ylabel('|ΔSF| (дБ)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # График 4: Расстояние движения
    ax4.plot(iterations, delta_d_values, 'g-', alpha=0.8, linewidth=1.5)
    ax4.set_title('Расстояние движения между точками')
    ax4.set_xlabel('Итерация')
    ax4.set_ylabel('Δd (метры)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('corrected_shadow_fading_test.png', dpi=300, bbox_inches='tight')
    print("✅ График сохранен: corrected_shadow_fading_test.png")

    # Детальный анализ
    print("\n" + "="*60)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*60)

    # Статистика LOS
    los_mean = np.mean(shadow_fading_los)
    los_std = np.std(shadow_fading_los)
    los_range = np.max(shadow_fading_los) - np.min(shadow_fading_los)
    los_smooth = np.mean(sf_los_diff)

    print(f"LOS Shadow Fading:")
    print(f"  Среднее: {los_mean:.2f} дБ")
    print(f"  СКО: {los_std:.2f} дБ")
    print(f"  Размах: {los_range:.2f} дБ")
    print(f"  Среднее изменение: {los_smooth:.2f} дБ")

    # Статистика NLOS  
    nlos_mean = np.mean(shadow_fading_nlos)
    nlos_std = np.std(shadow_fading_nlos)
    nlos_range = np.max(shadow_fading_nlos) - np.min(shadow_fading_nlos)
    nlos_smooth = np.mean(sf_nlos_diff)

    print(f"\nNLOS Shadow Fading:")
    print(f"  Среднее: {nlos_mean:.2f} дБ")
    print(f"  СКО: {nlos_std:.2f} дБ")
    print(f"  Размах: {nlos_range:.2f} дБ")
    print(f"  Среднее изменение: {nlos_smooth:.2f} дБ")

    # Анализ корреляции
    R_los_mean = np.mean(correlation_coeff_los)
    R_nlos_mean = np.mean(correlation_coeff_nlos)
    R_los_min = np.min(correlation_coeff_los)
    R_nlos_min = np.min(correlation_coeff_nlos)

    print(f"\nКорреляционные коэффициенты:")
    print(f"  LOS R (средний/мин): {R_los_mean:.3f} / {R_los_min:.3f}")
    print(f"  NLOS R (средний/мин): {R_nlos_mean:.3f} / {R_nlos_min:.3f}")

    # Проверка движения
    delta_d_mean = np.mean(delta_d_values[1:])  # исключаем первый 0
    print(f"\nДвижение:")
    print(f"  Среднее расстояние между точками: {delta_d_mean:.2f} м")

    # Финальная диагностика
    print("\n" + "="*40)
    print("ДИАГНОСТИКА КОРРЕЛЯЦИИ:")
    print("="*40)

    success_count = 0

    if los_smooth < 5 and nlos_smooth < 8:
        print("✅ Изменения плавные - корреляция работает!")
        success_count += 1
    else:
        print(f"❌ Слишком резкие изменения: LOS {los_smooth:.1f}, NLOS {nlos_smooth:.1f}")

    if R_los_min < 0.9 and R_nlos_min < 0.9:
        print("✅ Коэффициенты корреляции изменяются - движение есть!")
        success_count += 1
    else:
        print(f"❌ R всегда высокий: LOS min={R_los_min:.3f}, NLOS min={R_nlos_min:.3f}")

    if delta_d_mean > 0.5:
        print("✅ UE движется с нормальной скоростью!")
        success_count += 1
    else:
        print(f"❌ UE движется слишком медленно: {delta_d_mean:.2f} м/шаг")

    if los_range > 5 and nlos_range > 8:
        print("✅ Достаточная вариация shadow fading!")
        success_count += 1
    else:
        print(f"❌ Слишком мало вариации: LOS {los_range:.1f}, NLOS {nlos_range:.1f}")

    print(f"\n🏆 ОБЩАЯ ОЦЕНКА: {success_count}/4 тестов пройдено")

    if success_count >= 3:
        print("✅ КОРРЕЛЯЦИОННАЯ МОДЕЛЬ РАБОТАЕТ ПРАВИЛЬНО!")
    else:
        print("❌ КОРРЕЛЯЦИОННАЯ МОДЕЛЬ РАБОТАЕТ НЕПРАВИЛЬНО!")

    plt.show()

if __name__ == "__main__":
    test_corrected_shadow_fading()
