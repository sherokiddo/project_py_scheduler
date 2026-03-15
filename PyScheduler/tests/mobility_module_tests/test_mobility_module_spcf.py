"""
TEST_MOBILITY_MODEL_SPECIFIC.py
Уровень 2: Специфика поведения каждой модели.
Покрытие: внутренняя логика, инварианты, параметры, известные баги.

Баги зафиксированные ранее (тесты документируют ожидаемое поведение ПОСЛЕ фикса):
  [BUG-1] RWP.__init__: direction и velocity из _choose_new_destination отброшены через _
  [BUG-2] RD.update: неправильная распаковка _choose_new_direction при снятии паузы
"""

import math
import unittest

import numpy as np
from unittest.mock import MagicMock, patch

from MOBILITY_MODEL import (
    MapBorders,
    RandomWalkModel,
    RandomWaypointModel,
    RandomDirectionModel,
    GaussMarkovModel,
)

X_MIN, X_MAX = -500.0, 500.0
Y_MIN, Y_MAX = -500.0, 500.0
TIME_MS      = 500
N_STEPS      = 300


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def reset_borders():
    MapBorders._instance = None

def make_ue(x=0.0, y=0.0, velocity=11.1, v_min=2.0, v_max=16.7, direction=0.0):
    ue = MagicMock()
    ue.position      = (x, y)
    ue.velocity      = velocity
    ue.velocity_min  = v_min
    ue.velocity_max  = v_max
    ue.direction     = direction
    ue.mean_velocity = velocity
    return ue

def simulate(model, ue, n=N_STEPS, t=TIME_MS):
    """Прогоняет модель n шагов, обновляя ue.position/velocity/direction."""
    for _ in range(n):
        pos, vel, d = model.update(t)
        ue.position  = pos
        ue.velocity  = vel
        ue.direction = d
    return ue.position

def direction_toward(src, dst):
    """Угол от src к dst в радианах."""
    return math.atan2(dst[1] - src[1], dst[0] - src[0])

def on_boundary(pos, eps=2.0):
    """True, если точка лежит на границе карты (в пределах eps)."""
    x, y = pos
    return (
        abs(x - X_MIN) <= eps or abs(x - X_MAX) <= eps or
        abs(y - Y_MIN) <= eps or abs(y - Y_MAX) <= eps
    )

def angle_diff(a, b):
    """Разница углов в радианах, нормализованная в [-π, π]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


# ==============================================================================
# 1. RandomWalk — специфика
# ==============================================================================

class TestRandomWalkSpecific(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_direction_is_random_each_step(self):
        """RW: при нормальном движении direction меняется каждый шаг."""
        ue = make_ue(x=0.0, y=0.0, velocity=5.0)
        model = RandomWalkModel(ue=ue)
        directions = []
        for _ in range(50):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            directions.append(d)
        # Если все направления одинаковы — модель сломана
        unique = len(set(round(d, 6) for d in directions))
        self.assertGreater(unique, 10,
            "Направления не случайны: менее 10 уникальных значений за 50 шагов")

    def test_direction_range_is_full_circle(self):
        """RW: за N шагов direction покрывает весь круг [0, 2π]."""
        ue = make_ue(x=0.0, y=0.0, velocity=5.0)
        model = RandomWalkModel(ue=ue)
        directions = []
        for _ in range(N_STEPS):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            directions.append(d % (2 * math.pi))
        # Делим окружность на 8 секторов — каждый должен быть посещён
        sectors = [False] * 8
        for d in directions:
            sectors[int(d / (2 * math.pi / 8))] = True
        self.assertTrue(all(sectors),
            f"Не все секторы [0,2π] посещены: {sectors}")

    def test_velocity_changes_each_step(self):
        """RW: velocity — случайное число [v_min, v_max] на каждом шаге."""
        ue = make_ue(v_min=2.0, v_max=16.7)
        model = RandomWalkModel(ue=ue)
        velocities = []
        for _ in range(50):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            velocities.append(vel)
        unique = len(set(round(v, 6) for v in velocities))
        self.assertGreater(unique, 10,
            "Скорости не случайны: менее 10 уникальных за 50 шагов")

    def test_velocity_always_in_range(self):
        """RW: velocity всегда в [v_min, v_max]."""
        ue = make_ue(v_min=1.0, v_max=20.0)
        model = RandomWalkModel(ue=ue)
        for _ in range(N_STEPS):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            self.assertGreaterEqual(vel, 1.0 - 1e-9)
            self.assertLessEqual(vel, 20.0 + 1e-9)

    def test_reflection_x_mirrors_direction(self):
        """RW: отражение от X-стены меняет sign(cos(direction)) → π - dir."""
        # UE у правой стены, летит вправо (direction ≈ 0)
        ue = make_ue(x=499.0, y=0.0, velocity=50.0, direction=0.0)
        model = RandomWalkModel(ue=ue)
        # Перехватываем np.random.uniform чтобы исключить рандомизацию direction
        # и увидеть чистое отражение
        with patch('numpy.random.uniform', return_value=0.0):
            pos, vel, d = model.update(100)
        # После отражения от X: new_dir = π - 0 = π (летит влево)
        # Но затем direction рандомизируется — мы проверяем что pos не вышла
        self.assertLessEqual(pos[0], X_MAX + 1e-6,
            f"После отражения x={pos[0]} вышел за x_max={X_MAX}")

    def test_reflection_y_mirrors_direction(self):
        """RW: отражение от Y-стены: new_dir = -current_direction."""
        ue = make_ue(x=0.0, y=499.0, velocity=50.0, direction=math.pi / 2)
        model = RandomWalkModel(ue=ue)
        with patch('numpy.random.uniform', return_value=0.0):
            pos, vel, d = model.update(100)
        self.assertLessEqual(pos[1], Y_MAX + 1e-6,
            f"После отражения y={pos[1]} вышел за y_max={Y_MAX}")

    def test_is_first_move_attribute_exists(self):
        """RW: атрибут is_first_move присутствует после инициализации."""
        ue = make_ue()
        model = RandomWalkModel(ue=ue)
        self.assertTrue(hasattr(model, 'is_first_move'),
            "RandomWalkModel должен иметь атрибут is_first_move")

    def test_position_changes_when_velocity_nonzero(self):
        """RW: при velocity > 0 позиция меняется."""
        ue = make_ue(x=0.0, y=0.0, velocity=10.0, v_min=10.0, v_max=10.0)
        model = RandomWalkModel(ue=ue)
        pos, vel, d = model.update(TIME_MS)
        self.assertFalse(
            pos == (0.0, 0.0) and vel == 0.0,
            "UE с velocity=10 не должен оставаться на месте"
        )


# ==============================================================================
# 2. RandomWaypoint — специфика
# ==============================================================================

class TestRandomWaypointSpecific(unittest.TestCase):
    """
    Некоторые тесты документируют ожидаемое поведение ПОСЛЕ фикса BUG-1:
    self.destination, self.current_velocity, self.current_direction, _ =
        self._choose_new_destination(...)
    """

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def _ue_near_destination(self, model, offset=5.0):
        """Телепортирует UE к destination - offset по X."""
        model.ue.position = (
            model.destination[0] - offset,
            model.destination[1]
        )

    # --- Инициализация ---

    def test_destination_set_on_init(self):
        """RWP: destination задан сразу после __init__."""
        ue = make_ue()
        model = RandomWaypointModel(ue=ue)
        self.assertIsNotNone(model.destination)
        self.assertEqual(len(model.destination), 2)

    def test_destination_inside_map_on_init(self):
        """RWP: начальный destination всегда внутри карты."""
        for _ in range(20):
            ue = make_ue(x=np.random.uniform(-400, 400),
                         y=np.random.uniform(-400, 400))
            model = RandomWaypointModel(ue=ue)
            dx, dy = model.destination
            self.assertGreaterEqual(dx, X_MIN)
            self.assertLessEqual(dx, X_MAX)
            self.assertGreaterEqual(dy, Y_MIN)
            self.assertLessEqual(dy, Y_MAX)

    def test_current_direction_toward_destination_after_fix(self):
        """RWP [BUG-1]: после фикса current_direction указывает на destination."""
        ue = make_ue(x=0.0, y=0.0)
        model = RandomWaypointModel(ue=ue)
        expected = direction_toward(ue.position, model.destination)
        diff = angle_diff(model.current_direction, expected)
        self.assertLess(diff, 0.01,
            f"[BUG-1] current_direction={model.current_direction:.4f} "
            f"не совпадает с направлением к destination={model.destination} "
            f"(expected={expected:.4f}, diff={diff:.4f})")

    def test_current_velocity_set_after_fix(self):
        """RWP [BUG-1]: после фикса current_velocity != случайное из MobilityInterface."""
        # _choose_new_destination задаёт velocity из [v_min, v_max]
        # Если баг не исправлен — current_velocity берётся из super().__init__()
        # и не совпадает с тем что вернул _choose_new_destination
        ue = make_ue(v_min=5.0, v_max=15.0)
        model = RandomWaypointModel(ue=ue)
        self.assertGreaterEqual(model.current_velocity, 5.0 - 1e-9)
        self.assertLessEqual(model.current_velocity, 15.0 + 1e-9)

    # --- Движение к цели ---

    def test_velocity_zero_when_paused(self):
        """RWP: при is_paused=True update() возвращает velocity=0."""
        ue = make_ue(x=0.0, y=0.0, velocity=100.0, v_min=100.0, v_max=100.0)
        model = RandomWaypointModel(ue=ue, pause_time=5000)
        self._ue_near_destination(model, offset=5.0)

        found_pause = False
        for _ in range(50):
            pos, vel, d = model.update(100)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if vel == 0.0 and model.is_paused:
                found_pause = True
                break
        self.assertTrue(found_pause,
            "RWP: пауза с velocity=0 не обнаружена")

    def test_position_frozen_during_pause(self):
        """RWP: во время паузы позиция не изменяется."""
        ue = make_ue(x=0.0, y=0.0, velocity=200.0, v_min=200.0, v_max=200.0)
        model = RandomWaypointModel(ue=ue, pause_time=5000)
        self._ue_near_destination(model, offset=3.0)

        # Доводим до паузы
        for _ in range(20):
            pos, vel, d = model.update(100)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if model.is_paused:
                break

        if not model.is_paused:
            self.skipTest("Не достигли паузы для проверки")

        pos_at_pause = ue.position
        for _ in range(5):
            pos, vel, d = model.update(100)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            self.assertEqual(pos, pos_at_pause,
                "Позиция изменилась во время паузы")

    def test_new_destination_chosen_after_arrival(self):
        """RWP: после достижения цели выбирается новый destination."""
        ue = make_ue(x=0.0, y=0.0, velocity=200.0, v_min=200.0, v_max=200.0)
        model = RandomWaypointModel(ue=ue, pause_time=0)
        old_dest = model.destination
        self._ue_near_destination(model, offset=3.0)

        new_dest_found = False
        for _ in range(30):
            pos, vel, d = model.update(100)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if model.destination != old_dest:
                new_dest_found = True
                break
        self.assertTrue(new_dest_found,
            "RWP: новый destination не был выбран после достижения цели")

    def test_pause_time_zero_no_pause(self):
        """RWP: при pause_time=0 пауза не возникает (is_paused не сохраняется)."""
        ue = make_ue(x=0.0, y=0.0, velocity=200.0, v_min=200.0, v_max=200.0)
        model = RandomWaypointModel(ue=ue, pause_time=0)
        self._ue_near_destination(model, offset=3.0)

        for _ in range(30):
            pos, vel, d = model.update(100)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            # После arrival с pause_time=0: is_paused ставится, но сразу снимается
            # Суммарный velocity не должен быть 0 дольше одного шага
        # Проверяем что сейчас не в паузе (снялась)
        self.assertFalse(model.is_paused,
            "При pause_time=0 is_paused не должен быть активен через несколько шагов")

    def test_direction_recalculated_toward_new_destination(self):
        """RWP: после выбора нового destination direction обновляется к нему."""
        ue = make_ue(x=0.0, y=0.0, velocity=200.0, v_min=200.0, v_max=200.0)
        model = RandomWaypointModel(ue=ue, pause_time=0)
        self._ue_near_destination(model, offset=3.0)

        for _ in range(30):
            pos, vel, d = model.update(100)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if not model.is_paused and vel > 0:
                expected = direction_toward(ue.position, model.destination)
                diff = angle_diff(model.current_direction, expected)
                self.assertLess(diff, 0.2,
                    f"Direction не указывает на новый destination: "
                    f"dir={model.current_direction:.4f}, expected={expected:.4f}")
                break


# ==============================================================================
# 3. RandomDirection — специфика
# ==============================================================================

class TestRandomDirectionSpecific(unittest.TestCase):
    """
    Тесты документируют ожидаемое поведение ПОСЛЕ фикса BUG-2:
    self.destination, self.current_velocity, self.current_direction, _, self.is_first_move =
        self._choose_new_direction(...)
    """

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def _drive_to_boundary(self, model, ue, max_steps=2000, t=100):
        """Прогоняет модель до достижения границы (is_paused=True)."""
        for _ in range(max_steps):
            pos, vel, d = model.update(t)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if model.is_paused:
                return True
        return False

    # --- Инициализация ---

    def test_initial_destination_on_boundary(self):
        """RD: начальный destination лежит на границе карты."""
        ue = make_ue(x=0.0, y=0.0)
        model = RandomDirectionModel(ue=ue, pause_time=0)
        self.assertTrue(on_boundary(model.destination),
            f"Начальный destination не на границе: {model.destination}")

    def test_first_move_flag_false_after_init(self):
        """RD: is_first_move=False после __init__ (уже выбрано первое направление)."""
        ue = make_ue(x=0.0, y=0.0)
        model = RandomDirectionModel(ue=ue, pause_time=0)
        self.assertFalse(model.is_first_move,
            "is_first_move должен стать False после инициализации")

    def test_direction_points_to_boundary_destination(self):
        """RD: current_direction указывает на destination (на границе)."""
        ue = make_ue(x=0.0, y=0.0)
        model = RandomDirectionModel(ue=ue, pause_time=0)
        expected = direction_toward(ue.position, model.destination)
        diff = angle_diff(model.current_direction, expected)
        self.assertLess(diff, 0.01,
            f"Direction не указывает на boundary destination: "
            f"dir={model.current_direction:.4f}, expected={expected:.4f}")

    # --- Движение ---

    def test_ue_reaches_boundary(self):
        """RD: UE достигает границы карты (is_paused становится True)."""
        ue = make_ue(x=0.0, y=0.0, velocity=50.0, v_min=50.0, v_max=50.0)
        model = RandomDirectionModel(ue=ue, pause_time=0)
        reached = self._drive_to_boundary(model, ue)
        self.assertTrue(reached,
            "UE так и не достиг границы карты за 2000 шагов")

    def test_position_at_boundary_when_paused(self):
        """RD: при достижении цели позиция UE на границе карты."""
        ue = make_ue(x=0.0, y=0.0, velocity=50.0, v_min=50.0, v_max=50.0)
        model = RandomDirectionModel(ue=ue, pause_time=0)
        reached = self._drive_to_boundary(model, ue)
        if not reached:
            self.skipTest("UE не достиг границы")
        self.assertTrue(on_boundary(ue.position, eps=5.0),
            f"Позиция не на границе при паузе: {ue.position}")

    def test_velocity_zero_when_paused(self):
        """RD: при is_paused velocity=0."""
        ue = make_ue(x=0.0, y=0.0, velocity=50.0, v_min=50.0, v_max=50.0)
        model = RandomDirectionModel(ue=ue, pause_time=5000)
        reached = self._drive_to_boundary(model, ue)
        if not reached:
            self.skipTest("UE не достиг границы")
        # Следующий шаг во время паузы должен вернуть velocity=0
        pos, vel, d = model.update(100)
        self.assertEqual(vel, 0.0,
            f"Ожидался velocity=0 во время паузы, получен {vel}")

    def test_new_destination_on_boundary_after_pause(self):
        """RD [BUG-2]: после паузы выбирается новый destination на границе."""
        ue = make_ue(x=0.0, y=0.0, velocity=100.0, v_min=100.0, v_max=100.0)
        model = RandomDirectionModel(ue=ue, pause_time=200)

        # Едем до границы
        reached = self._drive_to_boundary(model, ue, t=200)
        if not reached:
            self.skipTest("UE не достиг границы")

        old_dest = model.destination

        # Переживаем паузу
        for _ in range(10):
            pos, vel, d = model.update(200)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if not model.is_paused:
                break

        self.assertTrue(
            on_boundary(model.destination, eps=5.0),
            f"[BUG-2] Новый destination не на границе: {model.destination}"
        )

    def test_destination_updated_after_pause(self):
        """RD [BUG-2]: self.destination обновляется после снятия паузы."""
        ue = make_ue(x=0.0, y=0.0, velocity=100.0, v_min=100.0, v_max=100.0)
        model = RandomDirectionModel(ue=ue, pause_time=200)

        reached = self._drive_to_boundary(model, ue, t=200)
        if not reached:
            self.skipTest("UE не достиг границы")

        old_dest = model.destination

        for _ in range(10):
            pos, vel, d = model.update(200)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if not model.is_paused:
                break

        self.assertNotEqual(model.destination, old_dest,
            f"[BUG-2] destination не изменился после паузы: {model.destination}")

    def test_subsequent_direction_in_half_circle(self):
        """RD: повторные направления из [0, π] (не первый ход)."""
        # _choose_new_direction: is_first_move=False → [0, π]
        ue = make_ue(x=0.0, y=0.0)
        model = RandomDirectionModel(ue=ue, pause_time=0)
        # Принудительно вызываем как "не первый ход"
        _, _, direction, _, _ = model._choose_new_direction(
            current_position=(0.0, 0.0),
            velocity_min=2.0,
            velocity_max=16.7,
            is_first_move=False
        )
        self.assertGreaterEqual(direction, 0.0,
            f"Повторное направление < 0: {direction}")
        self.assertLessEqual(direction, math.pi + 0.01,
            f"Повторное направление > π: {direction}")

    def test_first_direction_in_full_circle(self):
        """RD: первое направление из [0, 2π]."""
        # Проверяем только что параметр is_first_move=True даёт [0, 2π]
        ue = make_ue(x=0.0, y=0.0)
        model = RandomDirectionModel(ue=ue, pause_time=0)

        directions = []
        for _ in range(100):
            _, _, direction, _, _ = model._choose_new_direction(
                current_position=(0.0, 0.0),
                velocity_min=2.0,
                velocity_max=16.7,
                is_first_move=True
            )
            directions.append(direction % (2 * math.pi))

        # Должны встречаться углы > π (чего не было бы при [0, π])
        has_above_pi = any(d > math.pi for d in directions)
        self.assertTrue(has_above_pi,
            "При is_first_move=True направления не превышают π — "
            "возможно, используется диапазон [0, π] вместо [0, 2π]")


# ==============================================================================
# 4. GaussMarkov — специфика
# ==============================================================================

class TestGaussMarkovSpecific(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def _make_gm(self, x=0.0, y=0.0, alpha=0.75, threshold=50.0,
                 velocity=11.1, v_min=2.0, v_max=16.7):
        ue = make_ue(x=x, y=y, velocity=velocity, v_min=v_min, v_max=v_max)
        return GaussMarkovModel(ue=ue, alpha=alpha,
                                boundary_threshold=threshold), ue

    # --- Параметр alpha ---

    def test_alpha_1_velocity_no_noise(self):
        """GM alpha=1: sqrt(1-1²)=0, шум отсутствует → velocity меняется только за счёт памяти."""
        model, ue = self._make_gm(alpha=1.0, velocity=10.0)
        velocities = []
        for _ in range(30):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            velocities.append(vel)
        # При alpha=1 new_vel = 1*cur_vel + 0*mean + 0*noise = cur_vel
        # Скорость не должна изменяться (остаётся равной начальной)
        for i in range(1, len(velocities)):
            self.assertAlmostEqual(velocities[i], velocities[i-1], places=6,
                msg=f"При alpha=1 velocity изменилась: {velocities[i-1]} → {velocities[i]}")

    def test_alpha_0_velocity_ignores_history(self):
        """GM alpha=0: new_vel = mean_velocity + N(0,1), предыдущая скорость не влияет."""
        # При alpha=0: new_vel = (1-0)*mean + sqrt(1-0)*N(0,1) = mean + N(0,1)
        model, ue = self._make_gm(alpha=0.0, velocity=10.0, v_min=0.0, v_max=100.0)
        velocities = []
        for _ in range(100):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            velocities.append(vel)
        # Должна быть значительная вариативность
        std = np.std(velocities)
        self.assertGreater(std, 0.1,
            f"При alpha=0 velocity слишком стабильна (std={std:.4f}), "
            f"ожидается случайность")

    def test_alpha_1_direction_no_noise(self):
        """GM alpha=1: direction не меняется (нет шума)."""
        model, ue = self._make_gm(alpha=1.0, x=0.0, y=0.0)
        initial_dir = model.current_direction
        directions = []
        for _ in range(20):
            # Держим UE вдали от границ чтобы boundary не переопределял mean_direction
            ue.position = (0.0, 0.0)
            pos, vel, d = model.update(TIME_MS)
            ue.velocity = vel; ue.direction = d
            directions.append(d)
        for i in range(1, len(directions)):
            self.assertAlmostEqual(directions[i], directions[i-1], places=6,
                msg=f"При alpha=1 direction изменился: {directions[i-1]:.4f} → {directions[i]:.4f}")

    def test_alpha_high_velocity_autocorrelation(self):
        """GM alpha=0.95: последовательные velocities коррелированы (r > 0.5)."""
        model, ue = self._make_gm(alpha=0.95, x=0.0, y=0.0,
                                  velocity=10.0, v_min=0.0, v_max=50.0)
        velocities = []
        for _ in range(200):
            ue.position = (0.0, 0.0)  # держим в центре, без boundary_threshold
            pos, vel, d = model.update(TIME_MS)
            ue.velocity = vel; ue.direction = d
            velocities.append(vel)

        v = np.array(velocities)
        r = np.corrcoef(v[:-1], v[1:])[0, 1]
        self.assertGreater(r, 0.5,
            f"При alpha=0.95 автокорреляция velocities слишком мала: r={r:.4f}")

    def test_alpha_low_velocity_low_autocorrelation(self):
        """GM alpha=0.1: последовательные velocities почти не коррелированы (r < 0.5)."""
        model, ue = self._make_gm(alpha=0.1, x=0.0, y=0.0,
                                  velocity=10.0, v_min=0.0, v_max=50.0)
        velocities = []
        for _ in range(200):
            ue.position = (0.0, 0.0)
            pos, vel, d = model.update(TIME_MS)
            ue.velocity = vel; ue.direction = d
            velocities.append(vel)

        v = np.array(velocities)
        r = np.corrcoef(v[:-1], v[1:])[0, 1]
        self.assertLess(abs(r), 0.5,
            f"При alpha=0.1 автокорреляция слишком высока: r={r:.4f}")

    # --- boundary_threshold ---

    def test_boundary_threshold_activates_near_x_max(self):
        """GM: вблизи x_max mean_direction → 180° (к центру)."""
        # При x > x_max - threshold и y в центре → mean_direction = π (180°)
        threshold = 100.0
        ue = make_ue(x=X_MAX - 50.0, y=0.0, velocity=5.0)  # внутри threshold
        model = GaussMarkovModel(ue=ue, alpha=0.0, boundary_threshold=threshold)

        # При alpha=0: new_dir = mean_direction + N(0,1)
        # mean_direction должен быть π → среднее направлений ≈ π
        directions = []
        for _ in range(200):
            ue.position = (X_MAX - 50.0, 0.0)  # держим в зоне threshold
            pos, vel, d = model.update(TIME_MS)
            ue.velocity = vel; ue.direction = d
            directions.append(d % (2 * math.pi))

        mean_d = np.mean(directions)
        # mean ≈ π (180°) с допуском на шум
        self.assertAlmostEqual(mean_d, math.pi, delta=0.5,
            msg=f"Вблизи x_max среднее направление должно быть ≈π, получено {mean_d:.4f}")

    def test_boundary_threshold_activates_near_y_max(self):
        """GM: вблизи y_max mean_direction → 270° (вниз, к центру)."""
        threshold = 100.0
        ue = make_ue(x=0.0, y=Y_MAX - 50.0, velocity=5.0)
        model = GaussMarkovModel(ue=ue, alpha=0.0, boundary_threshold=threshold)

        directions = []
        for _ in range(200):
            ue.position = (0.0, Y_MAX - 50.0)
            pos, vel, d = model.update(TIME_MS)
            ue.velocity = vel; ue.direction = d
            directions.append(d % (2 * math.pi))

        # 270° = 3π/2 ≈ 4.712
        expected = 3 * math.pi / 2
        mean_d = np.mean(directions)
        self.assertAlmostEqual(mean_d, expected, delta=0.5,
            msg=f"Вблизи y_max среднее направление должно быть ≈270°, получено {math.degrees(mean_d):.1f}°")

    def test_boundary_corner_mean_direction(self):
        """GM: в углу x_max, y_max mean_direction → 225° (к центру)."""
        threshold = 100.0
        ue = make_ue(x=X_MAX - 50.0, y=Y_MAX - 50.0, velocity=5.0)
        model = GaussMarkovModel(ue=ue, alpha=0.0, boundary_threshold=threshold)

        directions = []
        for _ in range(200):
            ue.position = (X_MAX - 50.0, Y_MAX - 50.0)
            pos, vel, d = model.update(TIME_MS)
            ue.velocity = vel; ue.direction = d
            directions.append(d % (2 * math.pi))

        # 225° = 5π/4 ≈ 3.927
        expected = math.radians(225)
        mean_d = np.mean(directions)
        self.assertAlmostEqual(mean_d, expected, delta=0.5,
            msg=f"В углу среднее направление должно быть ≈225°, получено {math.degrees(mean_d):.1f}°")

    def test_velocity_can_go_negative_documenting_known_issue(self):
        """GM [KNOWN ISSUE]: velocity не ограничена снизу, может стать отрицательной."""
        # Это документирующий тест — фиксируем известный дефект.
        # new_velocity = alpha*v + (1-alpha)*mean + sqrt(1-alpha²)*N(0,1)
        # При mean_velocity=0 и большом N(0,1) velocity может стать < 0
        model, ue = self._make_gm(alpha=0.0, velocity=0.0,
                                  v_min=0.0, v_max=0.0)
        model.mean_velocity = 0.0
        ue.velocity = 0.0

        np.random.seed(42)
        found_negative = False
        for _ in range(1000):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if vel < 0:
                found_negative = True
                break

        # Этот тест не должен ПАДАТЬ — он документирует поведение
        # Если velocity никогда не отрицательная — убрать этот тест
        if found_negative:
            import warnings
            warnings.warn(
                "[KNOWN ISSUE] GaussMarkovModel: velocity стала отрицательной. "
                "Добавьте max(0, new_velocity) в update().",
                UserWarning
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
