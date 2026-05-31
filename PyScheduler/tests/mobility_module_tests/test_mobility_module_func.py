"""
TEST_MOBILITY_FUNCTIONAL.py
Уровень 2: Функциональные тесты моделей мобильности.
Покрытие: граничные условия, корректность движения, специфика каждой модели.
"""

import math
import unittest
import numpy as np
from unittest.mock import MagicMock

from MOBILITY_MODEL import (
    MapBorders,
    RandomWalkModel,
    RandomWaypointModel,
    RandomDirectionModel,
    GaussMarkovModel,
    DiagonalWalkModel,
)

X_MIN, X_MAX = -500.0, 500.0
Y_MIN, Y_MAX = -500.0, 500.0
TIME_MS      = 500
N_STEPS      = 500   # прогон для статистических проверок


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

def make_bs(x=0.0, y=0.0):
    bs = MagicMock()
    bs.position = (x, y)
    return bs

def in_bounds(pos):
    """True если позиция строго внутри карты (с допуском на float)."""
    x, y = pos
    eps = 1e-6
    return (X_MIN - eps <= x <= X_MAX + eps and
            Y_MIN - eps <= y <= Y_MAX + eps)

def simulate(model, ue, n_steps=N_STEPS, time_ms=TIME_MS):
    """Прогоняет модель n_steps шагов, обновляя ue.position после каждого."""
    positions = [ue.position]
    for _ in range(n_steps):
        pos, vel, direction = model.update(time_ms)
        ue.position  = pos
        ue.velocity  = vel
        ue.direction = direction
        positions.append(pos)
    return positions


# ==============================================================================
# 1. ГРАНИЧНЫЕ УСЛОВИЯ (самое критичное)
# ==============================================================================

class TestBoundaryConditions(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    # --- RandomWalk ---

    def test_rw_never_exits_map(self):
        """RandomWalk: за N шагов UE не выходит за границы карты."""
        ue = make_ue(x=0.0, y=0.0)
        model = RandomWalkModel(ue=ue)
        positions = simulate(model, ue)
        for i, pos in enumerate(positions):
            self.assertTrue(in_bounds(pos),
                f"RandomWalk вышел за границы на шаге {i}: {pos}")

    def test_rw_reflection_from_x_max(self):
        """RandomWalk: UE у правой границы — должен отразиться."""
        ue = make_ue(x=499.0, y=0.0, direction=0.0)  # летит вправо
        model = RandomWalkModel(ue=ue)
        pos, _, _ = model.update(TIME_MS)
        self.assertLessEqual(pos[0], X_MAX + 1e-6,
            f"После отражения x={pos[0]} вышел за x_max={X_MAX}")

    def test_rw_reflection_from_x_min(self):
        """RandomWalk: UE у левой границы — должен отразиться."""
        ue = make_ue(x=-499.0, y=0.0, direction=math.pi)  # летит влево
        model = RandomWalkModel(ue=ue)
        pos, _, _ = model.update(TIME_MS)
        self.assertGreaterEqual(pos[0], X_MIN - 1e-6,
            f"После отражения x={pos[0]} вышел за x_min={X_MIN}")

    def test_rw_reflection_from_y_max(self):
        """RandomWalk: UE у верхней границы — должен отразиться."""
        ue = make_ue(x=0.0, y=499.0, direction=math.pi/2)  # летит вверх
        model = RandomWalkModel(ue=ue)
        pos, _, _ = model.update(TIME_MS)
        self.assertLessEqual(pos[1], Y_MAX + 1e-6)

    def test_rw_reflection_from_y_min(self):
        """RandomWalk: UE у нижней границы — должен отразиться."""
        ue = make_ue(x=0.0, y=-499.0, direction=-math.pi/2)  # летит вниз
        model = RandomWalkModel(ue=ue)
        pos, _, _ = model.update(TIME_MS)
        self.assertGreaterEqual(pos[1], Y_MIN - 1e-6)

    def test_rw_corner_reflection(self):
        """RandomWalk: UE в углу карты — не вылетает."""
        ue = make_ue(x=498.0, y=498.0, direction=math.pi/4)  # летит в угол
        model = RandomWalkModel(ue=ue)
        for _ in range(10):
            pos, vel, d = model.update(TIME_MS)
            ue.position  = pos
            ue.velocity  = vel
            ue.direction = d
            self.assertTrue(in_bounds(pos), f"Угловой отскок провален: {pos}")

    def test_rw_start_on_border(self):
        """RandomWalk: UE стартует ровно на границе — не вылетает."""
        ue = make_ue(x=X_MAX, y=0.0, direction=0.0)
        model = RandomWalkModel(ue=ue)
        pos, _, _ = model.update(TIME_MS)
        self.assertTrue(in_bounds(pos), f"Старт на границе: {pos}")

    # --- GaussMarkov ---

    def test_gm_never_exits_map(self):
        """GaussMarkov: за N шагов UE не выходит за границы."""
        ue = make_ue(x=0.0, y=0.0)
        model = GaussMarkovModel(ue=ue, boundary_threshold=50.0)
        positions = simulate(model, ue)
        for i, pos in enumerate(positions):
            self.assertTrue(in_bounds(pos),
                f"GaussMarkov вышел за границы на шаге {i}: {pos}")

    # --- RandomWaypoint ---

    def test_rwp_destination_inside_map(self):
        """RandomWaypoint: destination всегда внутри карты."""
        ue = make_ue()
        model = RandomWaypointModel(ue=ue)
        model.update(0)
        dest = model.destination
        self.assertTrue(X_MIN <= dest[0] <= X_MAX,
            f"destination.x={dest[0]} вне карты")
        self.assertTrue(Y_MIN <= dest[1] <= Y_MAX,
            f"destination.y={dest[1]} вне карты")

    def test_rwp_never_exits_map(self):
        """RandomWaypoint: за N шагов UE не выходит за границы."""
        ue = make_ue()
        model = RandomWaypointModel(ue=ue, pause_time=0)
        positions = simulate(model, ue)
        for i, pos in enumerate(positions):
            self.assertTrue(in_bounds(pos),
                f"RandomWaypoint вышел за границы на шаге {i}: {pos}")

    # --- RandomDirection ---

    def test_rd_never_exits_map(self):
        """RandomDirection: за N шагов UE не выходит за границы."""
        ue = make_ue()
        model = RandomDirectionModel(ue=ue, pause_time=0)
        positions = simulate(model, ue)
        for i, pos in enumerate(positions):
            self.assertTrue(in_bounds(pos),
                f"RandomDirection вышел за границы на шаге {i}: {pos}")

    def test_rd_destination_on_boundary(self):
        """RandomDirection: destination должен лежать на границе карты."""
        ue = make_ue(x=0.0, y=0.0)
        model = RandomDirectionModel(ue=ue, pause_time=0)
        model.update(0)
        dest = model.destination
        on_x_border = (abs(dest[0] - X_MIN) < 1.0 or abs(dest[0] - X_MAX) < 1.0)
        on_y_border = (abs(dest[1] - Y_MIN) < 1.0 or abs(dest[1] - Y_MAX) < 1.0)
        self.assertTrue(on_x_border or on_y_border,
            f"Destination не на границе: {dest}")


# ==============================================================================
# 2. КОРРЕКТНОСТЬ ДВИЖЕНИЯ
# ==============================================================================

class TestMovementCorrectness(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_velocity_in_range_random_walk(self):
        """RandomWalk: velocity всегда в [v_min, v_max]."""
        ue = make_ue(v_min=2.0, v_max=16.7)
        model = RandomWalkModel(ue=ue)
        for _ in range(N_STEPS):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            self.assertGreaterEqual(vel, 0.0)
            # vel может быть 0 только если модель явно это допускает
            if vel > 0:
                self.assertGreaterEqual(vel, ue.velocity_min - 1e-6)
                self.assertLessEqual(vel, ue.velocity_max + 1e-6)

    def test_velocity_zero_means_no_movement(self):
        """Если velocity=0, позиция не должна меняться."""
        ue = make_ue(x=100.0, y=100.0, velocity=0.0, v_min=0.0, v_max=0.0)
        model = RandomWalkModel(ue=ue)
        pos, _, _ = model.update(TIME_MS)
        self.assertAlmostEqual(pos[0], 100.0, places=3,
            msg="При velocity=0 x изменился")
        self.assertAlmostEqual(pos[1], 100.0, places=3,
            msg="При velocity=0 y изменился")

    def test_direction_is_finite_all_models(self):
        """Direction конечное число (не NaN, не inf) для всех моделей."""
        bs = make_bs()
        models = [
            RandomWalkModel(ue=make_ue()),
            RandomWaypointModel(ue=make_ue(), pause_time=0),
            RandomDirectionModel(ue=make_ue(), pause_time=0),
            GaussMarkovModel(ue=make_ue()),
            DiagonalWalkModel(ue=make_ue(x=200.0, y=200.0), bs=bs, pause_time=0),
        ]
        for model in models:
            ue = model.ue
            for _ in range(50):
                pos, vel, d = model.update(TIME_MS)
                ue.position = pos; ue.velocity = vel; ue.direction = d
                self.assertTrue(math.isfinite(d),
                    f"{type(model).__name__}: direction={d} не конечное")

    def test_position_changes_over_time(self):
        """UE со скоростью > 0 должен двигаться (позиция меняется за N шагов)."""
        ue = make_ue(x=0.0, y=0.0, velocity=11.1)
        model = RandomWalkModel(ue=ue)
        start = ue.position
        positions = simulate(model, ue, n_steps=20)
        moved = any(
            abs(p[0] - start[0]) > 0.01 or abs(p[1] - start[1]) > 0.01
            for p in positions[1:]
        )
        self.assertTrue(moved, "UE не сдвинулся за 20 шагов при velocity>0")


# ==============================================================================
# 3. СПЕЦИФИКА КАЖДОЙ МОДЕЛИ
# ==============================================================================

class TestRandomWalkSpecific(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_direction_changes_every_step(self):
        """RandomWalk: направление меняется на каждом шаге (без отражения)."""
        ue = make_ue(x=0.0, y=0.0, direction=0.0)
        model = RandomWalkModel(ue=ue)
        directions = []
        for _ in range(20):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            directions.append(d)
        unique = len(set(round(d, 4) for d in directions))
        self.assertGreater(unique, 1,
            "RandomWalk: направление не меняется между шагами")


class TestRandomWaypointSpecific(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_pause_time_zero_no_stop(self):
        """RandomWaypoint: pause_time=0 — UE не останавливается надолго."""
        ue = make_ue(x=0.0, y=0.0)
        model = RandomWaypointModel(ue=ue, pause_time=0)
        zero_velocity_count = 0
        for _ in range(50):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if vel == 0.0:
                zero_velocity_count += 1
        # Допускаем одну остановку при достижении цели, но не 50%
        self.assertLess(zero_velocity_count, 25,
            "RandomWaypoint с pause_time=0 слишком часто стоит")

    def test_new_destination_chosen_after_arrival(self):
        """RandomWaypoint: после достижения цели выбирается новая точка."""
        ue = make_ue(x=0.0, y=0.0, velocity=100.0, v_min=100.0, v_max=100.0)
        model = RandomWaypointModel(ue=ue, pause_time=0)
        model.update(0)
        first_dest = model.destination.copy()
        destination_changed = False
        for _ in range(50):
            pos, vel, d = model.update(10000)
            ue.position = pos;
            ue.velocity = vel;
            ue.direction = d

            if not np.array_equal(model.destination, first_dest):
                destination_changed = True
                break
        self.assertTrue(destination_changed, "RandomWaypoint: новая цель не была выбрана после 50 шагов")

    def test_pause_reduces_velocity_to_zero(self):
        """RandomWaypoint: при pause_time>0 velocity=0 во время паузы."""
        ue = make_ue(x=0.0, y=0.0, velocity=200.0, v_min=200.0, v_max=200.0)
        model = RandomWaypointModel(ue=ue, pause_time=5000)
        model.update(0)
        ue.position = (model.destination[0] - 5.0, model.destination[1])
        for _ in range(100):
            pos, vel, d = model.update(100)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if vel == 0.0 and model.is_paused:
                found_pause = True
                break
        self.assertTrue(found_pause,
            "RandomWaypoint: пауза с velocity=0 не обнаружена")


class TestGaussMarkovSpecific(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_boundary_triggers_correction(self):
        """GaussMarkov: при выходе за границы происходит отскок."""
        ue = make_ue(x=X_MAX + 5.0, y=0.0)
        model = GaussMarkovModel(ue=ue)
        model.update(0)

        model.current_pos = np.array([X_MAX + 5.0, 0.0, 0.0])
        model.mean_direction = 0.0
        model.direction = 0.0
        model.velocity = 10.0

        model._start()

        self.assertAlmostEqual(model.mean_direction, math.pi, places=1,msg="GaussMarkov: у правой границы угол mean_direction не скорректирован")
    def test_alpha_zero_memoryless(self):
        """GaussMarkov alpha=0: нет памяти, velocity → mean_velocity."""
        ue = make_ue(x=0.0, y=0.0, velocity=5.0)
        model = GaussMarkovModel(ue=ue, alpha=0.0)
        velocities = []
        for _ in range(100):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            velocities.append(vel)
        avg = sum(velocities) / len(velocities)
        self.assertAlmostEqual(avg, model.mean_velocity, delta=3.0,
            msg="GaussMarkov alpha=0: среднее velocity далеко от mean_velocity")

    def test_alpha_one_full_memory(self):
        """GaussMarkov alpha=1: полная память, скорость почти не меняется."""
        ue = make_ue(x=0.0, y=0.0, velocity=11.1)
        model = GaussMarkovModel(ue=ue, alpha=1.0)
        _, v1, _ = model.update(TIME_MS)
        _, v2, _ = model.update(TIME_MS)
        self.assertAlmostEqual(v1, v2, delta=2.0,
            msg="GaussMarkov alpha=1: скорость слишком сильно меняется")


class TestDiagonalWalkSpecific(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_destination_is_opposite_to_bs(self):
        """DiagonalWalk: цель симметрична стартовой позиции относительно BS."""
        bs = make_bs(x=0.0, y=0.0)
        ue = make_ue(x=200.0, y=100.0)
        model = DiagonalWalkModel(ue=ue, bs=bs, pause_time=0)
        # После первого update destination должен быть вычислен
        model.update(TIME_MS)
        dest = model._diagonal_destination
        if dest is not None:
            expected_x = 2 * bs.position[0] - 200.0   # = -200
            expected_y = 2 * bs.position[1] - 100.0   # = -100
            self.assertAlmostEqual(dest[0], expected_x, delta=1.0,
                msg=f"DiagonalWalk: dest.x={dest[0]}, ожидалось {expected_x}")
            self.assertAlmostEqual(dest[1], expected_y, delta=1.0,
                msg=f"DiagonalWalk: dest.y={dest[1]}, ожидалось {expected_y}")

    def test_ue_moves_toward_destination(self):
        """DiagonalWalk: UE приближается к цели с каждым шагом."""
        bs = make_bs(x=0.0, y=0.0)
        ue = make_ue(x=100.0, y=100.0, velocity=11.1, v_min=11.1, v_max=11.1)
        model = DiagonalWalkModel(ue=ue, bs=bs, pause_time=0)
        model.update(TIME_MS)  # инициализация destination
        dest = model._diagonal_destination
        if dest is None:
            return
        prev_dist = math.hypot(ue.position[0] - dest[0],
                                ue.position[1] - dest[1])
        for _ in range(10):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if model.is_paused:
                break
            curr_dist = math.hypot(pos[0] - dest[0], pos[1] - dest[1])
            self.assertLessEqual(curr_dist, prev_dist + 1e-3,
                f"DiagonalWalk: дистанция до цели увеличилась: {prev_dist:.2f} → {curr_dist:.2f}")
            prev_dist = curr_dist

    def test_pause_activates_on_arrival(self):
        """DiagonalWalk: при достижении цели is_paused=True."""
        bs = make_bs(x=0.0, y=0.0)
        ue = make_ue(x=10.0, y=10.0, velocity=500.0, v_min=500.0, v_max=500.0)
        model = DiagonalWalkModel(ue=ue, bs=bs, pause_time=5000)
        found_pause = False
        for _ in range(20):
            pos, vel, d = model.update(TIME_MS)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if model.is_paused:
                found_pause = True
                break
        self.assertTrue(found_pause,
            "DiagonalWalk: пауза при достижении цели не сработала")


class TestRandomDirectionSpecific(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_new_direction_chosen_after_arrival(self):
        """RandomDirection: после достижения границы выбирается новое направление."""
        ue = make_ue(x=0.0, y=0.0, velocity=200.0, v_min=200.0, v_max=200.0)
        model = RandomDirectionModel(ue=ue, pause_time=0)
        model.update(0)
        first_dir = model.current_direction
        direction_changed = False
        for _ in range(50):
            pos, vel, d = model.update(10000)
            ue.position = pos; ue.velocity = vel; ue.direction = d
            if abs(model.current_direction - first_dir) > 1e-4:
                direction_changed = True
                break
        self.assertTrue(direction_changed, "RandomDirection: новое направление не было выбрано")


# ==============================================================================
# ЗАПУСК
# ==============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
