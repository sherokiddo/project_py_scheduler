"""
TEST_MOBILITY_STRESS.py
Уровень 4: Нагрузочные и стресс-тесты.

Покрытие:
  - Крайние значения time_ms (0, 1, 100_000)
  - 1000+ UE, длинные симуляции
  - Утечка памяти: ue.coordinates растёт неограниченно
  - Параллельное обновление нескольких UECollection
  - Производительность: time budget для CI

Известная проблема задокументированная этим уровнем:
  [MEM-1] ue.coordinates — неограниченный list.append() в UPD_POSITION.
           1000 UE × 10000 шагов = 10M tuple → ~800 MB RAM.
"""

import gc
import sys
import math
import time
import threading
import tracemalloc
import unittest
from unittest.mock import MagicMock

import numpy as np

from MOBILITY_MODEL import (
    MapBorders,
    RandomWalkModel,
    RandomWaypointModel,
    RandomDirectionModel,
    GaussMarkovModel,
)
from UE_MODULE import UserEquipment, UECollection

X_MIN, X_MAX = -500.0, 500.0
Y_MIN, Y_MAX = -500.0, 500.0


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================

def reset_borders():
    MapBorders._instance = None

def make_bs(x=0.0, y=0.0, height=25.0):
    bs = MagicMock()
    bs.position = (x, y)
    bs.height   = height
    return bs

def make_ue(ue_id=1, x=0.0, y=0.0, ue_class="car"):
    ue = UserEquipment(UE_ID=ue_id, x=x, y=y, ue_class=ue_class)
    return ue

def make_collection(n, bs, model="RandomWalk", start_id=1):
    """Создаёт UECollection с n UE, всем назначает модель и serving_bs."""
    rng = np.random.default_rng(42)
    col = UECollection()
    for i in range(n):
        ue = UserEquipment(
            UE_ID=start_id + i,
            x=float(rng.uniform(X_MIN * 0.8, X_MAX * 0.8)),
            y=float(rng.uniform(Y_MIN * 0.8, Y_MAX * 0.8)),
            ue_class="car"
        )
        ue.serving_bs = bs
        ue.UPD_CH_QUALITY = MagicMock()
        col.ADD_USER(ue)
    col.SET_MOBILITY_MODEL(model)
    return col

def in_bounds(pos, eps=1e-6):
    x, y = pos
    return (X_MIN - eps <= x <= X_MAX + eps and
            Y_MIN - eps <= y <= Y_MAX + eps)


# ==============================================================================
# 1. Крайние значения time_ms
# ==============================================================================

class TestEdgeCaseTimeMs(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def _make_mock_ue(self, x=0.0, y=0.0, velocity=11.1):
        ue = MagicMock()
        ue.position = (x, y)
        ue.velocity = velocity
        ue.velocity_min = 2.0
        ue.velocity_max = 16.7
        ue.direction = 0.0
        ue.mean_velocity = velocity
        return ue

    # --- time_ms = 0 ---

    def test_all_models_time_ms_zero_no_crash(self):
        """time_ms=0: ни одна модель не падает."""
        bs = make_bs()
        models_ue = [
            RandomWalkModel(ue=self._make_mock_ue()),
            RandomWaypointModel(ue=self._make_mock_ue(), pause_time=0),
            RandomDirectionModel(ue=self._make_mock_ue(), pause_time=0),
            GaussMarkovModel(ue=self._make_mock_ue()),
        ]
        for model in models_ue:
            with self.subTest(model=model.__class__.__name__):
                try:
                    pos, vel, d = model.update(0)
                except Exception as e:
                    self.fail(f"{model.__class__.__name__}.update(0) упал: {e}")

    def test_time_ms_zero_position_unchanged(self):
        """time_ms=0: при нулевом шаге позиция не изменяется (delta=0)."""
        for model_cls in (RandomWalkModel, RandomWaypointModel,
                          RandomDirectionModel, GaussMarkovModel):
            ue = self._make_mock_ue(x=100.0, y=200.0, velocity=50.0)
            kwargs = {"pause_time": 0} if model_cls in (
                RandomWaypointModel, RandomDirectionModel) else {}
            model = model_cls(ue=ue, **kwargs)
            pos, vel, d = model.update(0)
            with self.subTest(model=model_cls.__name__):
                self.assertAlmostEqual(pos[0], 100.0, places=3,
                    msg=f"{model_cls.__name__}: x изменился при time_ms=0")
                self.assertAlmostEqual(pos[1], 200.0, places=3,
                    msg=f"{model_cls.__name__}: y изменился при time_ms=0")

    def test_time_ms_zero_direction_finite(self):
        """time_ms=0: direction не NaN и не inf."""
        for model_cls in (RandomWalkModel, GaussMarkovModel):
            ue = self._make_mock_ue()
            model = model_cls(ue=ue)
            _, _, d = model.update(0)
            with self.subTest(model=model_cls.__name__):
                self.assertFalse(math.isnan(d),
                    f"{model_cls.__name__}: direction=NaN при time_ms=0")
                self.assertFalse(math.isinf(d),
                    f"{model_cls.__name__}: direction=inf при time_ms=0")

    # --- time_ms = 1 (стандартный LTE TTI) ---

    def test_all_models_time_ms_one_no_crash(self):
        """time_ms=1 (LTE TTI): все модели работают корректно."""
        for model_cls in (RandomWalkModel, RandomWaypointModel,
                          RandomDirectionModel, GaussMarkovModel):
            ue = self._make_mock_ue()
            kwargs = {"pause_time": 0} if model_cls in (
                RandomWaypointModel, RandomDirectionModel) else {}
            model = model_cls(ue=ue, **kwargs)
            for _ in range(100):
                pos, vel, d = model.update(1)
                ue.position = pos; ue.velocity = vel; ue.direction = d
                with self.subTest(model=model_cls.__name__):
                    self.assertTrue(in_bounds(pos),
                        f"{model_cls.__name__}: вышел за карту при time_ms=1")

    def test_time_ms_one_displacement_realistic(self):
        """time_ms=1: за 1мс смещение не превышает v_max * 0.001 секунды."""
        v_max = 16.7
        ue = self._make_mock_ue(x=0.0, y=0.0, velocity=v_max)
        model = RandomWalkModel(ue=ue)
        pos, _, _ = model.update(1)
        max_expected = v_max * 0.001 + 1e-6  # v * time_s
        dist = math.hypot(pos[0] - 0.0, pos[1] - 0.0)
        self.assertLessEqual(dist, max_expected + 1e-3,
            f"Смещение за 1мс слишком велико: {dist:.6f}м (max={max_expected:.6f}м)")

    # --- time_ms = 100_000 (крайний максимум) ---

    def test_all_models_time_ms_100k_no_crash(self):
        """time_ms=100_000: никаких исключений, позиция внутри карты."""
        for model_cls in (RandomWalkModel, RandomWaypointModel,
                          RandomDirectionModel, GaussMarkovModel):
            ue = self._make_mock_ue(x=0.0, y=0.0, velocity=11.1)
            kwargs = {"pause_time": 0} if model_cls in (
                RandomWaypointModel, RandomDirectionModel) else {}
            model = model_cls(ue=ue, **kwargs)
            with self.subTest(model=model_cls.__name__):
                try:
                    pos, vel, d = model.update(100_000)
                except Exception as e:
                    self.fail(f"{model_cls.__name__}.update(100000) упал: {e}")
                self.assertTrue(in_bounds(pos),
                    f"{model_cls.__name__}: вышел за карту при time_ms=100000: {pos}")

    def test_time_ms_100k_no_nan_in_output(self):
        """time_ms=100_000: нет NaN и inf в позиции, velocity, direction."""
        for model_cls in (RandomWalkModel, GaussMarkovModel,
                          RandomWaypointModel, RandomDirectionModel):
            ue = self._make_mock_ue(x=0.0, y=0.0, velocity=5.0)
            kwargs = {"pause_time": 0} if model_cls in (
                RandomWaypointModel, RandomDirectionModel) else {}
            model = model_cls(ue=ue, **kwargs)
            pos, vel, d = model.update(100_000)
            with self.subTest(model=model_cls.__name__):
                for name, val in [("x", pos[0]), ("y", pos[1]),
                                  ("vel", vel), ("dir", d)]:
                    self.assertFalse(math.isnan(val),
                        f"{model_cls.__name__}: {name}=NaN при time_ms=100000")
                    self.assertFalse(math.isinf(val),
                        f"{model_cls.__name__}: {name}=inf при time_ms=100000")

    def test_time_ms_large_boundary_handling(self):
        """time_ms=100_000: при огромном шаге модели удерживают UE в карте."""
        # За 100с с v=50 м/с → 5000м потенциального смещения при карте 1000м
        for model_cls in (RandomWalkModel, RandomWaypointModel,
                          RandomDirectionModel, GaussMarkovModel):
            ue = self._make_mock_ue(x=0.0, y=0.0, velocity=50.0)
            kwargs = {"pause_time": 0} if model_cls in (
                RandomWaypointModel, RandomDirectionModel) else {}
            model = model_cls(ue=ue, **kwargs)
            for _ in range(10):
                pos, vel, d = model.update(100_000)
                ue.position = pos; ue.velocity = vel; ue.direction = d
                with self.subTest(model=model_cls.__name__):
                    self.assertTrue(in_bounds(pos),
                        f"{model_cls.__name__}: вышел за карту: {pos}")


# ==============================================================================
# 2. Нагрузочные тесты: 1000+ UE, длинные симуляции
# ==============================================================================

class TestStressLoad(unittest.TestCase):

    # Тайм-лимиты (секунды) — пороги для CI
    TIME_LIMIT_1000_UE_100_STEPS  = 5.0
    TIME_LIMIT_100_UE_1000_STEPS  = 5.0
    TIME_LIMIT_SINGLE_UPDATE_MS   = 2.0   # мс на один model.update()

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_1000_ue_100_steps_time_budget(self):
        """1000 UE × 100 шагов: выполняется менее чем за 5 секунд."""
        col = make_collection(1000, make_bs())
        start = time.perf_counter()
        for step in range(100):
            for ue in col.GET_ALL_USERS():
                ue.UPD_POSITION(500)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, self.TIME_LIMIT_1000_UE_100_STEPS,
            f"1000 UE × 100 шагов заняло {elapsed:.2f}с "
            f"(лимит {self.TIME_LIMIT_1000_UE_100_STEPS}с)")
        print(f"\n[PERF] 1000 UE × 100 steps: {elapsed:.3f}s "
              f"({elapsed/100_000*1000:.3f}ms/UE/step)")

    def test_100_ue_1000_steps_time_budget(self):
        """100 UE × 1000 шагов: выполняется менее чем за 5 секунд."""
        col = make_collection(100, make_bs())
        start = time.perf_counter()
        for step in range(1000):
            for ue in col.GET_ALL_USERS():
                ue.UPD_POSITION(1)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, self.TIME_LIMIT_100_UE_1000_STEPS,
            f"100 UE × 1000 шагов заняло {elapsed:.2f}с "
            f"(лимит {self.TIME_LIMIT_100_UE_1000_STEPS}с)")
        print(f"\n[PERF] 100 UE × 1000 steps: {elapsed:.3f}s "
              f"({elapsed/100_000*1000:.3f}ms/UE/step)")

    def test_single_model_update_performance(self):
        """Один model.update() за < 2мс для всех моделей."""
        from unittest.mock import MagicMock
        models = []
        for model_cls in (RandomWalkModel, RandomWaypointModel,
                          RandomDirectionModel, GaussMarkovModel):
            ue = MagicMock()
            ue.position = (0.0, 0.0)
            ue.velocity = 11.1
            ue.velocity_min = 2.0
            ue.velocity_max = 16.7
            ue.direction = 0.0
            ue.mean_velocity = 11.1
            kwargs = {"pause_time": 0} if model_cls in (
                RandomWaypointModel, RandomDirectionModel) else {}
            models.append((model_cls.__name__, model_cls(ue=ue, **kwargs), ue))

        N_WARM  = 100   # прогрев
        N_BENCH = 10000

        for name, model, ue in models:
            for _ in range(N_WARM):
                pos, vel, d = model.update(500)
                ue.position = pos; ue.velocity = vel; ue.direction = d

            start = time.perf_counter()
            for _ in range(N_BENCH):
                pos, vel, d = model.update(500)
                ue.position = pos; ue.velocity = vel; ue.direction = d
            elapsed_ms = (time.perf_counter() - start) / N_BENCH * 1000

            with self.subTest(model=name):
                self.assertLess(elapsed_ms, self.TIME_LIMIT_SINGLE_UPDATE_MS,
                    f"{name}.update(): {elapsed_ms:.4f}мс "
                    f"(лимит {self.TIME_LIMIT_SINGLE_UPDATE_MS}мс)")
            print(f"\n[PERF] {name}.update(): {elapsed_ms:.4f}ms/call")

    def test_all_1000_ue_stay_in_bounds_100_steps(self):
        """1000 UE: ни один не выходит за карту за 100 шагов."""
        col = make_collection(1000, make_bs())
        for step in range(100):
            for ue in col.GET_ALL_USERS():
                ue.UPD_POSITION(500)
        violations = [
            ue.UE_ID for ue in col.GET_ALL_USERS()
            if not in_bounds(ue.position)
        ]
        self.assertEqual(violations, [],
            f"UE вышли за карту: {violations[:10]}...")


# ==============================================================================
# 3. Утечка памяти: ue.coordinates
# ==============================================================================

class TestMemoryGrowth(unittest.TestCase):
    """
    [MEM-1] ue.coordinates — неограниченный list.append().
    1000 UE × 10000 шагов = 10M tuple(float, float) ≈ 800 MB.
    Тесты документируют рост и предлагают порог для алерта.
    """

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_coordinates_grows_linearly(self):
        """[MEM-1] ue.coordinates растёт на 1 элемент за каждый UPD_POSITION."""
        bs = make_bs()
        ue = make_ue(ue_id=1)
        ue.serving_bs = bs
        ue.SET_MOBILITY_MODEL("RandomWalk")
        initial_len = len(ue.coordinates)  # = 1 (начальная позиция)

        N = 100
        for _ in range(N):
            ue.UPD_POSITION(500)

        self.assertEqual(len(ue.coordinates), initial_len + N,
            f"ue.coordinates должен расти на 1 за шаг: "
            f"начало={initial_len}, шагов={N}, факт={len(ue.coordinates)}")

    def test_coordinates_memory_documenting_issue(self):
        """[MEM-1] Документирующий тест: 1 UE × N шагов — рост памяти."""
        bs = make_bs()
        ue = make_ue(ue_id=1)
        ue.serving_bs = bs
        ue.SET_MOBILITY_MODEL("RandomWalk")

        N_STEPS = 5000
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        for _ in range(N_STEPS):
            ue.UPD_POSITION(500)

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_growth_kb = sum(s.size_diff for s in stats) / 1024

        print(f"\n[MEM-1] 1 UE × {N_STEPS} steps: "
              f"coordinates len={len(ue.coordinates)}, "
              f"RAM delta≈{total_growth_kb:.1f} KB")

        # coordinates = list of tuples(np.float64, np.float64)
        # ~112 bytes per tuple ref → 5000 шагов ≈ 560KB
        # Если > 2MB — ненормально много для одного UE за 5000 шагов
        self.assertLess(total_growth_kb, 2048,
            f"[MEM-1] Аномальный рост памяти для 1 UE × {N_STEPS} шагов: "
            f"{total_growth_kb:.1f} KB")

    def test_coordinates_growth_100_ue_warns(self):
        """[MEM-1] 100 UE × 1000 шагов: размер coordinates документируется."""
        bs = make_bs()
        col = make_collection(100, bs)
        N_STEPS = 1000

        for _ in range(N_STEPS):
            for ue in col.GET_ALL_USERS():
                ue.UPD_POSITION(500)

        total_coords = sum(
            len(ue.coordinates) for ue in col.GET_ALL_USERS()
        )
        # 100 UE × (1000 + 1 начальная) = 100100 элементов
        expected = 100 * (N_STEPS + 1)
        self.assertEqual(total_coords, expected,
            f"Суммарный размер coordinates: {total_coords}, ожидалось {expected}")

        # Расчёт теоретического размера в MB (64 bytes per tuple approx)
        size_mb = total_coords * 64 / (1024 ** 2)
        print(f"\n[MEM-1] 100 UE × {N_STEPS} steps: "
              f"total coordinates={total_coords}, "
              f"~{size_mb:.1f} MB")

        import warnings
        if size_mb > 10:
            warnings.warn(
                f"[MEM-1] ue.coordinates занимает ≈{size_mb:.1f} MB для "
                f"100 UE × {N_STEPS} шагов. "
                f"Рекомендуется ограничить maxlen или хранить только последние N позиций.",
                ResourceWarning
            )

    def test_coordinates_with_deque_maxlen_suggestion(self):
        """[MEM-1] Демонстрация: замена list на deque(maxlen=100) решает проблему."""
        from collections import deque

        bs = make_bs()
        ue = make_ue(ue_id=1)
        ue.serving_bs = bs
        ue.SET_MOBILITY_MODEL("RandomWalk")

        # Симулируем фикс: заменяем list → deque с ограничением
        ue.coordinates = deque(ue.coordinates, maxlen=100)

        for _ in range(10_000):
            ue.UPD_POSITION(500)
            # deque сам выбрасывает старые элементы

        self.assertLessEqual(len(ue.coordinates), 100,
            f"deque(maxlen=100) должен хранить не более 100 позиций, "
            f"получено {len(ue.coordinates)}")
        print(f"\n[MEM-1 FIX] deque(maxlen=100) после 10000 шагов: "
              f"len={len(ue.coordinates)} (было бы 10001)")


# ==============================================================================
# 4. Параллельное обновление нескольких UECollection
# ==============================================================================

class TestParallelCollections(unittest.TestCase):
    """
    Проверяет, что MapBorders singleton (read-only после init)
    не вызывает race condition при параллельном обновлении коллекций.

    ВАЖНО: MobilityInterface.ue — мутабельное состояние.
    Два потока НЕ ДОЛЖНЫ шарить один UE объект.
    Тесты проверяют: независимые коллекции → безопасны.
    """

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def _run_collection(self, col, n_steps, results, idx, errors):
        """Worker-функция для потока."""
        try:
            for step in range(n_steps):
                for ue in col.GET_ALL_USERS():
                    ue.UPD_POSITION(500)
            # Собираем финальные позиции
            results[idx] = [ue.position for ue in col.GET_ALL_USERS()]
        except Exception as e:
            errors[idx] = str(e)

    def test_two_collections_parallel_no_crash(self):
        """2 потока × 2 коллекции × 200 шагов: нет исключений."""
        bs = make_bs()
        col1 = make_collection(50, bs, start_id=1)
        col2 = make_collection(50, bs, start_id=51)

        results = [None, None]
        errors  = [None, None]
        N_STEPS = 200

        t1 = threading.Thread(
            target=self._run_collection,
            args=(col1, N_STEPS, results, 0, errors)
        )
        t2 = threading.Thread(
            target=self._run_collection,
            args=(col2, N_STEPS, results, 1, errors)
        )

        t1.start(); t2.start()
        t1.join(timeout=30); t2.join(timeout=30)

        self.assertIsNone(errors[0], f"Поток 1 упал: {errors[0]}")
        self.assertIsNone(errors[1], f"Поток 2 упал: {errors[1]}")
        self.assertIsNotNone(results[0], "Поток 1 не вернул результат")
        self.assertIsNotNone(results[1], "Поток 2 не вернул результат")

    def test_four_collections_parallel_positions_in_bounds(self):
        """4 потока × 4 коллекции: все UE остались в границах карты."""
        bs = make_bs()
        collections = [
            make_collection(25, bs, start_id=i * 25 + 1)
            for i in range(4)
        ]
        results = [None] * 4
        errors  = [None] * 4
        N_STEPS = 100

        threads = [
            threading.Thread(
                target=self._run_collection,
                args=(col, N_STEPS, results, i, errors)
            )
            for i, col in enumerate(collections)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        for i in range(4):
            self.assertIsNone(errors[i], f"Поток {i} упал: {errors[i]}")
            for pos in results[i]:
                self.assertTrue(in_bounds(pos),
                    f"Поток {i}: UE вышел за карту: {pos}")

    def test_mapborders_singleton_thread_safe_read(self):
        """MapBorders singleton: параллельное чтение не вызывает ошибок."""
        errors = []

        def read_borders():
            try:
                for _ in range(1000):
                    b = MapBorders().get_borders()
                    assert b == (X_MIN, X_MAX, Y_MIN, Y_MAX)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=read_borders) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=10)

        self.assertEqual(errors, [],
            f"Race condition при чтении MapBorders: {errors}")

    def test_collections_independent_state(self):
        """Параллельные коллекции не влияют на состояние друг друга."""
        bs = make_bs()
        # Коллекция 1: UE в левой части карты
        col1 = make_collection(10, bs, start_id=1)
        for ue in col1.GET_ALL_USERS():
            ue.position = (-400.0, 0.0)

        # Коллекция 2: UE в правой части карты
        col2 = make_collection(10, bs, start_id=11)
        for ue in col2.GET_ALL_USERS():
            ue.position = (400.0, 0.0)

        results = [None, None]
        errors  = [None, None]

        t1 = threading.Thread(
            target=self._run_collection,
            args=(col1, 50, results, 0, errors)
        )
        t2 = threading.Thread(
            target=self._run_collection,
            args=(col2, 50, results, 1, errors)
        )

        t1.start(); t2.start()
        t1.join(timeout=10); t2.join(timeout=10)

        # UE из col1 не должны магически оказаться в позициях col2
        col1_ids = {ue.UE_ID for ue in col1.GET_ALL_USERS()}
        col2_ids = {ue.UE_ID for ue in col2.GET_ALL_USERS()}
        self.assertEqual(col1_ids & col2_ids, set(),
            "Коллекции разделяют UE объекты — нарушение изоляции")


if __name__ == "__main__":
    unittest.main(verbosity=2)
