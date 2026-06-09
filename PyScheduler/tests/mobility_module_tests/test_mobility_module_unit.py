"""
TEST_MOBILITY_UNIT.py
Уровень 1: Unit-тесты моделей мобильности.
Покрытие: MapBorders, MobilityInterface (factory), возвращаемые типы и диапазоны.
"""

import math
import unittest
import numpy as np
from unittest.mock import MagicMock

from MOBILITY_MODEL import (
    MapBorders,
    MobilityInterface,
    RandomWalkModel,
    RandomWaypointModel,
    RandomDirectionModel,
    GaussMarkovModel,
    DiagonalWalkModel,
)


# ==============================================================================
# ВСПОМОГАТЕЛЬНАЯ ФАБРИКА ЗАГЛУШЕК
# ==============================================================================

def make_ue(x=0.0, y=0.0, ue_class="pedestrian"):
    """Создаёт минимальную заглушку UserEquipment без импорта UE_MODULE."""
    ue = MagicMock()
    ue.position        = (x, y)
    ue.velocity        = 1.4
    ue.velocity_min    = 0.5
    ue.velocity_max    = 1.8
    ue.direction       = 0.0
    ue.ue_class        = ue_class
    ue.mean_velocity   = 1.4
    return ue

def make_bs(x=0.0, y=0.0):
    bs = MagicMock()
    bs.position = (x, y)
    return bs

def reset_borders():
    MapBorders._instance = None


# ==============================================================================
# 1. ТЕСТЫ MapBorders
# ==============================================================================

class TestMapBorders(unittest.TestCase):

    def setUp(self):
        reset_borders()

    def tearDown(self):
        reset_borders()

    # --- Корректное создание ---

    def test_create_valid(self):
        mb = MapBorders(-500, 500, -500, 500)
        self.assertEqual(mb.get_borders(), (-500, 500, -500, 500))

    def test_create_float_borders(self):
        mb = MapBorders(-1000.5, 1000.5, -500.0, 500.0)
        x_min, x_max, y_min, y_max = mb.get_borders()
        self.assertAlmostEqual(x_min, -1000.5)
        self.assertAlmostEqual(x_max, 1000.5)

    # --- Валидация ---

    def test_x_max_less_than_x_min_raises(self):
        with self.assertRaises(ValueError):
            MapBorders(500, -500, -500, 500)

    def test_x_max_equal_x_min_raises(self):
        with self.assertRaises(ValueError):
            MapBorders(0, 0, -500, 500)

    def test_y_max_less_than_y_min_raises(self):
        with self.assertRaises(ValueError):
            MapBorders(-500, 500, 500, -500)

    def test_non_numeric_x_min_raises(self):
        with self.assertRaises(TypeError):
            MapBorders("left", 500, -500, 500)

    def test_non_numeric_y_max_raises(self):
        with self.assertRaises(TypeError):
            MapBorders(-500, 500, -500, "top")

    # --- Синглтон ---

    def test_singleton_same_object(self):
        mb1 = MapBorders(-500, 500, -500, 500)
        mb2 = MapBorders(-999, 999, -999, 999)   # повторный вызов — тот же объект
        self.assertIs(mb1, mb2)

    def test_singleton_ignores_second_args(self):
        MapBorders(-500, 500, -500, 500)
        mb2 = MapBorders(-999, 999, -999, 999)
        self.assertEqual(mb2.get_borders(), (-500, 500, -500, 500))

    def test_singleton_reset(self):
        MapBorders(-500, 500, -500, 500)
        reset_borders()
        mb = MapBorders(-100, 100, -100, 100)
        self.assertEqual(mb.get_borders(), (-100, 100, -100, 100))

    def test_get_borders_returns_tuple_of_four(self):
        mb = MapBorders(-500, 500, -500, 500)
        borders = mb.get_borders()
        self.assertEqual(len(borders), 4)


# ==============================================================================
# 2. ТЕСТЫ MobilityInterface (Factory)
# ==============================================================================

class TestMobilityFactory(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(-500, 500, -500, 500)
        self.ue = make_ue()
        self.bs = make_bs()

    def tearDown(self):
        reset_borders()

    # --- Создание моделей ---

    def test_create_random_walk(self):
        model = MobilityInterface.create("RandomWalk", ue=self.ue)
        self.assertIsInstance(model, RandomWalkModel)

    def test_create_random_waypoint(self):
        model = MobilityInterface.create("RandomWaypoint", ue=self.ue)
        self.assertIsInstance(model, RandomWaypointModel)

    def test_create_random_direction(self):
        model = MobilityInterface.create("RandomDirection", ue=self.ue)
        self.assertIsInstance(model, RandomDirectionModel)

    def test_create_gauss_markov(self):
        model = MobilityInterface.create("GaussMarkov", ue=self.ue)
        self.assertIsInstance(model, GaussMarkovModel)

    def test_create_diagonal_walk(self):
        model = MobilityInterface.create(
            "DiagonalWalk", ue=self.ue, bs=self.bs, pause_time=0
        )
        self.assertIsInstance(model, DiagonalWalkModel)

    # --- Все модели наследуют MobilityInterface ---

    def test_all_models_inherit_interface(self):
        models = [
            MobilityInterface.create("RandomWalk", ue=self.ue),
            MobilityInterface.create("RandomWaypoint", ue=self.ue),
            MobilityInterface.create("RandomDirection", ue=self.ue),
            MobilityInterface.create("GaussMarkov", ue=self.ue),
            MobilityInterface.create("DiagonalWalk", ue=self.ue,
                                     bs=self.bs, pause_time=0),
        ]
        for model in models:
            self.assertIsInstance(model, MobilityInterface)

    # --- Неизвестная модель ---

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            MobilityInterface.create("FlightModel", ue=self.ue)

    # --- Все модели имеют метод update ---

    def test_all_models_have_update(self):
        models = [
            MobilityInterface.create("RandomWalk", ue=self.ue),
            MobilityInterface.create("RandomWaypoint", ue=self.ue),
            MobilityInterface.create("RandomDirection", ue=self.ue),
            MobilityInterface.create("GaussMarkov", ue=self.ue),
            MobilityInterface.create("DiagonalWalk", ue=self.ue,
                                     bs=self.bs, pause_time=0),
        ]
        for model in models:
            self.assertTrue(callable(getattr(model, "update", None)))


# ==============================================================================
# 3. ТЕСТЫ ВОЗВРАЩАЕМЫХ ЗНАЧЕНИЙ (все модели)
# ==============================================================================

class TestUpdateReturnTypes(unittest.TestCase):
    """
    Проверяем что update() возвращает ровно три значения корректных типов:
    (position: tuple[float,float], velocity: float >= 0, direction: float)
    """

    def setUp(self):
        reset_borders()
        MapBorders(-500, 500, -500, 500)
        self.ue  = make_ue(x=100.0, y=100.0)
        self.bs  = make_bs(x=0.0, y=0.0)
        self.time_ms = 500

    def tearDown(self):
        reset_borders()

    def _check_return(self, result, model_name):
        pos, vel, direction = result

        # position — tuple из двух float
        self.assertIsInstance(pos, tuple,
            f"{model_name}: position должна быть tuple")
        self.assertEqual(len(pos), 2,
            f"{model_name}: position должна содержать два элемента")
        self.assertIsInstance(pos[0], (int, float),
            f"{model_name}: pos[0] должен быть числом")
        self.assertIsInstance(pos[1], (int, float),
            f"{model_name}: pos[1] должен быть числом")

        # velocity — не отрицательное
        self.assertGreaterEqual(vel, 0.0,
            f"{model_name}: velocity не может быть отрицательной")

        # direction — конечное число (не NaN, не inf)
        self.assertTrue(math.isfinite(direction),
            f"{model_name}: direction должен быть конечным числом")

    def test_random_walk_return_types(self):
        model = RandomWalkModel(ue=self.ue)
        result = model.update(self.time_ms)
        self._check_return(result, "RandomWalk")

    def test_random_waypoint_return_types(self):
        model = RandomWaypointModel(ue=self.ue)
        result = model.update(self.time_ms)
        self._check_return(result, "RandomWaypoint")

    def test_random_direction_return_types(self):
        model = RandomDirectionModel(ue=self.ue)
        result = model.update(self.time_ms)
        self._check_return(result, "RandomDirection")

    def test_gauss_markov_return_types(self):
        model = GaussMarkovModel(ue=self.ue)
        result = model.update(self.time_ms)
        self._check_return(result, "GaussMarkov")

    def test_diagonal_walk_return_types(self):
        model = DiagonalWalkModel(ue=self.ue, bs=self.bs, pause_time=0)
        result = model.update(self.time_ms)
        self._check_return(result, "DiagonalWalk")


# ==============================================================================
# 4. ТЕСТЫ ГРАНИЧНЫХ ЗНАЧЕНИЙ time_ms
# ==============================================================================

class TestUpdateEdgeCases(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(-500, 500, -500, 500)
        self.ue = make_ue(x=0.0, y=0.0)
        self.bs = make_bs()

    def tearDown(self):
        reset_borders()

    def test_time_ms_zero_no_exception(self):
        """time_ms=0 не должен вызывать деление на ноль или NaN."""
        for ModelClass in [RandomWalkModel, RandomWaypointModel,
                           RandomDirectionModel, GaussMarkovModel]:
            with self.subTest(model=ModelClass.__name__):
                model = ModelClass(ue=self.ue)
                pos, vel, d = model.update(0)
                self.assertFalse(math.isnan(pos[0]),
                    f"{ModelClass.__name__}: pos[0] = NaN при time_ms=0")
                self.assertFalse(math.isnan(pos[1]),
                    f"{ModelClass.__name__}: pos[1] = NaN при time_ms=0")

    def test_large_time_ms_no_exception(self):
        """Очень большой time_ms не должен ронять модель."""
        for ModelClass in [RandomWalkModel, GaussMarkovModel]:
            with self.subTest(model=ModelClass.__name__):
                model = ModelClass(ue=self.ue)
                pos, vel, d = model.update(999999)
                self.assertFalse(math.isnan(pos[0]))

    def test_diagonal_walk_time_ms_zero(self):
        model = DiagonalWalkModel(ue=self.ue, bs=self.bs, pause_time=0)
        pos, vel, d = model.update(0)
        self.assertFalse(math.isnan(pos[0]))


# ==============================================================================
# ЗАПУСК
# ==============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
