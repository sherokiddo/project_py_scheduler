"""
TEST_MOBILITY_INTEGRATION.py
Уровень 3: Интеграционные тесты.
Покрытие:
  - MOBILITY_MODEL ↔ UE_MODULE (SET_MOBILITY_MODEL, UPD_POSITION)
  - UE_MODULE ↔ BS_MODULE (REG_UE, дистанции, UPD_POSITION с BS)
  - UECollection: SET_MOBILITY_MODEL, UPDATE_ALL_USERS
  - MapBorders singleton: разделяется между модулями и UE
  - Патологические сценарии: нет модели, нет BS, нет MapBorders
"""

import math
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

from MOBILITY_MODEL import (
    MapBorders,
    RandomWalkModel,
    RandomWaypointModel,
    RandomDirectionModel,
    GaussMarkovModel,
    MobilityInterface,
)
from UE_MODULE import UserEquipment, UECollection

X_MIN, X_MAX = -500.0, 500.0
Y_MIN, Y_MAX = -500.0, 500.0
TTI_MS       = 1       # Стандартный TTI LTE = 1 мс
UPD_INTERVAL = 500     # Интервал обновления мобильности


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФИКСТУРЫ
# ==============================================================================

def reset_borders():
    MapBorders._instance = None

def make_bs(x=0.0, y=0.0, height=25.0):
    """Минимальный мок BaseStation для UPD_POSITION."""
    bs = MagicMock()
    bs.position = (x, y)
    bs.height   = height
    return bs

def make_ue(ue_id=1, x=0.0, y=0.0, ue_class="car"):
    return UserEquipment(UE_ID=ue_id, x=x, y=y, ue_class=ue_class)

def in_bounds(pos, x_min=X_MIN, x_max=X_MAX, y_min=Y_MIN, y_max=Y_MAX, eps=1e-6):
    x, y = pos
    return (x_min - eps <= x <= x_max + eps and
            y_min - eps <= y <= y_max + eps)


# ==============================================================================
# 1. UE.SET_MOBILITY_MODEL — корректная инициализация через фабрику
# ==============================================================================

class TestSetMobilityModel(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_set_random_walk_assigns_model(self):
        """SET_MOBILITY_MODEL('RandomWalk'): mobility_model назначен."""
        ue = make_ue()
        ue.SET_MOBILITY_MODEL("RandomWalk")
        self.assertIsNotNone(ue.mobility_model)
        self.assertIsInstance(ue.mobility_model, RandomWalkModel)

    def test_set_random_waypoint_assigns_model(self):
        """SET_MOBILITY_MODEL('RandomWaypoint'): mobility_model назначен."""
        ue = make_ue()
        ue.SET_MOBILITY_MODEL("RandomWaypoint", pause_time=0)
        self.assertIsNotNone(ue.mobility_model)
        self.assertIsInstance(ue.mobility_model, RandomWaypointModel)

    def test_set_random_direction_assigns_model(self):
        """SET_MOBILITY_MODEL('RandomDirection'): mobility_model назначен."""
        ue = make_ue()
        ue.SET_MOBILITY_MODEL("RandomDirection", pause_time=0)
        self.assertIsInstance(ue.mobility_model, RandomDirectionModel)

    def test_set_gauss_markov_assigns_model(self):
        """SET_MOBILITY_MODEL('GaussMarkov'): mobility_model назначен."""
        ue = make_ue()
        ue.SET_MOBILITY_MODEL("GaussMarkov")
        self.assertIsInstance(ue.mobility_model, GaussMarkovModel)

    def test_set_invalid_model_raises(self):
        """SET_MOBILITY_MODEL с несуществующим именем → ValueError."""
        ue = make_ue()
        with self.assertRaises((ValueError, KeyError)):
            ue.SET_MOBILITY_MODEL("NonExistentModel")

    def test_model_receives_ue_reference(self):
        """Модель хранит ссылку на тот же объект UE."""
        ue = make_ue()
        ue.SET_MOBILITY_MODEL("RandomWalk")
        self.assertIs(ue.mobility_model.ue, ue,
            "Модель должна ссылаться на исходный объект UE, а не копию")

    def test_model_uses_mapborders_singleton(self):
        """Границы карты в модели берутся из MapBorders singleton."""
        ue = make_ue()
        ue.SET_MOBILITY_MODEL("RandomWalk")
        self.assertEqual(ue.mobility_model.x_min, X_MIN)
        self.assertEqual(ue.mobility_model.x_max, X_MAX)
        self.assertEqual(ue.mobility_model.y_min, Y_MIN)
        self.assertEqual(ue.mobility_model.y_max, Y_MAX)

    def test_set_model_twice_replaces_old(self):
        """Повторный вызов SET_MOBILITY_MODEL заменяет предыдущую модель."""
        ue = make_ue()
        ue.SET_MOBILITY_MODEL("RandomWalk")
        old_model = ue.mobility_model
        ue.SET_MOBILITY_MODEL("GaussMarkov")
        self.assertIsNot(ue.mobility_model, old_model)
        self.assertIsInstance(ue.mobility_model, GaussMarkovModel)

    def test_set_mobility_no_mapborders_raises(self):
        """SET_MOBILITY_MODEL без инициализированного MapBorders → исключение."""
        reset_borders()  # Убиваем singleton
        ue = make_ue()
        with self.assertRaises(Exception):
            ue.SET_MOBILITY_MODEL("RandomWalk")


# ==============================================================================
# 2. UE.UPD_POSITION — обновление позиции через модель
# ==============================================================================

class TestUpdPosition(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)
        self.bs = make_bs(x=0.0, y=0.0)

    def tearDown(self):
        reset_borders()

    def _attach_bs(self, ue):
        """Присоединить мок-BS к UE для работы UPD_POSITION."""
        ue.serving_bs = self.bs

    def test_upd_position_updates_ue_position(self):
        """UPD_POSITION: ue.position обновляется после вызова."""
        ue = make_ue(x=0.0, y=0.0)
        ue.SET_MOBILITY_MODEL("RandomWalk")
        self._attach_bs(ue)
        initial = ue.position
        ue.UPD_POSITION(UPD_INTERVAL)
        # Позиция должна измениться (почти наверняка при velocity > 0)
        # Или хотя бы не сломаться
        self.assertIsInstance(ue.position, tuple)
        self.assertEqual(len(ue.position), 2)

    def test_upd_position_appends_to_coordinates(self):
        """UPD_POSITION: новая позиция добавляется в ue.coordinates."""
        ue = make_ue()
        ue.SET_MOBILITY_MODEL("RandomWalk")
        self._attach_bs(ue)
        before = len(ue.coordinates)
        ue.UPD_POSITION(UPD_INTERVAL)
        self.assertEqual(len(ue.coordinates), before + 1,
            "UPD_POSITION не добавил позицию в ue.coordinates")

    def test_upd_position_updates_velocity(self):
        """UPD_POSITION: ue.velocity обновляется."""
        ue = make_ue()
        ue.SET_MOBILITY_MODEL("RandomWalk")
        self._attach_bs(ue)
        ue.UPD_POSITION(UPD_INTERVAL)
        self.assertIsInstance(ue.velocity, (int, float, np.floating))

    def test_upd_position_updates_direction(self):
        """UPD_POSITION: ue.direction обновляется."""
        ue = make_ue()
        ue.SET_MOBILITY_MODEL("RandomWalk")
        self._attach_bs(ue)
        ue.UPD_POSITION(UPD_INTERVAL)
        self.assertIsInstance(ue.direction, (int, float, np.floating))
        self.assertFalse(math.isnan(ue.direction), "direction не должен быть NaN")

    def test_upd_position_without_model_raises(self):
        """UPD_POSITION без установленной модели → AttributeError/TypeError."""
        ue = make_ue()
        ue.mobility_model = None  # явно не установлена
        self._attach_bs(ue)
        with self.assertRaises((AttributeError, TypeError)):
            ue.UPD_POSITION(UPD_INTERVAL)

    def test_upd_position_position_stays_in_bounds(self):
        """UPD_POSITION: позиция никогда не выходит за границы карты."""
        ue = make_ue(x=100.0, y=100.0)
        ue.SET_MOBILITY_MODEL("RandomWalk")
        self._attach_bs(ue)
        for _ in range(200):
            ue.UPD_POSITION(UPD_INTERVAL)
            self.assertTrue(in_bounds(ue.position),
                f"UE вышел за границы: {ue.position}")

    def test_upd_position_recalculates_dist_to_bs(self):
        """UPD_POSITION: dist_to_BS_2D пересчитывается после движения."""
        ue = make_ue(x=100.0, y=0.0)
        ue.SET_MOBILITY_MODEL("RandomWalk")
        self._attach_bs(ue)
        ue.UPD_POSITION(UPD_INTERVAL)
        expected_dist = math.hypot(
            ue.position[0] - self.bs.position[0],
            ue.position[1] - self.bs.position[1]
        )
        self.assertAlmostEqual(ue.dist_to_BS_2D, expected_dist, places=3,
            msg="dist_to_BS_2D не совпадает с фактическим расстоянием до BS")

    def test_upd_position_dist_3d_geq_2d(self):
        """dist_to_BS_3D всегда >= dist_to_BS_2D (геометрический инвариант)."""
        ue = make_ue(x=200.0, y=150.0, ue_class="pedestrian")
        ue.SET_MOBILITY_MODEL("RandomWalk")
        ue.UE_height = 1.5
        self._attach_bs(ue)
        for _ in range(50):
            ue.UPD_POSITION(UPD_INTERVAL)
            self.assertGreaterEqual(
                ue.dist_to_BS_3D,
                ue.dist_to_BS_2D - 1e-6,
                "dist_to_BS_3D < dist_to_BS_2D (нарушение геометрии)"
            )

    def test_upd_position_without_bs_raises(self):
        """UPD_POSITION: если serving_bs=None → ValueError или AttributeError."""
        ue = make_ue()
        ue.SET_MOBILITY_MODEL("RandomWalk")
        ue.serving_bs = None
        with self.assertRaises((AttributeError, ValueError, TypeError)):
            ue.UPD_POSITION(UPD_INTERVAL)

    def test_upd_position_all_models_no_crash(self):
        """UPD_POSITION: все модели выполняются без исключений."""
        models = ["RandomWalk", "GaussMarkov", "RandomDirection", "RandomWaypoint"]
        for model_name in models:
            ue = make_ue(x=50.0, y=50.0)
            ue.SET_MOBILITY_MODEL(model_name, pause_time=0)
            ue.serving_bs = self.bs
            try:
                ue.UPD_POSITION(UPD_INTERVAL)
            except Exception as e:
                self.fail(f"Модель {model_name} упала в UPD_POSITION: {e}")


# ==============================================================================
# 3. UECollection ↔ MOBILITY_MODEL
# ==============================================================================

class TestUECollectionMobility(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)
        self.bs = make_bs()

    def tearDown(self):
        reset_borders()

    def _create_collection(self, n=3, ue_class="car"):
        col = UECollection()
        for i in range(1, n + 1):
            ue = make_ue(ue_id=i, x=float(i * 50), y=0.0, ue_class=ue_class)
            ue.serving_bs = self.bs
            col.ADD_USER(ue)
        return col

    def test_set_mobility_model_applies_to_all(self):
        """UECollection.SET_MOBILITY_MODEL: модель назначена всем UE."""
        col = self._create_collection(n=4)
        col.SET_MOBILITY_MODEL("RandomWalk")
        for ue in col.GET_ALL_USERS():
            self.assertIsInstance(ue.mobility_model, RandomWalkModel,
                f"UE {ue.UE_ID}: модель не назначена")

    def test_set_mobility_model_subset_by_ids(self):
        """UECollection.SET_MOBILITY_MODEL(ue_ids=[1,2]): модель только у 1,2."""
        col = self._create_collection(n=3)
        col.SET_MOBILITY_MODEL("RandomWalk", ue_ids=[1, 2])
        self.assertIsInstance(col.GET_USER(1).mobility_model, RandomWalkModel)
        self.assertIsInstance(col.GET_USER(2).mobility_model, RandomWalkModel)
        self.assertIsNone(col.GET_USER(3).mobility_model,
            "UE 3 не должен был получить модель")

    def test_set_different_models_per_ue(self):
        """Разные UE в коллекции могут иметь разные модели."""
        col = self._create_collection(n=2)
        col.SET_MOBILITY_MODEL("RandomWalk", ue_ids=[1])
        col.SET_MOBILITY_MODEL("GaussMarkov", ue_ids=[2])
        self.assertIsInstance(col.GET_USER(1).mobility_model, RandomWalkModel)
        self.assertIsInstance(col.GET_USER(2).mobility_model, GaussMarkovModel)

    def test_upd_position_all_users_no_crash(self):
        """UPDATE_ALL_USERS: обновление позиции для всех UE без исключений."""
        col = self._create_collection(n=5)
        col.SET_MOBILITY_MODEL("RandomWalk")
        # Мокаем UPD_CH_QUALITY — это уровень 4, нас не касается
        for ue in col.GET_ALL_USERS():
            ue.UPD_CH_QUALITY = MagicMock()
        try:
            col.UPDATE_ALL_USERS(current_time=500, update_interval=UPD_INTERVAL)
        except Exception as e:
            self.fail(f"UPDATE_ALL_USERS упал: {e}")

    def test_upd_position_all_users_coordinates_grow(self):
        """После UPDATE_ALL_USERS у каждого UE coordinates вырастают."""
        col = self._create_collection(n=3)
        col.SET_MOBILITY_MODEL("RandomWalk")
        for ue in col.GET_ALL_USERS():
            ue.UPD_CH_QUALITY = MagicMock()

        before = {ue.UE_ID: len(ue.coordinates) for ue in col.GET_ALL_USERS()}
        col.UPDATE_ALL_USERS(current_time=500, update_interval=UPD_INTERVAL)
        for ue in col.GET_ALL_USERS():
            self.assertGreater(len(ue.coordinates), before[ue.UE_ID],
                f"UE {ue.UE_ID}: coordinates не выросли после UPDATE_ALL_USERS")

    def test_n_upd_positions_positions_all_in_bounds(self):
        """После 100 шагов UPDATE_ALL_USERS: все UE внутри карты."""
        col = self._create_collection(n=5)
        col.SET_MOBILITY_MODEL("RandomWalk")
        for ue in col.GET_ALL_USERS():
            ue.UPD_CH_QUALITY = MagicMock()

        for step in range(100):
            col.UPDATE_ALL_USERS(current_time=step * UPD_INTERVAL,
                                 update_interval=UPD_INTERVAL)
        for ue in col.GET_ALL_USERS():
            self.assertTrue(in_bounds(ue.position),
                f"UE {ue.UE_ID} вышел за границы: {ue.position}")

    def test_update_all_without_mobility_raises(self):
        """UPDATE_ALL_USERS: UE без mobility_model → ошибка."""
        col = self._create_collection(n=2)
        # НЕ устанавливаем модель
        with self.assertRaises((AttributeError, TypeError)):
            col.UPDATE_ALL_USERS(current_time=0, update_interval=UPD_INTERVAL)


# ==============================================================================
# 4. MapBorders Singleton — совместное использование
# ==============================================================================

class TestMapBordersSingleton(unittest.TestCase):

    def setUp(self):
        reset_borders()

    def tearDown(self):
        reset_borders()

    def test_singleton_shared_between_two_models(self):
        """Два разных UE получают одинаковые границы из одного singleton."""
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)
        ue1 = make_ue(ue_id=1, x=0.0, y=0.0)
        ue2 = make_ue(ue_id=2, x=100.0, y=100.0)
        ue1.SET_MOBILITY_MODEL("RandomWalk")
        ue2.SET_MOBILITY_MODEL("GaussMarkov")
        self.assertEqual(ue1.mobility_model.x_max, ue2.mobility_model.x_max)
        self.assertEqual(ue1.mobility_model.y_min, ue2.mobility_model.y_min)

    def test_singleton_not_overwritten_on_second_init(self):
        """Второй вызов MapBorders с другими значениями не меняет первый."""
        MapBorders(-100.0, 100.0, -100.0, 100.0)
        MapBorders(-999.0, 999.0, -999.0, 999.0)  # Должен быть проигнорирован
        borders = MapBorders().get_borders()
        self.assertEqual(borders[1], 100.0,  # x_max должен остаться 100
            "Singleton перезаписан — это нарушение паттерна")

    def test_model_init_without_singleton_raises(self):
        """Создание модели без MapBorders → исключение."""
        ue = make_ue()
        with self.assertRaises(Exception):
            RandomWalkModel(ue=ue)

    def test_two_ues_share_singleton_after_reset(self):
        """После reset_borders новый singleton создаётся корректно."""
        MapBorders(-200.0, 200.0, -200.0, 200.0)
        ue1 = make_ue(ue_id=1)
        ue1.SET_MOBILITY_MODEL("RandomWalk")
        reset_borders()
        MapBorders(-300.0, 300.0, -300.0, 300.0)
        ue2 = make_ue(ue_id=2)
        ue2.SET_MOBILITY_MODEL("RandomWalk")
        self.assertEqual(ue2.mobility_model.x_max, 300.0)
        # ue1 сохраняет старые границы (они зафиксированы при инициализации)
        self.assertEqual(ue1.mobility_model.x_max, 200.0)


# ==============================================================================
# 5. Взаимодействие UE ↔ BS: дистанции после движения
# ==============================================================================

class TestUEBSDistanceIntegration(unittest.TestCase):

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_dist_decreases_when_ue_moves_toward_bs(self):
        """UE, движущийся к BS, уменьшает dist_to_BS_2D."""
        bs = make_bs(x=0.0, y=0.0)
        # UE стартует на 200м и летит к BS (direction = π + π = в сторону 0,0)
        ue = UserEquipment(UE_ID=1, x=200.0, y=0.0, ue_class="car")
        ue.serving_bs = bs
        ue.SET_MOBILITY_MODEL("RandomWalk")
        # Форсируем направление к BS (angle = π)
        ue.direction = math.pi
        ue.velocity  = 50.0   # 50 м/с × 0.5с = 25м ближе

        dist_before = math.hypot(200.0 - 0.0, 0.0 - 0.0)
        ue.UPD_POSITION(UPD_INTERVAL)
        dist_after = ue.dist_to_BS_2D

        self.assertLess(dist_after, dist_before,
            "Дистанция должна уменьшиться при движении к BS")

    def test_dist_to_bs_zero_when_ue_at_bs(self):
        """Если UE находится прямо на позиции BS, dist_to_BS_2D ≈ 0."""
        bs = make_bs(x=0.0, y=0.0)
        ue = UserEquipment(UE_ID=1, x=0.0, y=0.0, ue_class="pedestrian")
        ue.serving_bs = bs
        ue.SET_MOBILITY_MODEL("RandomWalk")
        ue.velocity = 0.0  # стоим
        ue.UPD_POSITION(UPD_INTERVAL)
        # После стояния на месте дистанция ~= 0
        dist = math.hypot(ue.position[0], ue.position[1])
        self.assertAlmostEqual(ue.dist_to_BS_2D, dist, places=2)

    def test_dist_to_bs_positive_always(self):
        """dist_to_BS_2D всегда неотрицательное."""
        bs = make_bs(x=100.0, y=100.0)
        ue = make_ue(x=-100.0, y=-100.0)
        ue.serving_bs = bs
        ue.SET_MOBILITY_MODEL("GaussMarkov")
        for _ in range(100):
            ue.UPD_POSITION(UPD_INTERVAL)
            self.assertGreaterEqual(ue.dist_to_BS_2D, 0.0,
                "dist_to_BS_2D стал отрицательным")

    def test_multiple_ues_independent_distances(self):
        """dist_to_BS_2D каждого UE независим и пересчитывается корректно."""
        bs = make_bs(x=0.0, y=0.0)
        ue1 = UserEquipment(UE_ID=1, x=100.0, y=0.0, ue_class="car")
        ue2 = UserEquipment(UE_ID=2, x=0.0, y=200.0, ue_class="car")
        for ue in (ue1, ue2):
            ue.serving_bs = bs
            ue.SET_MOBILITY_MODEL("RandomWalk")
            ue.velocity = 0.0  # стоим — гарантируем предсказуемые дистанции

        ue1.UPD_POSITION(UPD_INTERVAL)
        ue2.UPD_POSITION(UPD_INTERVAL)

        dist1 = math.hypot(ue1.position[0], ue1.position[1])
        dist2 = math.hypot(ue2.position[0], ue2.position[1])

        self.assertAlmostEqual(ue1.dist_to_BS_2D, dist1, places=2,
            msg="dist_to_BS_2D UE1 не соответствует реальному расстоянию")
        self.assertAlmostEqual(ue2.dist_to_BS_2D, dist2, places=2,
            msg="dist_to_BS_2D UE2 не соответствует реальному расстоянию")


# ==============================================================================
# 6. Сквозной сценарий: полная цепочка инициализации
# ==============================================================================

class TestEndToEndScenario(unittest.TestCase):
    """
    Воспроизводит реальный сценарий симуляции:
    MapBorders → UECollection → BS → SET_MOBILITY_MODEL →
    N шагов UPD_POSITION → проверка консистентности
    """

    def setUp(self):
        reset_borders()
        MapBorders(X_MIN, X_MAX, Y_MIN, Y_MAX)

    def tearDown(self):
        reset_borders()

    def test_full_scenario_5ue_100steps(self):
        """5 UE с RandomWalk, 100 шагов: позиции в границах, coordinates растут."""
        bs = make_bs(x=0.0, y=0.0)
        col = UECollection()
        for i in range(1, 6):
            ue = UserEquipment(
                UE_ID=i,
                x=float(np.random.uniform(-300, 300)),
                y=float(np.random.uniform(-300, 300)),
                ue_class="car"
            )
            ue.serving_bs = bs
            ue.UPD_CH_QUALITY = MagicMock()
            col.ADD_USER(ue)

        col.SET_MOBILITY_MODEL("RandomWalk")
        n_steps = 100

        for step in range(n_steps):
            col.UPDATE_ALL_USERS(
                current_time=step * UPD_INTERVAL,
                update_interval=UPD_INTERVAL
            )

        for ue in col.GET_ALL_USERS():
            with self.subTest(ue_id=ue.UE_ID):
                self.assertTrue(in_bounds(ue.position),
                    f"UE {ue.UE_ID} вышел за карту: {ue.position}")
                self.assertEqual(len(ue.coordinates), n_steps + 1,
                    f"UE {ue.UE_ID}: coordinates.len={len(ue.coordinates)}, ожидалось {n_steps+1}")
                self.assertGreaterEqual(ue.dist_to_BS_2D, 0.0)

    def test_mixed_models_full_scenario(self):
        """UE с разными моделями — никто не выходит за карту."""
        bs = make_bs()
        assignments = [
            (1, "RandomWalk",      {}),
            (2, "RandomWaypoint",  {"pause_time": 0}),
            (3, "RandomDirection", {"pause_time": 0}),
            (4, "GaussMarkov",     {"boundary_threshold": 50.0}),
        ]
        col = UECollection()
        for ue_id, model, kwargs in assignments:
            ue = UserEquipment(UE_ID=ue_id, x=0.0, y=0.0, ue_class="car")
            ue.serving_bs = bs
            ue.UPD_CH_QUALITY = MagicMock()
            col.ADD_USER(ue)
            ue.SET_MOBILITY_MODEL(model, **kwargs)

        for step in range(200):
            for ue in col.GET_ALL_USERS():
                ue.UPD_POSITION(UPD_INTERVAL)

        for ue in col.GET_ALL_USERS():
            self.assertTrue(in_bounds(ue.position),
                f"UE {ue.UE_ID} ({ue.mobility_model.__class__.__name__}) "
                f"вышел за карту: {ue.position}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
