"""Общие проверки RNG: Mobility, Channel, Traffic.

Проверки валидации намеренно требуют отклонять некорректные значения.
Их падение означает незакрытый дефект, а не причину ослаблять проверку.
BS здесь заменена простым набором входных данных; чужие модули не меняются.
Тесты не подтверждают полное соответствие математических моделей 3GPP.
"""

import inspect
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import CHANNEL_MODEL as channel
import GLOBALS
import TRAFFIC_MODEL as traffic  
import MOBILITY_MODEL as mobility
from MOBILITY_MODEL import (
    MapBorders, MobilityInterface, RandomWalkModel, RandomWaypointModel,
    RandomDirectionModel, GaussMarkovModel, DiagonalWalkModel,
)
from RNG import RandomGenerator


CONDITIONS = ("RMa", "UMa", "UMi", "RMa-SC", "UMa-SC", "UMi-SC")
ENGINES = ("MRG32k3a", "Philox")


class ControlledProvider:
    """Заглушка публичного контракта RNG с журналом вызовов."""

    def __init__(self, fraction=0.9, normal_z=1.0):
        self.fraction = fraction
        self.normal_z = normal_z
        self.calls = []

    def uniform(self, min_val=0.0, max_val=1.0, stream_offset=0):
        self.calls.append(("uniform", min_val, max_val, stream_offset))
        return min_val + self.fraction * (max_val - min_val)

    def normal(self, mean=0.0, variance=1.0, bound=float("inf"), stream_offset=0):
        self.calls.append(("normal", mean, variance, stream_offset))
        return mean + math.sqrt(variance) * self.normal_z

    def exponential(self, mean=1.0, bound=0.0, stream_offset=0):
        self.calls.append(("exponential", mean, bound, stream_offset))
        return mean

    def get_integer(self, min_val, max_val, stream_offset=0):
        self.calls.append(("get_integer", min_val, max_val, stream_offset))
        return max_val


def make_bs(tdl=False):
    return SimpleNamespace(
        frequency_Hz=3.5e9,
        frequency_GHz=3.5,
        bandwidth=5,
        rb_per_slot=25,
        antenna_gain=15.0,
        height=25.0,
        enable_tdl=tdl,
        MACROCELL_TX_POWER={5: 43.0},
        MICROCELL_TX_POWER={5: 30.0},
        registered_ues={i: SimpleNamespace(velocity=1.0) for i in range(30)},
    )


def make_model(condition="UMa", tdl=False, engine="MRG32k3a", seed=42,
               run_idx=1, macro=None, small=None, disable_sc=False, **kwargs):
    if macro is None:
        macro = RandomGenerator(seed=seed, run_idx=run_idx, engine=engine)
    if small is None:
        small = RandomGenerator(seed=seed, run_idx=run_idx,
                                base_stream_idx=100, engine=engine)
    bs = make_bs(tdl)
    if disable_sc:
        cls = channel.ChannelInterface.get_conditions()[condition]
        return cls(bs=bs, rng=macro, tdl_rng=small,
                   spatial_consistency=False, **kwargs)
    return channel.ChannelInterface(bs, macro, small).create(condition, **kwargs)


def inputs(uid=10, distance=120.0, height=1.5, ue_class="outdoor"):
    return dict(
        UE_ID=uid,
        displacement=5.0,
        d_2D=distance,
        d_2D_in=10.0 if ue_class == "indoor" else 0.0,
        d_3D=float(np.hypot(distance, 25.0 - height)),
        UE_height=height,
        ue_class=ue_class,
        current_position=np.array([distance, 0.0]),
    )


def history(condition, tdl=False, engine="MRG32k3a", others=False,
            seed=42, run_idx=1):
    model = make_model(condition, tdl, engine, seed, run_idx,
                       cond_update_period=1.0)
    results, shadow = [], []
    for step in range(3):
        GLOBALS.CURRENT_TIME = step * 10.0
        for uid in ([1, 2, 3, 4, 5, 10] if others else [10]):
            result = model.calculate_SINR(**inputs(uid, 120.0 + step * 5 + uid))
            if uid == 10:
                results.append(np.asarray(result).copy())
                if model.spatial_consistency:
                    shadow.append(model._get_channel_state(uid).last_shadow_fading)
                else:
                    shadow.append(model.SHADOW_FADING_INFO[uid]["SF_value"])
    return results, shadow


class ChannelTestCase(unittest.TestCase):
    def setUp(self):
        saved = GLOBALS.CURRENT_TIME
        self.addCleanup(setattr, GLOBALS, "CURRENT_TIME", saved)
        GLOBALS.CURRENT_TIME = 0.0


class TestChannelRNG(ChannelTestCase):
    def test_no_global_random_references(self):
        source = inspect.getsource(channel)
        self.assertNotIn("np.random", source)
        self.assertNotIn("numpy.random", source)

    def test_global_numpy_random_is_not_called(self):
        names = ("random", "normal", "uniform", "choice", "randint", "default_rng")
        def forbidden(*args, **kwargs):
            raise AssertionError("Модель обратилась к глобальному NumPy RNG")
        with patch.multiple(np.random, **{name: forbidden for name in names}):
            for condition in CONDITIONS:
                with self.subTest(condition=condition):
                    model = make_model(condition, tdl=True)
                    self.assertTrue(np.all(np.isfinite(
                        model.calculate_SINR(**inputs(height=20.0, distance=500.0)))))

    def test_injected_providers_reach_models_and_tdl(self):
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                macro, small = ControlledProvider(), ControlledProvider()
                model = make_model(condition, macro=macro, small=small)
                self.assertIs(model.rng, macro)
                self.assertIs(model.freq_fading.rng, small)

    def test_all_random_calls_use_target_ue_id(self):
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                macro, small = ControlledProvider(), ControlledProvider()
                model = make_model(condition, tdl=True, macro=macro, small=small)
                model.calculate_SINR(**inputs(uid=19, height=20.0, distance=500.0))
                self.assertTrue(macro.calls)
                self.assertTrue(small.calls)
                self.assertEqual({call[-1] for call in macro.calls + small.calls}, {19})

    def test_shadow_fading_variance_and_correlated_step(self):
        rng = ControlledProvider()
        model = make_model(macro=rng)
        self.assertEqual(model._calculate_shadow_fading(19, 0, "LOS", 4, 37), 4)
        value = model._calculate_shadow_fading(19, 37, "LOS", 4, 37)
        r = math.exp(-1)
        self.assertAlmostEqual(value, r * 4 + math.sqrt(1 - r * r) * 4)
        self.assertEqual(rng.calls, [("normal", 0.0, 16, 19)] * 2)

    def test_sc_shadow_fading_variance_and_correlated_step(self):
        rng = ControlledProvider()
        model = make_model("RMa-SC", macro=rng)
        first = model._calculate_correlated_shadow_fading_sc(np.array([0., 0.]), 6, True, 19)
        second = model._calculate_correlated_shadow_fading_sc(np.array([37., 0.]), 6, True, 19)
        r = math.exp(-1)
        self.assertEqual(first, 6)
        self.assertAlmostEqual(second, r * 6 + math.sqrt(1 - r * r) * 6)
        self.assertEqual(rng.calls, [("normal", 0.0, 36, 19)] * 2)

    def test_building_o2i_variance_for_both_profiles(self):
        for sc in (False, True):
            for profile, sigma in (("low", 4.4), ("high", 6.5)):
                with self.subTest(sc=sc, profile=profile):
                    rng = ControlledProvider()
                    model = make_model("UMa-SC" if sc else "UMa", macro=rng)
                    if sc:
                        value = model._calculate_o2i_building_loss_sc(profile, 10., 3.5, UE_ID=19)
                    else:
                        value = model._calculate_o2i_building_loss(19, "LOS", profile, 10., 3.5)
                    self.assertTrue(np.isfinite(value))
                    self.assertEqual(rng.calls, [("normal", 0.0, sigma**2, 19)])

    def test_car_o2i_mean_variance_and_cache(self):
        for sc in (False, True):
            with self.subTest(sc=sc):
                rng = ControlledProvider()
                model = make_model("UMa-SC" if sc else "UMa", macro=rng)
                method = model._calculate_o2i_car_loss_sc if sc else model._calculate_o2i_car_loss
                self.assertEqual(method(19), 14)
                self.assertEqual(method(19), 14)
                self.assertEqual(rng.calls, [("normal", 9.0, 25.0, 19)])

    def test_uma_breakpoint_choice_uses_inclusive_integer_bounds(self):
        for condition in ("UMa", "UMa-SC"):
            with self.subTest(condition=condition):
                rng = ControlledProvider(fraction=0.999)
                model = make_model(condition, macro=rng)
                value = model._calculate_breakpoint_distance(UE_ID=19, UE_height=20., d_2D=500.)
                self.assertEqual(rng.calls[-1], ("get_integer", 0, 2, 19))
                self.assertAlmostEqual(value, 4 * (25 - 18) * (20 - 18) * 3.5e9 / 3e8)

    def test_sc_los_initialization_and_transition(self):
        rng = ControlledProvider(fraction=0.9)
        model = make_model("RMa-SC", macro=rng)
        self.assertTrue(model._calculate_spatially_consistent_los(np.array([0., 0.]), 0.7, 19))
        state = model._get_channel_state(19)
        self.assertFalse(state.first_call)
        np.testing.assert_array_equal(state.last_los_position, [0., 0.])
        self.assertFalse(model._calculate_spatially_consistent_los(np.array([1000., 0.]), 0.1, 19))
        self.assertEqual({call[-1] for call in rng.calls}, {19})

    def test_sc_los_and_shadow_positions_do_not_interfere(self):
        model = make_model("RMa-SC", macro=ControlledProvider())
        model._calculate_spatially_consistent_los(np.array([0., 0.]), 0.7, 19)
        model._calculate_correlated_shadow_fading_sc(np.array([20., 0.]), 4, True, 19)
        state = model._get_channel_state(19)
        np.testing.assert_array_equal(state.last_los_position, [0., 0.])
        np.testing.assert_array_equal(state.last_sf_position, [20., 0.])

    def test_sc_o2i_cache_is_independent_for_each_ue(self):
        rng = ControlledProvider(normal_z=1)
        model = make_model("UMa-SC", macro=rng)
        self.assertEqual(model._calculate_o2i_car_loss_sc(19), 14)
        rng.normal_z = 2
        self.assertEqual(model._calculate_o2i_car_loss_sc(20), 19)
        self.assertEqual(model._calculate_o2i_car_loss_sc(19), 14)
        self.assertIsNot(model._get_channel_state(19), model._get_channel_state(20))

    def test_tdl_angles_and_phases_have_required_shape_and_bounds(self):
        model = make_model(tdl=True)
        tdl = model.freq_fading
        tdl.calculate_channel_gain(19, "LOS", "UMaModel", "outdoor")
        data = tdl.ues_info[19]
        rows = max(len(tdl.nlos_norm_delay), len(tdl.los_norm_delay))
        self.assertEqual(data["alpha"].shape, (rows, 32))
        self.assertEqual(data["phi"].shape, (rows, 32))
        base = 2 * np.pi * np.arange(1, 33) / 32
        self.assertTrue(np.all(np.abs(data["alpha"] - base) <= np.pi / 32 + 1e-12))
        self.assertTrue(np.all(data["phi"] >= -np.pi))
        self.assertTrue(np.all(data["phi"] < np.pi))
        self.assertGreaterEqual(data["phi_los"], -np.pi)
        self.assertLess(data["phi_los"], np.pi)

    def test_tdl_does_not_regenerate_cached_phases(self):
        rng = ControlledProvider()
        model = make_model(tdl=True, small=rng)
        tdl = model.freq_fading
        tdl.calculate_channel_gain(19, "NLOS", "UMaModel", "outdoor")
        count = len(rng.calls)
        GLOBALS.CURRENT_TIME = 10.
        tdl.calculate_channel_gain(19, "NLOS", "UMaModel", "outdoor")
        self.assertEqual(len(rng.calls), count)
        tdl.calculate_channel_gain(19, "LOS", "UMaModel", "outdoor")
        self.assertEqual(len(rng.calls), count + 1) 

    def test_tdl_gain_has_unit_mean_linear_power(self):
        for engine in ENGINES:
            with self.subTest(engine=engine):
                model = make_model(tdl=True, engine=engine)
                gain = model.freq_fading.calculate_channel_gain(19, "LOS", "UMaModel", "outdoor")
                self.assertEqual(gain.shape, (25,))
                self.assertTrue(np.all(np.isfinite(gain)))
                self.assertAlmostEqual(float(np.mean(10 ** (gain / 10))), 1.0)

    def test_all_models_modes_and_ue_classes_produce_finite_results(self):
        for condition in CONDITIONS:
            for engine in ENGINES:
                for tdl in (False, True):
                    for ue_class in ("outdoor", "indoor", "car"):
                        with self.subTest(condition=condition, engine=engine, tdl=tdl, ue_class=ue_class):
                            model = make_model(condition, tdl, engine)
                            result = np.asarray(model.calculate_SINR(**inputs(ue_class=ue_class)))
                            self.assertEqual(result.shape, (25,) if tdl else ())
                            self.assertTrue(np.all(np.isfinite(result)))

    def test_sc_models_work_with_sc_disabled(self):
        for condition in CONDITIONS[3:]:
            with self.subTest(condition=condition):
                model = make_model(condition, disable_sc=True)
                self.assertTrue(np.isfinite(model.calculate_SINR(**inputs())))

    def test_same_seed_and_run_reproduce_results(self):
        for condition in CONDITIONS:
            for engine in ENGINES:
                with self.subTest(condition=condition, engine=engine):
                    a, _ = history(condition, True, engine)
                    b, _ = history(condition, True, engine)
                    np.testing.assert_array_equal(a, b)

    def test_different_run_changes_random_history(self):
        for engine in ENGINES:
            with self.subTest(engine=engine):
                a, _ = history("UMa", True, engine, run_idx=1)
                b, _ = history("UMa", True, engine, run_idx=2)
                self.assertFalse(np.array_equal(a, b))

    def test_other_ues_do_not_change_target_ue(self):
        for condition in CONDITIONS:
            for engine in ENGINES:
                with self.subTest(condition=condition, engine=engine):
                    a, _ = history(condition, True, engine)
                    b, _ = history(condition, True, engine, others=True)
                    np.testing.assert_array_equal(a, b)

    def test_tdl_toggle_preserves_macro_history_with_separate_providers(self):
        for condition in CONDITIONS:
            for engine in ENGINES:
                with self.subTest(condition=condition, engine=engine):
                    _, a = history(condition, False, engine)
                    _, b = history(condition, True, engine)
                    np.testing.assert_array_equal(a, b)

    def test_model_instances_have_independent_caches(self):
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                a, b = make_model(condition), make_model(condition)
                self.assertIsNot(a.SHADOW_FADING_INFO, b.SHADOW_FADING_INFO)
                self.assertIsNot(a.CHANNEL_COND_INFO, b.CHANNEL_COND_INFO)
                self.assertIsNot(a.O2I_INFO, b.O2I_INFO)
                self.assertIsNot(a.freq_fading.ues_info, b.freq_fading.ues_info)
                if a.spatial_consistency:
                    self.assertIsNot(a._get_channel_state(19), b._get_channel_state(19))


class TestChannelValidation(ChannelTestCase):
    def reject_input(self, **changes):
        model = make_model()
        kwargs = inputs()
        kwargs.update(changes)
        with np.errstate(all="ignore"):
            with self.assertRaises(ValueError):
                model.calculate_SINR(**kwargs)

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            make_model("UNKNOWN")

    def test_missing_rng_rejected(self):
        with self.assertRaises(ValueError):
            channel.ChannelInterface(make_bs(), None)

    def test_negative_update_period_rejected(self):
        with self.assertRaises(ValueError):
            make_model(cond_update_period=-1.)

    def test_nan_update_period_rejected(self):
        with self.assertRaises(ValueError):
            make_model(cond_update_period=float("nan"))

    def test_infinite_update_period_rejected(self):
        with self.assertRaises(ValueError):
            make_model(cond_update_period=float("inf"))

    def test_invalid_nlos_profile_rejected(self):
        with self.assertRaises(ValueError):
            make_model(freq_fad_nlos_model="TDL-D")

    def test_invalid_los_profile_rejected(self):
        with self.assertRaises(ValueError):
            make_model(freq_fad_los_model="TDL-A")

    def test_invalid_delay_spread_profile_rejected(self):
        with self.assertRaises(ValueError):
            make_model(ds_profile="UNKNOWN")

    def test_nonfinite_los_angle_rejected(self):
        with self.assertRaises(ValueError):
            make_model(los_arrival_angle=float("nan"))

    def test_zero_street_width_rejected(self):
        with self.assertRaises(ValueError):
            make_model("RMa", W=0.)

    def test_negative_building_height_rejected(self):
        with self.assertRaises(ValueError):
            make_model("RMa-SC", h=-1.)

    def test_nan_building_height_rejected(self):
        with self.assertRaises(ValueError):
            make_model("RMa", h=float("nan"))

    def test_unknown_o2i_model_rejected(self):
        with self.assertRaises(ValueError):
            make_model(o2i_model="UNKNOWN")

    def test_negative_distance_rejected(self):
        self.reject_input(d_2D=-1.)

    def test_nan_distance_rejected(self):
        self.reject_input(d_2D=float("nan"))

    def test_zero_3d_distance_rejected(self):
        self.reject_input(d_3D=0.)

    def test_nan_3d_distance_rejected(self):
        self.reject_input(d_3D=float("nan"))

    def test_negative_indoor_distance_rejected(self):
        self.reject_input(d_2D_in=-1., ue_class="indoor")

    def test_nonpositive_ue_height_rejected(self):
        self.reject_input(UE_height=0.)

    def test_unsupported_uma_height_rejected_cleanly(self):
        self.reject_input(UE_height=24.)

    def test_nonfinite_ue_height_rejected_cleanly(self):
        self.reject_input(UE_height=float("nan"))

    def test_unknown_ue_class_rejected(self):
        self.reject_input(ue_class="UNKNOWN")

    def test_negative_displacement_rejected(self):
        model = make_model()
        model.calculate_SINR(**inputs())
        with np.errstate(all="ignore"):
            with self.assertRaises(ValueError):
                model.calculate_SINR(**dict(inputs(), displacement=-1.))

    def test_missing_sc_position_rejected(self):
        model = make_model("UMa-SC")
        with self.assertRaises(ValueError):
            model.calculate_SINR(**dict(inputs(), current_position=None))

    def test_wrong_sc_position_shape_rejected(self):
        model = make_model("UMa-SC")
        with self.assertRaises(ValueError):
            model.calculate_SINR(**dict(inputs(), current_position=[1., 2., 3.]))

    def test_nonfinite_sc_position_rejected(self):
        model = make_model("UMa-SC")
        with self.assertRaises(ValueError):
            model.calculate_SINR(**dict(inputs(), current_position=[float("nan"), 0.]))

    def test_negative_ue_id_rejected(self):
        self.reject_input(UE_ID=-1)

    def test_invalid_tdl_condition_rejected_cleanly(self):
        model = make_model(tdl=True)
        with self.assertRaises(ValueError):
            model.freq_fading.calculate_channel_gain(10, "UNKNOWN", "UMaModel", "outdoor")



# ============================================================================
# MOBILITY: перенесённая проверка и дополнительные свойства интеграции RNG
# ============================================================================


def make_mobility_ue(x=0.0, y=0.0, ue_class="pedestrian", uid=37):
    ue = MagicMock()
    ue.UE_ID = uid
    ue.position = (x, y)
    ue.velocity = 1.4
    ue.velocity_min = 0.5
    ue.velocity_max = 1.8
    ue.direction = 0.0
    ue.ue_class = ue_class
    ue.mean_velocity = 1.4
    return ue


def make_mobility_bs(x=0.0, y=0.0):
    bs = MagicMock()
    bs.position = (x, y)
    return bs


def reset_mobility_borders():
    MapBorders.reset()


def mobility_model(name, ue, rng):
    extra = {"bs": make_mobility_bs(), "pause_time": 0} if name == "DiagonalWalk" else {}
    return MobilityInterface.create(name, ue=ue, rng=rng, **extra)


def mobility_history(name, others=False, engine="MRG32k3a"):
    rng = RandomGenerator(seed=42, run_idx=1, engine=engine)
    ues, models = {}, {}
    ids = [38, 37] if others else [37]
    for uid in ids:
        ues[uid] = make_mobility_ue(x=100., y=100., uid=uid)
        models[uid] = mobility_model(name, ues[uid], rng)
    result = []
    for _ in range(3):
        for uid in ids:
            position, velocity, direction = models[uid].update(100)
            ues[uid].position = position
            ues[uid].velocity = velocity
            ues[uid].direction = direction
            if uid == 37:
                result.append((float(position[0]), float(position[1]), float(velocity), float(direction)))
    return result

class TestMobilityModelsRngIntegration(unittest.TestCase):

    def setUp(self):
        reset_mobility_borders()
        MapBorders(-500, 500, -500, 500)

    def tearDown(self):
        reset_mobility_borders()

    def test_all_models_use_injected_rng_with_ue_stream_offset(self):
        class RecordingRng:
            def __init__(self):
                self.calls = []

            def uniform(self, min_val=0.0, max_val=1.0, stream_offset=0):
                self.calls.append(("uniform", stream_offset))
                return (min_val + max_val) / 2

            def normal(
                self, mean=0.0, variance=1.0,
                bound=float("inf"), stream_offset=0
            ):
                self.calls.append(("normal", stream_offset))
                return mean

        model_cases = [
            (RandomWalkModel, {}, ["uniform"] * 4),
            (RandomWaypointModel, {}, ["uniform"] * 5),
            (RandomDirectionModel, {}, ["uniform"] * 4),
            (GaussMarkovModel, {}, ["uniform", "uniform", "normal", "normal"]),
            (
                DiagonalWalkModel,
                {"bs": make_mobility_bs(), "pause_time": 0},
                ["uniform", "uniform"],
            ),
        ]

        for model_class, extra_args, expected_methods in model_cases:
            with self.subTest(model=model_class.__name__):
                ue = make_mobility_ue(x=100.0, y=100.0)
                ue.UE_ID = 37
                rng = RecordingRng()

                model = model_class(ue=ue, rng=rng, **extra_args)
                model.update(100)

                self.assertIs(model.rng, rng)
                self.assertEqual(
                    [method for method, _ in rng.calls],
                    expected_methods,
                )
                self.assertTrue(
                    all(offset == 37 for _, offset in rng.calls),
                    f"{model_class.__name__} не передала UE_ID как stream_offset",
                )

    def test_factory_passes_injected_provider(self):
        for name in MobilityInterface.get_models():
            with self.subTest(model=name):
                rng = ControlledProvider()
                model = mobility_model(name, make_mobility_ue(x=100., y=100.), rng)
                self.assertIs(model.rng, rng)
                self.assertTrue(rng.calls)
                self.assertEqual({call[-1] for call in rng.calls}, {37})

    def test_models_do_not_call_global_numpy_random(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("Mobility обратилась к глобальному NumPy RNG")
        names = ("random", "normal", "uniform", "choice", "randint", "default_rng")
        with patch.multiple(np.random, **{name: forbidden for name in names}):
            for name in MobilityInterface.get_models():
                with self.subTest(model=name):
                    mobility_history(name)

    def test_same_seed_and_run_reproduce_mobility(self):
        for name in MobilityInterface.get_models():
            for engine in ENGINES:
                with self.subTest(model=name, engine=engine):
                    np.testing.assert_array_equal(
                        mobility_history(name, engine=engine),
                        mobility_history(name, engine=engine),
                    )

    def test_other_ues_do_not_change_mobility_of_target(self):
        for name in MobilityInterface.get_models():
            for engine in ENGINES:
                with self.subTest(model=name, engine=engine):
                    np.testing.assert_array_equal(
                        mobility_history(name, engine=engine),
                        mobility_history(name, others=True, engine=engine),
                    )


# ============================================================================
# TRAFFIC: целевой контракт RNG для Poisson, OnOff и MMPP.
# ============================================================================

TRAFFIC_CASES = {
    "Poisson": {"packet_rate": 20.0},
    "OnOff": {"packet_rate": 20.0, "duration_on": 1.0, "duration_off": 1.0},
    "MMPP": {
        "packet_rates": [20.0, 50.0],
        "transition_matrix": np.array([[0.0, 0.2], [0.3, 0.0]]),
    },
}


def traffic_history(test, name, others=False, engine="MRG32k3a"):
    rng = RandomGenerator(seed=42, run_idx=1, engine=engine)
    model = test.create_traffic(name, rng)
    result = []
    for end_time in (120, 240, 360):
        for uid in ([38, 37] if others else [37]):
            packets = model.generate_traffic(uid, end_time, 120)
            if uid == 37:
                result.extend((p.ue_id, int(p.size), float(p.creation_time)) for p in packets)
    return result


class TestTrafficRNGIntegration(unittest.TestCase):
    def create_traffic(self, name, rng):
        params = dict(TRAFFIC_CASES[name])
        if "transition_matrix" in params:
            params["transition_matrix"] = params["transition_matrix"].copy()
        try:
            return traffic.TrafficModelFactory.create_model(
                name, rng=rng, min_packet_size=150, max_packet_size=1500, **params
            )
        except TypeError as exc:
            if "unexpected keyword argument 'rng'" in str(exc):
                self.fail(
                    f"Traffic {name}: конструктор не принимает rng. "
                    "Интеграция RandomProvider в эту модель ещё не выполнена."
                )
            raise

    def test_no_global_random_references_in_traffic(self):
        source = inspect.getsource(traffic)
        self.assertFalse(
            "np.random" in source or "numpy.random" in source,
            "Traffic пока использует глобальный NumPy RNG",
        )

    def test_factory_passes_provider_to_traffic_models(self):
        for name in TRAFFIC_CASES:
            with self.subTest(model=name):
                rng = ControlledProvider()
                model = self.create_traffic(name, rng)
                self.assertIs(model.rng, rng)

    def test_traffic_uses_provider_with_target_ue_id(self):
        expected_streams = {
            "Poisson": {74, 75},
            "OnOff": {74},
            "MMPP": {74, 75},
        }

        for name in TRAFFIC_CASES:
            with self.subTest(model=name):
                rng = ControlledProvider()

                # Проверяем именно переданный генератор.
                with patch.object(GLOBALS, "RNG", None, create=True):
                    model = self.create_traffic(name, rng)
                    model.generate_traffic(37, 120, 120)

                self.assertTrue(rng.calls)
                self.assertEqual(
                    {call[-1] for call in rng.calls},
                    expected_streams[name],
                )

    def test_poisson_uses_milliseconds_and_correct_integer_bounds(self):
        rng = ControlledProvider()

        with patch.object(GLOBALS, "RNG", None, create=True):
            model = self.create_traffic("Poisson", rng)
            packets = model.generate_traffic(37, 1000, 120)

        self.assertIn(
            ("exponential", 50.0, 0.0, 74),
            rng.calls,
        )

        self.assertIn(
            ("get_integer", 150, 1499, 75),
            rng.calls,
        )

        self.assertEqual(len(packets), 2)
        self.assertEqual(
            [packet.creation_time for packet in packets],
            [930.0, 980.0],
        )

        for packet in packets:
            self.assertEqual(packet.ue_id, 37)
            self.assertEqual(packet.size, 1499)

    def test_same_seed_and_run_reproduce_traffic(self):
        for name in TRAFFIC_CASES:
            for engine in ENGINES:
                with self.subTest(model=name, engine=engine):
                    self.assertEqual(
                        traffic_history(self, name, engine=engine),
                        traffic_history(self, name, engine=engine),
                    )

    def test_other_ues_do_not_change_target_traffic(self):
        for name in TRAFFIC_CASES:
            with self.subTest(model=name):
                self.assertEqual(
                    traffic_history(self, name),
                    traffic_history(self, name, others=True),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
