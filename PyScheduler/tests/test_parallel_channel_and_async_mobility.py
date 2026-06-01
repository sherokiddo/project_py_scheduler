"""
Единый набор тестов для новых фич:
  1) parallel channel;
  2) async mobility;
  3) equivalence-проверки: новый snapshot-путь должен совпадать со старым sync-путём.

Запуск из папки PyScheduler:
    python -m pytest -q tests/test_parallel_channel_and_async_mobility.py

Что именно проверяется:
    - базовая математика ChannelParallel;
    - применение ChannelSnapshot к UE;
    - интеграция ChannelParallelProvider с UECollection;
    - реальный запуск channel worker-процессов;
    - equivalence старого UPD_CH_QUALITY() и нового APPLY_CHANNEL_SNAPSHOT();
    - применение MobilitySnapshot к UE;
    - интеграция MobilityAsyncProvider с UECollection;
    - реальный запуск mobility worker-процесса;
    - equivalence старого UPD_POSITION() и нового APPLY_MOBILITY_SNAPSHOT().
"""

import copy
import math

import numpy as np
import pytest

from CHANNEL_PARALLEL import (
    ChannelInput,
    ChannelParallelProvider,
    ChannelSnapshot,
    _calculate_distances_for_input,
    _calculate_one_channel_snapshot,
    _sinr_to_cqi,
)
from MOBILITY_ASYNC import (
    MobilityAsyncProvider,
    MobilitySnapshot,
    build_worker_specs,
)
from MOBILITY_MODEL import MapBorders
from UE_MODULE import UECollection, UserEquipment


# =============================================================================
# Shared helpers
# =============================================================================


class DummyBaseStationForTests:
    def __init__(self, *, position=(100.0, 50.0), height=25.0, enable_tdl=False, bandwidth=5):
        self.position = position
        self.height = height
        self.enable_tdl = enable_tdl
        self.bandwidth = bandwidth
        self.channel_model = None


class DeterministicChannelModelForTests:
    """
    Детерминированная channel-модель.

    Важно: результат зависит от входных параметров, поэтому equivalence-тесты
    реально проверяют сохранение семантики, а не просто факт вызова.
    """

    def __init__(self, bs):
        self.bs = bs

    def calculate_SINR(
        self,
        ue_id,
        displacement,
        dist_to_bs_2d,
        dist_to_bs_2d_in,
        dist_to_bs_3d,
        ue_height,
        ue_class,
    ):
        class_shift = 0.25 if ue_class == "pedestrian" else 0.5

        base = (
            10.0
            + int(ue_id) * 0.1
            + float(displacement) * 10.0
            - float(dist_to_bs_2d) * 0.001
            - float(dist_to_bs_2d_in) * 0.0005
            - float(dist_to_bs_3d) * 0.0002
            + float(ue_height) * 0.01
            + class_shift
        )

        return np.array([base, base + 1.0, base + 2.0, base + 3.0], dtype=float)


class DeterministicTdlChannelModelForTests(DeterministicChannelModelForTests):
    def calculate_SINR(
        self,
        ue_id,
        displacement,
        dist_to_bs_2d,
        dist_to_bs_2d_in,
        dist_to_bs_3d,
        ue_height,
        ue_class,
    ):
        base = super().calculate_SINR(
            ue_id,
            displacement,
            dist_to_bs_2d,
            dist_to_bs_2d_in,
            dist_to_bs_3d,
            ue_height,
            ue_class,
        )[0]

        # Достаточно RB, чтобы проверить cqi_subband при enable_tdl=True.
        return np.array([base + i * 0.1 for i in range(40)], dtype=float)


def make_channel_ue_and_bs(*, ue_id=1, enable_tdl=False, model_cls=DeterministicChannelModelForTests):
    bs = DummyBaseStationForTests(enable_tdl=enable_tdl, bandwidth=5)
    model = model_cls(bs)
    bs.channel_model = model

    ue = UserEquipment(UE_ID=ue_id, x=130.0 + ue_id, y=80.0 + ue_id)
    ue.serving_bs = bs
    ue.position = (130.0 + ue_id, 80.0 + ue_id)
    ue.velocity = 1.2
    ue.UE_height = 1.5
    ue.ue_class = "pedestrian"
    ue.is_indoor = False

    return ue, bs


def make_channel_input_from_ue(ue, step_idx=1):
    return ChannelInput(
        ue_id=int(ue.UE_ID),
        step_idx=int(step_idx),
        position=tuple(ue.position),
        velocity=float(ue.velocity),
        ue_class=str(ue.ue_class),
        is_indoor=bool(ue.is_indoor),
        indoor_boundaries=tuple(getattr(ue, "indoor_boundaries", (0, 0, 0, 0))),
        ue_height=float(ue.UE_height),
    )


def assert_channel_state_equal(sync_ue, snap_ue):
    assert snap_ue.SINR == pytest.approx(sync_ue.SINR)
    assert snap_ue.cqi == sync_ue.cqi
    assert snap_ue.cqi_subband == sync_ue.cqi_subband

    assert snap_ue.UE_height == pytest.approx(sync_ue.UE_height)
    assert snap_ue.dist_to_BS_2D == pytest.approx(sync_ue.dist_to_BS_2D)
    assert snap_ue.dist_to_BS_2D_in == pytest.approx(sync_ue.dist_to_BS_2D_in)
    assert snap_ue.dist_to_BS_2D_out == pytest.approx(sync_ue.dist_to_BS_2D_out)
    assert snap_ue.dist_to_BS_3D == pytest.approx(sync_ue.dist_to_BS_3D)

    assert snap_ue.SINR_values[-1] == pytest.approx(sync_ue.SINR_values[-1])
    assert snap_ue.CQI_values[-1] == sync_ue.CQI_values[-1]


def make_test_mobility_bs():
    return DummyBaseStationForTests(position=(0.0, 0.0), height=10.0)


def make_mobility_ue(*, ue_id=1):
    """
    UE с настоящей RandomWalk-моделью.

    Для equivalence-теста важно:
      - границы карты инициализированы;
      - seed фиксирован;
      - sync и async стартуют с одинакового состояния модели.
    """
    MapBorders.reset()
    MapBorders(0.0, 1000.0, 0.0, 1000.0)

    np.random.seed(12345)
    ue = UserEquipment(UE_ID=ue_id, x=100.0 + ue_id, y=100.0 + ue_id)
    ue.serving_bs = make_test_mobility_bs()
    ue.velocity = 1.2
    ue.direction = 0.25
    ue.SET_MOBILITY_MODEL("RandomWalk")

    return ue


def assert_mobility_state_equal(sync_ue, snap_ue):
    assert snap_ue.position[0] == pytest.approx(sync_ue.position[0])
    assert snap_ue.position[1] == pytest.approx(sync_ue.position[1])
    assert snap_ue.velocity == pytest.approx(sync_ue.velocity)
    assert snap_ue.direction == pytest.approx(sync_ue.direction)

    assert snap_ue.dist_to_BS_2D == pytest.approx(sync_ue.dist_to_BS_2D)
    assert snap_ue.dist_to_BS_2D_out == pytest.approx(sync_ue.dist_to_BS_2D_out)
    assert snap_ue.dist_to_BS_3D == pytest.approx(sync_ue.dist_to_BS_3D)
    assert snap_ue.coordinates[-1] == pytest.approx(sync_ue.coordinates[-1])


# =============================================================================
# Channel: core tests
# =============================================================================


@pytest.mark.parametrize(
    "sinr, expected",
    [
        (-100.0, 1),
        (-6.934, 1),
        (22.976, 15),
        (100.0, 15),
    ],
)
def test_channel_sinr_to_cqi_bounds(sinr, expected):
    assert _sinr_to_cqi(sinr) == expected


def test_channel_sinr_to_cqi_is_monotonic():
    values = [_sinr_to_cqi(s) for s in [-7, -3, 0, 5, 10, 15, 20, 23]]
    assert values == sorted(values)
    assert all(1 <= cqi <= 15 for cqi in values)


def test_channel_outdoor_distance_calculation_matches_expected_geometry():
    item = ChannelInput(
        ue_id=1,
        step_idx=0,
        position=(3.0, 4.0),
        velocity=1.0,
        ue_class="pedestrian",
        is_indoor=False,
        indoor_boundaries=(0.0, 0.0, 0.0, 0.0),
        ue_height=1.5,
    )

    d_2d, d_2d_in, d_2d_out, d_3d = _calculate_distances_for_input(
        item,
        bs_position=(0.0, 0.0),
        bs_height=11.5,
    )

    assert d_2d == pytest.approx(5.0)
    assert d_2d_in == pytest.approx(0.0)
    assert d_2d_out == pytest.approx(5.0)
    assert d_3d == pytest.approx(math.sqrt(5.0**2 + 10.0**2))


def test_channel_apply_snapshot_updates_all_fields():
    ue = UserEquipment(UE_ID=101, x=0.0, y=0.0)

    snapshot = ChannelSnapshot(
        ue_id=101,
        step_idx=7,
        sinr=12.5,
        cqi=9,
        cqi_subband=(7, 8, 9),
        ue_height=1.5,
        dist_to_bs_2d=100.0,
        dist_to_bs_2d_in=0.0,
        dist_to_bs_2d_out=100.0,
        dist_to_bs_3d=101.0,
    )

    ue.APPLY_CHANNEL_SNAPSHOT(snapshot)

    assert ue.SINR == pytest.approx(12.5)
    assert ue.cqi == 9
    assert ue.cqi_subband == [7, 8, 9]
    assert ue.UE_height == pytest.approx(1.5)
    assert ue.dist_to_BS_2D == pytest.approx(100.0)
    assert ue.dist_to_BS_2D_in == pytest.approx(0.0)
    assert ue.dist_to_BS_2D_out == pytest.approx(100.0)
    assert ue.dist_to_BS_3D == pytest.approx(101.0)
    assert ue.SINR_values[-1] == pytest.approx(12.5)
    assert ue.CQI_values[-1] == 9


# =============================================================================
# Channel: UECollection integration/fallback tests
# =============================================================================


def make_channel_snapshot(ue_id: int, step_idx: int, cqi: int = 10) -> ChannelSnapshot:
    return ChannelSnapshot(
        ue_id=ue_id,
        step_idx=step_idx,
        sinr=10.0 + ue_id,
        cqi=cqi,
        cqi_subband=(cqi,),
        ue_height=1.5,
        dist_to_bs_2d=50.0 + ue_id,
        dist_to_bs_2d_in=0.0,
        dist_to_bs_2d_out=50.0 + ue_id,
        dist_to_bs_3d=55.0 + ue_id,
    )


class OneBatchChannelProviderForTests:
    def __init__(self, snapshots_by_ue_id):
        self.snapshots_by_ue_id = snapshots_by_ue_id
        self.calls = []

    def calculate_batch(self, users, *, step_idx, channel_update_interval):
        user_ids = [ue.UE_ID for ue in users]
        self.calls.append((user_ids, step_idx, channel_update_interval))
        return self.snapshots_by_ue_id


class NoneChannelProviderForTests:
    def __init__(self):
        self.calls = 0

    def calculate_batch(self, users, *, step_idx, channel_update_interval):
        self.calls += 1
        return None


def test_channel_provider_is_called_once_per_step_and_snapshot_is_applied():
    collection = UECollection()
    ue1 = UserEquipment(UE_ID=1)
    ue2 = UserEquipment(UE_ID=2)
    collection.ADD_USER(ue1)
    collection.ADD_USER(ue2)

    fallback_calls = {1: 0, 2: 0}
    ue1.UPD_CH_QUALITY = lambda interval: fallback_calls.__setitem__(1, fallback_calls[1] + 1)
    ue2.UPD_CH_QUALITY = lambda interval: fallback_calls.__setitem__(2, fallback_calls[2] + 1)

    provider = OneBatchChannelProviderForTests({1: make_channel_snapshot(1, step_idx=3, cqi=11)})
    collection.SET_CHANNEL_PROVIDER(provider)

    collection.UPDATE_ALL_USERS(
        current_time=30,
        update_interval=10,
        mobility_update_interval=10_000,
        channel_update_interval=10,
    )

    assert provider.calls == [([1, 2], 3, 10)]

    assert ue1.cqi == 11
    assert ue1.SINR == 11.0
    assert fallback_calls[1] == 0

    # Для UE2 snapshot отсутствует — должен сработать старый fallback.
    assert fallback_calls[2] == 1


def test_channel_provider_none_result_falls_back_for_all_users():
    collection = UECollection()
    ue1 = UserEquipment(UE_ID=1)
    ue2 = UserEquipment(UE_ID=2)
    collection.ADD_USER(ue1)
    collection.ADD_USER(ue2)

    fallback_calls = {1: 0, 2: 0}
    ue1.UPD_CH_QUALITY = lambda interval: fallback_calls.__setitem__(1, fallback_calls[1] + 1)
    ue2.UPD_CH_QUALITY = lambda interval: fallback_calls.__setitem__(2, fallback_calls[2] + 1)

    provider = NoneChannelProviderForTests()
    collection.SET_CHANNEL_PROVIDER(provider)

    collection.UPDATE_ALL_USERS(
        current_time=40,
        update_interval=10,
        mobility_update_interval=10_000,
        channel_update_interval=10,
    )

    assert provider.calls == 1
    assert fallback_calls == {1: 1, 2: 1}


# =============================================================================
# Channel: process smoke/equivalence tests
# =============================================================================


def test_channel_parallel_provider_returns_snapshots_from_worker_processes():
    users = [
        UserEquipment(UE_ID=1, x=3.0, y=4.0),
        UserEquipment(UE_ID=2, x=6.0, y=8.0),
    ]

    bs = DummyBaseStationForTests(position=(0.0, 0.0), height=10.0, enable_tdl=False, bandwidth=5)
    channel_model = DeterministicChannelModelForTests(bs)
    bs.channel_model = channel_model

    for ue in users:
        ue.serving_bs = bs
        ue.velocity = 1.0
        ue.UE_height = 1.5

    provider = ChannelParallelProvider(
        channel_model,
        workers=2,
        timeout_s=5.0,
        seed=123,
        verbose=False,
    )

    provider.start()
    try:
        batch = provider.calculate_batch(
            users,
            step_idx=1,
            channel_update_interval=10,
        )
    finally:
        provider.shutdown()

    assert batch is not None
    assert set(batch.keys()) == {1, 2}
    assert provider.steps_processed == 1
    assert provider.ue_snapshots_processed == 2


def test_channel_single_snapshot_path_matches_old_upd_ch_quality_outdoor_no_tdl():
    sync_ue, bs = make_channel_ue_and_bs(ue_id=11, enable_tdl=False)
    snap_ue = copy.deepcopy(sync_ue)
    snap_ue.serving_bs = bs

    sync_ue.UPD_CH_QUALITY(channel_update_interval=10)

    snapshot = _calculate_one_channel_snapshot(
        channel_model=bs.channel_model,
        item=make_channel_input_from_ue(snap_ue, step_idx=1),
        channel_update_interval=10,
    )
    snap_ue.APPLY_CHANNEL_SNAPSHOT(snapshot)

    assert_channel_state_equal(sync_ue, snap_ue)


def test_channel_single_snapshot_path_matches_old_upd_ch_quality_with_tdl_subbands():
    sync_ue, bs = make_channel_ue_and_bs(
        ue_id=12,
        enable_tdl=True,
        model_cls=DeterministicTdlChannelModelForTests,
    )
    snap_ue = copy.deepcopy(sync_ue)
    snap_ue.serving_bs = bs

    sync_ue.UPD_CH_QUALITY(channel_update_interval=10)

    snapshot = _calculate_one_channel_snapshot(
        channel_model=bs.channel_model,
        item=make_channel_input_from_ue(snap_ue, step_idx=1),
        channel_update_interval=10,
    )
    snap_ue.APPLY_CHANNEL_SNAPSHOT(snapshot)

    assert_channel_state_equal(sync_ue, snap_ue)
    assert len(sync_ue.cqi_subband) > 0


def test_channel_parallel_provider_batch_matches_old_upd_ch_quality_for_multiple_users():
    users_sync = []
    users_parallel = []

    shared_bs = None
    for ue_id in range(1, 7):
        ue, bs = make_channel_ue_and_bs(ue_id=ue_id, enable_tdl=False)
        if shared_bs is None:
            shared_bs = bs
        ue.serving_bs = shared_bs

        users_sync.append(copy.deepcopy(ue))
        users_sync[-1].serving_bs = shared_bs

        users_parallel.append(copy.deepcopy(ue))
        users_parallel[-1].serving_bs = shared_bs

    for ue in users_sync:
        ue.UPD_CH_QUALITY(channel_update_interval=10)

    provider = ChannelParallelProvider(
        shared_bs.channel_model,
        workers=2,
        timeout_s=5.0,
        seed=42,
        verbose=False,
    )

    provider.start()
    try:
        batch = provider.calculate_batch(
            users_parallel,
            step_idx=1,
            channel_update_interval=10,
        )
    finally:
        provider.shutdown()

    assert batch is not None
    assert set(batch.keys()) == {ue.UE_ID for ue in users_parallel}

    for ue in users_parallel:
        ue.APPLY_CHANNEL_SNAPSHOT(batch[ue.UE_ID])

    for sync_ue, parallel_ue in zip(users_sync, users_parallel):
        assert_channel_state_equal(sync_ue, parallel_ue)

    assert provider.steps_processed == 1
    assert provider.ue_snapshots_processed == len(users_parallel)


# =============================================================================
# Async mobility: core/integration/equivalence tests
# =============================================================================


def test_mobility_apply_snapshot_updates_position_velocity_direction_and_distances():
    ue = UserEquipment(UE_ID=501, x=0.0, y=0.0)
    ue.serving_bs = make_test_mobility_bs()
    ue.UE_height = 1.5

    snapshot = MobilitySnapshot(
        ue_id=501,
        step_idx=3,
        position=(3.0, 4.0),
        velocity=2.5,
        direction=0.75,
    )

    ue.APPLY_MOBILITY_SNAPSHOT(snapshot)

    assert ue.position == pytest.approx((3.0, 4.0))
    assert ue.velocity == pytest.approx(2.5)
    assert ue.direction == pytest.approx(0.75)
    assert ue.coordinates[-1] == pytest.approx((3.0, 4.0))
    assert ue.dist_to_BS_2D == pytest.approx(5.0)
    assert ue.dist_to_BS_2D_out == pytest.approx(5.0)
    assert ue.dist_to_BS_3D == pytest.approx(math.sqrt(5.0**2 + (10.0 - 1.5) ** 2))


class OneBatchMobilityProviderForTests:
    def __init__(self, snapshots_by_ue_id):
        self.snapshots_by_ue_id = snapshots_by_ue_id
        self.calls = []

    def get_batch(self, step_idx):
        self.calls.append(step_idx)
        return self.snapshots_by_ue_id


class NoneMobilityProviderForTests:
    def __init__(self):
        self.calls = 0

    def get_batch(self, step_idx):
        self.calls += 1
        return None


def test_mobility_provider_is_called_once_per_step_and_snapshot_is_applied():
    collection = UECollection()
    ue1 = UserEquipment(UE_ID=1, x=0.0, y=0.0)
    ue2 = UserEquipment(UE_ID=2, x=0.0, y=0.0)

    for ue in (ue1, ue2):
        ue.serving_bs = make_test_mobility_bs()
        ue.UE_height = 1.5

    collection.ADD_USER(ue1)
    collection.ADD_USER(ue2)

    fallback_calls = {1: 0, 2: 0}
    ue1.UPD_POSITION = lambda interval: fallback_calls.__setitem__(1, fallback_calls[1] + 1)
    ue2.UPD_POSITION = lambda interval: fallback_calls.__setitem__(2, fallback_calls[2] + 1)

    provider = OneBatchMobilityProviderForTests(
        {
            1: MobilitySnapshot(
                ue_id=1,
                step_idx=2,
                position=(10.0, 20.0),
                velocity=1.5,
                direction=0.2,
            )
        }
    )
    collection.SET_MOBILITY_PROVIDER(provider)

    collection.UPDATE_ALL_USERS(
        current_time=20,
        update_interval=10,
        mobility_update_interval=10,
        channel_update_interval=10_000,
    )

    assert provider.calls == [2]

    assert ue1.position == pytest.approx((10.0, 20.0))
    assert ue1.velocity == pytest.approx(1.5)
    assert ue1.direction == pytest.approx(0.2)
    assert fallback_calls[1] == 0

    # Для UE2 snapshot отсутствует — должен сработать старый fallback.
    assert fallback_calls[2] == 1


def test_mobility_provider_none_result_falls_back_for_all_users():
    collection = UECollection()
    ue1 = UserEquipment(UE_ID=1)
    ue2 = UserEquipment(UE_ID=2)
    collection.ADD_USER(ue1)
    collection.ADD_USER(ue2)

    fallback_calls = {1: 0, 2: 0}
    ue1.UPD_POSITION = lambda interval: fallback_calls.__setitem__(1, fallback_calls[1] + 1)
    ue2.UPD_POSITION = lambda interval: fallback_calls.__setitem__(2, fallback_calls[2] + 1)

    provider = NoneMobilityProviderForTests()
    collection.SET_MOBILITY_PROVIDER(provider)

    collection.UPDATE_ALL_USERS(
        current_time=30,
        update_interval=10,
        mobility_update_interval=10,
        channel_update_interval=10_000,
    )

    assert provider.calls == 1
    assert fallback_calls == {1: 1, 2: 1}


def test_mobility_build_worker_specs_does_not_keep_live_ue_reference():
    ue = make_mobility_ue(ue_id=10)

    specs = build_worker_specs([ue])

    assert len(specs) == 1
    assert specs[0].ue_state.ue_id == 10
    assert specs[0].model_class_name == ue.mobility_model.__class__.__name__
    assert "ue" not in specs[0].model_state


def test_mobility_async_provider_returns_snapshots_from_worker_process():
    ue = make_mobility_ue(ue_id=20)

    provider = MobilityAsyncProvider(
        mobility_interval_ms=10,
        sim_duration_ms=20,
        prefetch_steps=4,
        cache_steps=8,
        snapshot_timeout_ms=1000,
        seed=777,
        verbose=False,
    )

    provider.start([ue])
    try:
        batch = provider.get_batch(0)
    finally:
        provider.shutdown()

    assert batch is not None
    assert set(batch.keys()) == {20}

    snap = batch[20]
    assert snap.ue_id == 20
    assert snap.step_idx == 0
    assert isinstance(snap.position, tuple)
    assert len(snap.position) == 2
    assert isinstance(snap.velocity, float)
    assert isinstance(snap.direction, float)
    assert provider.steps_received >= 1


def test_mobility_async_snapshot_matches_old_upd_position_for_first_step():
    """
    Equivalence-тест для mobility:
      старый путь: ue.UPD_POSITION()
      новый путь: MobilityAsyncProvider -> MobilitySnapshot -> APPLY_MOBILITY_SNAPSHOT()

    Для RandomWalk фиксируем seed прямо перед шагом, поэтому worker и sync-путь
    используют одинаковую последовательность np.random.
    """
    base_ue = make_mobility_ue(ue_id=30)

    sync_ue = copy.deepcopy(base_ue)
    sync_ue.serving_bs = make_test_mobility_bs()

    async_ue = copy.deepcopy(base_ue)
    async_ue.serving_bs = make_test_mobility_bs()

    np.random.seed(777)
    sync_ue.UPD_POSITION(update_interval=10)

    provider = MobilityAsyncProvider(
        mobility_interval_ms=10,
        sim_duration_ms=10,
        prefetch_steps=4,
        cache_steps=8,
        snapshot_timeout_ms=1000,
        seed=777,
        verbose=False,
    )

    provider.start([async_ue])
    try:
        batch = provider.get_batch(0)
    finally:
        provider.shutdown()

    assert batch is not None
    assert 30 in batch

    async_ue.APPLY_MOBILITY_SNAPSHOT(batch[30])

    assert_mobility_state_equal(sync_ue, async_ue)
