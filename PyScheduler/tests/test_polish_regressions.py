import math
from unittest.mock import MagicMock

import numpy as np

from MOBILITY_MODEL import MapBorders, RandomDirectionModel, GaussMarkovModel
from SCHEDULER import SchedulerInterface
from TRAFFIC_MODEL import Packet, PoissonModel


def _reset_borders():
    MapBorders._instance = None
    MapBorders(-500.0, 500.0, -500.0, 500.0)


def _make_ue(x=0.0, y=0.0, velocity=10.0):
    ue = MagicMock()
    ue.position = (x, y)
    ue.velocity = velocity
    ue.velocity_min = 2.0
    ue.velocity_max = 16.7
    ue.direction = 0.0
    ue.mean_velocity = velocity
    return ue


def test_poisson_fast_path_keeps_packets_ordered_and_in_interval():
    state = np.random.get_state()
    try:
        np.random.seed(123)
        model = PoissonModel(packet_rate=5000, min_packet_size=150, max_packet_size=1500)
        packets = model.generate_traffic(ue_id=7, current_time=100, update_interval=1)
    finally:
        np.random.set_state(state)

    assert all(isinstance(pkt, Packet) for pkt in packets)
    assert all(99 <= pkt.creation_time <= 100 for pkt in packets)
    assert [pkt.creation_time for pkt in packets] == sorted(pkt.creation_time for pkt in packets)
    assert all(150 <= pkt.size < 1500 for pkt in packets)
    assert all(pkt.deadline == pkt.creation_time + pkt.ttl_ms for pkt in packets)


def test_random_direction_non_first_direction_stays_in_half_circle():
    _reset_borders()
    model = RandomDirectionModel(ue=_make_ue(), pause_time=0)
    for _ in range(100):
        _, _, direction, _, _ = model._choose_new_direction(
            current_position=(0.0, 0.0),
            velocity_min=2.0,
            velocity_max=16.7,
            is_first_move=False,
        )
        assert 0.0 <= direction <= math.pi


def test_gauss_markov_velocity_is_clamped_non_negative():
    _reset_borders()
    model = GaussMarkovModel(ue=_make_ue(velocity=0.0), alpha=0.0)
    model.mean_velocity = 0.0

    state = np.random.get_state()
    try:
        np.random.seed(42)
        for _ in range(100):
            pos, velocity, direction = model.update(1)
            model.ue.position = pos
            model.ue.velocity = velocity
            model.ue.direction = direction
            assert velocity >= 0.0
            assert math.isfinite(direction)
    finally:
        np.random.set_state(state)


def test_scheduler_registry_keeps_fd_rr_alias():
    grid = MagicMock()
    grid.bandwidth = 5
    grid.rb_per_slot = 25
    grid.GET_RBG_SIZE.return_value = 2
    bs = MagicMock()

    sched = SchedulerInterface.create("FD_RR", grid, bs, verbose=False, enable_window=False)
    assert sched.__class__.__name__ == "FDxFairGreedyScheduler"
