import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from INTERFACES import (  # noqa: E402
    AllocationGrant,
    AllocationResult,
    BufferState,
    ChannelState,
    GeneratedPacket,
    GeneratedPackets,
    MetricsSnapshot,
    PositionState,
    SchedulerUserView,
)


def test_channel_state_is_immutable_snapshot():
    subband_cqi = [8, 9, 10]

    state = ChannelState(
        ue_id=1,
        sinr=12.5,
        cqi=9,
        pathloss=103.2,
        cqi_subband=tuple(subband_cqi),
    )
    subband_cqi.append(11)

    assert state.cqi_subband == (8, 9, 10)
    with pytest.raises(FrozenInstanceError):
        state.cqi = 10


def test_generated_packets_group_packet_snapshots_by_ue():
    packet = GeneratedPacket(
        ue_id=1,
        size=512,
        creation_time=25,
        priority=2,
        qci=9,
        bearer_id=1,
    )

    generated = GeneratedPackets(ue_id=1, packets=(packet,))

    assert generated.ue_id == 1
    assert generated.packets == (packet,)


def test_scheduler_user_view_combines_channel_and_buffer_data():
    channel = ChannelState(
        ue_id=1,
        sinr=11.0,
        cqi=8,
        pathloss=98.5,
        cqi_subband=(7, 8),
    )
    buffer = BufferState(ue_id=1, buffer_size=4096, timestamp=100)

    view = SchedulerUserView(
        ue_id=channel.ue_id,
        cqi=channel.cqi,
        buffer_size=buffer.buffer_size,
        sinr=channel.sinr,
        pathloss=channel.pathloss,
        cqi_subband=channel.cqi_subband,
    )

    assert view == SchedulerUserView(
        ue_id=1,
        cqi=8,
        buffer_size=4096,
        sinr=11.0,
        pathloss=98.5,
        cqi_subband=(7, 8),
    )


def test_metrics_snapshot_collects_small_module_outputs():
    position = PositionState(ue_id=1, x=10.0, y=20.0, velocity=1.5, direction=0.25)
    channel = ChannelState(ue_id=1, sinr=10.0, cqi=7, pathloss=100.0)
    buffer = BufferState(ue_id=1, buffer_size=2048, timestamp=5)
    allocation = AllocationResult(
        tti=5,
        grants=(
            AllocationGrant(
                ue_id=1,
                rb_indices=(0, 1, 2),
                transmitted_bytes=1500,
            ),
        ),
    )

    snapshot = MetricsSnapshot(
        tti=5,
        throughput=120.5,
        fairness=1.0,
        rb_allocated=3,
        buffer_size=2048,
        cqi_avg=7.0,
        positions=(position,),
        channels=(channel,),
        buffers=(buffer,),
        allocations=(allocation,),
    )

    assert snapshot.positions == (position,)
    assert snapshot.channels == (channel,)
    assert snapshot.allocations[0].grants[0].rb_indices == (0, 1, 2)
