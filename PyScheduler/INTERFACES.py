from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class PositionState:
    ue_id: int
    x: float
    y: float
    velocity: float
    direction: float


@dataclass(frozen=True, slots=True)
class ChannelQualityState:
    ue_id: int
    sinr: float
    cqi: int
    cqi_subband: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ChannelState:
    ue_id: int
    sinr: float
    cqi: int
    pathloss: float
    cqi_subband: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class GeneratedPacket:
    ue_id: int
    size: int
    creation_time: int
    priority: int = 0
    qci: Optional[int] = None
    bearer_id: Optional[int] = None


@dataclass(frozen=True, slots=True)
class GeneratedPackets:
    ue_id: int
    packets: tuple[GeneratedPacket, ...]


@dataclass(frozen=True, slots=True)
class BufferState:
    ue_id: int
    buffer_size: int
    timestamp: int
    lcid: Optional[int] = None
    qci: Optional[int] = None
    priority: int = 0
    hol_delay: Optional[int] = None


@dataclass(frozen=True, slots=True)
class SchedulerUserView:
    ue_id: int
    cqi: int
    buffer_size: int
    sinr: float = 0.0
    pathloss: float = 0.0
    cqi_subband: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class AllocationGrant:
    ue_id: int
    rb_indices: tuple[int, ...]
    transmitted_bytes: int = 0


@dataclass(frozen=True, slots=True)
class AllocationResult:
    tti: int
    grants: tuple[AllocationGrant, ...]


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    tti: int
    throughput: float = 0.0
    fairness: float = 0.0
    rb_allocated: int = 0
    buffer_size: int = 0
    cqi_avg: float = 0.0
    positions: tuple[PositionState, ...] = ()
    channels: tuple[ChannelState, ...] = ()
    buffers: tuple[BufferState, ...] = ()
    allocations: tuple[AllocationResult, ...] = ()