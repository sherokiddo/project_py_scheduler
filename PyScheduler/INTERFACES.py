from dataclasses import dataclass

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
