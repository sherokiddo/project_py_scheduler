import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Stats_process(Enum):
    IDLE = "IDLE"
    WAIT_RETX = "WAIT_RETX"


@dataclass
class HARQProcess:
    process_id: int
    tb_size_bytes: int = 0
    rv: int = 0
    retx_count: int = 0
    state: Stats_process = Stats_process.IDLE
    last_event: str = "NONE"
    last_tti: int = -1

class HARQIface:
    enabled: bool = True
    max_retx: int = 3
    seed: Optional[int] = None
    num_processes: int = 8
    _rng: Optional[random.Random] = None
    _ue_processes: Dict[int, List[HARQProcess]] = {}
    _ue_next_pid: Dict[int, int] = {}
