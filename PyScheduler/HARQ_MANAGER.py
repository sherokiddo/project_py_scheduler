import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class HARQ_State(Enum):
    IDLE = "IDLE"              # свободен
    NEW_TX = "NEW_TX"          # новая передача запланирована
    WAIT_ACK = "WAIT_ACK"      # ждём ACK/NACK
    WAIT_RETX = "WAIT_RETX"    # требуется ретрансляция
    FAILED = "FAILED"          # превышен max_retx


@dataclass
class HARQProcess:
    process_id: int
    tb_size_bytes: int = 0
    rv: int = 0
    retx_count: int = 0
    state: HARQ_State = HARQ_State.IDLE
    last_event: str = "NONE"
    last_tti: int = -1

class HARQIface:
    def __init__(self, enabled=True, max_retx=3, num_processes=8, seed=None):

        self.enabled = enabled
        self.max_retx = max_retx
        self.num_processes = num_processes
        self.seed = seed

        self._rng = random.Random(seed)

        self._ue_processes = {}
        self._ue_next_pid = {}

    def create_process(self, ue, ue_id):
        self._ue_processes[ue_id] = [
            HARQProcess(process_id=i)
            for i in range(self.num_processes)]
        self._ue_next_pid[ue_id] = 0
