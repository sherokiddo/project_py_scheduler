"""
Тесты сегментирования пакетов при извлечении из буферов
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import GLOBALS

def test_packet_fragmentation(manager, single_bearer, packet, grant):
    manager.create_ue_buffer(1, None, single_bearer)
    manager.add_packet(1, packet(size=500))

    ue_stack = manager.ue_stacks.get(1)
    lcid = manager._get_lcid_from_bearer_id(1)
    rlc_entity = ue_stack.rlc_entities.get(lcid)

    assert rlc_entity.tx_buffer[0].size == 500
    assert rlc_entity.tx_buffer[0].is_fragment == False
    assert rlc_entity.tx_buffer[0].creation_time == 0
    assert len(rlc_entity.tx_buffer) == 1

    GLOBALS.CURRENT_TIME = 50
    
    grant_1 = grant(num_bytes=264)
    packets, extracted = manager.get_packets([grant_1])

    assert extracted == 264
    assert packets[0].size == 264
    assert packets[0].is_fragment == True
    assert packets[0].creation_time == 0

    assert rlc_entity.tx_buffer[0].size == 236
    assert rlc_entity.tx_buffer[0].is_fragment == True
    assert rlc_entity.tx_buffer[0].creation_time == 0
    assert len(rlc_entity.tx_buffer) == 1

    grant_2 = grant(num_bytes=235)
    packets, extracted = manager.get_packets([grant_2])

    assert extracted == 235
    assert packets[0].size == 235
    assert packets[0].is_fragment == True
    assert packets[0].creation_time == 0

    assert rlc_entity.tx_buffer[0].size == 1
    assert rlc_entity.tx_buffer[0].is_fragment == True
    assert rlc_entity.tx_buffer[0].creation_time == 0
    assert len(rlc_entity.tx_buffer) == 1

    grant_3 = grant(num_bytes=1500)
    packets, extracted = manager.get_packets([grant_3])

    assert extracted == 1
    assert packets[0].size == 1
    assert packets[0].is_fragment == True
    assert packets[0].creation_time == 0

    assert len(rlc_entity.tx_buffer) == 0

