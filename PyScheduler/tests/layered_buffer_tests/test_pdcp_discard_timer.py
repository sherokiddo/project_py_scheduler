"""
Тесты на проверку отбрасывания пакетов механизмом PDCP discard
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import GLOBALS
from TRAFFIC_MODEL import BearerInfo, UeBearersInfo

def test_packet_discard(manager, packet, qci, grant):

    bearers_info = UeBearersInfo(
        ue_id=1,
        num_bearers=3,
        active_bearers=3,
        bearers={
            1: BearerInfo(bearer_id=1, model="TestModel", qci=1, gbr=None, mbr=None, enabled=True), # 100 ms
            2: BearerInfo(bearer_id=2, model="TestModel", qci=2, gbr=None, mbr=None, enabled=True), # 150 ms
            3: BearerInfo(bearer_id=3, model="TestModel", qci=5, gbr=None, mbr=None, enabled=True), # 300 ms
        },
    )

    manager.create_ue_buffer(1, None, bearers_info)

    manager.add_packet(1, packet(
        size=250, ue_id=1, bearer_id=1, creation_time=0, qci=qci.VOIP
        ))
    
    manager.add_packet(1, packet(
        size=250, ue_id=1, bearer_id=1, creation_time=40, qci=qci.VOIP
        ))
    
    manager.add_packet(1, packet(
        size=1250, ue_id=1, bearer_id=2, creation_time=15, qci=qci.CONV_VIDEO
        ))
    
    manager.add_packet(1, packet(
        size=150, ue_id=1, bearer_id=3, creation_time=0, qci=qci.IMS
        ))
    
    # Сегментируем пакет, PDCP discard не должен его отбрасывать
    grant = grant(num_bytes=50, lcid=5)
    manager.get_packets([grant])
    
    ue_stack = manager.ue_stacks.get(1)
    lcid_1 = manager._get_lcid_from_bearer_id(1)
    lcid_2 = manager._get_lcid_from_bearer_id(2)
    lcid_3 = manager._get_lcid_from_bearer_id(3)
    rlc_entity_1 = ue_stack.rlc_entities.get(lcid_1)
    rlc_entity_2 = ue_stack.rlc_entities.get(lcid_2)
    rlc_entity_3 = ue_stack.rlc_entities.get(lcid_3)

    # ==================================================

    GLOBALS.CURRENT_TIME = 100
    manager.upd_buffers_all()

    # QCI 1
    assert len(rlc_entity_1.tx_buffer) == 2
    assert rlc_entity_1.current_tx_buffer_size == 500
    assert rlc_entity_1.packets_expired == 0

    # QCI 2
    assert len(rlc_entity_2.tx_buffer) == 1
    assert rlc_entity_2.current_tx_buffer_size == 1250
    assert rlc_entity_2.packets_expired == 0

    # QCI 5
    assert len(rlc_entity_3.tx_buffer) == 1
    assert rlc_entity_3.current_tx_buffer_size == 100
    assert rlc_entity_3.packets_expired == 0

    # ==================================================

    GLOBALS.CURRENT_TIME = 101
    manager.upd_buffers_all()

    # QCI 1
    assert len(rlc_entity_1.tx_buffer) == 1
    assert rlc_entity_1.current_tx_buffer_size == 250
    assert rlc_entity_1.packets_expired == 1

    # QCI 2
    assert len(rlc_entity_2.tx_buffer) == 1
    assert rlc_entity_2.current_tx_buffer_size == 1250
    assert rlc_entity_2.packets_expired == 0

    # QCI 5
    assert len(rlc_entity_3.tx_buffer) == 1
    assert rlc_entity_3.current_tx_buffer_size == 100
    assert rlc_entity_3.packets_expired == 0

    # ==================================================

    GLOBALS.CURRENT_TIME = 151
    manager.upd_buffers_all()

    # QCI 1
    assert len(rlc_entity_1.tx_buffer) == 0
    assert rlc_entity_1.current_tx_buffer_size == 0
    assert rlc_entity_1.packets_expired == 2

    # QCI 2
    assert len(rlc_entity_2.tx_buffer) == 1
    assert rlc_entity_2.current_tx_buffer_size == 1250
    assert rlc_entity_2.packets_expired == 0

    # QCI 5
    assert len(rlc_entity_3.tx_buffer) == 1
    assert rlc_entity_3.current_tx_buffer_size == 100
    assert rlc_entity_3.packets_expired == 0

    # ==================================================

    GLOBALS.CURRENT_TIME = 166
    manager.upd_buffers_all()

    # QCI 2
    assert len(rlc_entity_2.tx_buffer) == 0
    assert rlc_entity_2.current_tx_buffer_size == 0
    assert rlc_entity_2.packets_expired == 1

    # QCI 5
    assert len(rlc_entity_3.tx_buffer) == 1
    assert rlc_entity_3.current_tx_buffer_size == 100
    assert rlc_entity_3.packets_expired == 0

    # ==================================================

    GLOBALS.CURRENT_TIME = 300
    manager.upd_buffers_all()

    # QCI 5
    assert len(rlc_entity_3.tx_buffer) == 1
    assert rlc_entity_3.current_tx_buffer_size == 100
    assert rlc_entity_3.packets_expired == 0

    # ==================================================

    GLOBALS.CURRENT_TIME = 301
    manager.upd_buffers_all()

    # QCI 5
    assert len(rlc_entity_3.tx_buffer) == 1
    assert rlc_entity_3.current_tx_buffer_size == 100
    assert rlc_entity_3.packets_expired == 0



