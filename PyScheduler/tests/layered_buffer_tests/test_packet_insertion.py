"""
Тесты добавления пакетов в буферы
"""
import pytest

def test_add_single_packet_one_bearer(manager, single_bearer, packet):
    manager.create_ue_buffer(1, None, single_bearer)
    p = packet(size=200)
    assert manager.add_packet(1, p)
    assert manager.current_total_size == 200

    ue_stack = manager.ue_stacks.get(1)
    lcid = manager._get_lcid_from_bearer_id(1)
    rlc_entity = ue_stack.rlc_entities.get(lcid)
    
    assert rlc_entity.current_tx_buffer_size == 200
    assert rlc_entity.packets_added == 1
    assert rlc_entity.packets_dropped == 0
    assert rlc_entity.packets_expired == 0
    assert rlc_entity.extracted_bytes == 0

def test_add_single_packet_two_bearers(manager, two_bearers, packet):
    manager.create_ue_buffer(1, None, two_bearers)
    p = packet(size=200)
    assert manager.add_packet(1, p)
    assert manager.current_total_size == 200

    ue_stack = manager.ue_stacks.get(1)
    lcid_1 = manager._get_lcid_from_bearer_id(1)
    rlc_entity_1 = ue_stack.rlc_entities.get(lcid_1)

    assert rlc_entity_1.current_tx_buffer_size == 200
    assert rlc_entity_1.packets_added == 1
    assert rlc_entity_1.packets_dropped == 0
    assert rlc_entity_1.packets_expired == 0
    assert rlc_entity_1.extracted_bytes == 0

    lcid_2 = manager._get_lcid_from_bearer_id(2)
    rlc_entity_2 = ue_stack.rlc_entities.get(lcid_2)

    assert rlc_entity_2.current_tx_buffer_size == 0
    assert rlc_entity_2.packets_added == 0
    assert rlc_entity_2.packets_dropped == 0
    assert rlc_entity_2.packets_expired == 0
    assert rlc_entity_2.extracted_bytes == 0

def test_add_multiple_packets_one_bearer(manager, single_bearer, packet):
    manager.create_ue_buffer(1, None, single_bearer)
    for _ in range(5):
        manager.add_packet(1, packet(size=100))
    assert manager.current_total_size == 500

    ue_stack = manager.ue_stacks.get(1)
    lcid = manager._get_lcid_from_bearer_id(1)
    rlc_entity = ue_stack.rlc_entities.get(lcid)
    
    assert rlc_entity.current_tx_buffer_size == 500
    assert rlc_entity.packets_added == 5
    assert rlc_entity.packets_dropped == 0
    assert rlc_entity.packets_expired == 0
    assert rlc_entity.extracted_bytes == 0

def test_add_multiple_packets_two_bearers(manager, two_bearers, packet, qci):
    manager.create_ue_buffer(1, None, two_bearers)
    for _ in range(5):
        manager.add_packet(1, packet(size=100, ue_id=1, bearer_id=1, qci=qci.DEFAULT))

    assert manager.current_total_size == 500

    for _ in range(10):
        manager.add_packet(1, packet(size=250, ue_id=1, bearer_id=2, qci=qci.VOIP))

    assert manager.current_total_size == 3000

    with pytest.raises(ValueError):
        manager.add_packet(1, packet(size=250, ue_id=1, bearer_id=3, qci=qci.CONV_VIDEO))

    ue_stack_1 = manager.ue_stacks.get(1)
    lcid_1 = manager._get_lcid_from_bearer_id(1)
    rlc_entity_1 = ue_stack_1.rlc_entities.get(lcid_1)
    
    assert rlc_entity_1.current_tx_buffer_size == 500
    assert rlc_entity_1.packets_added == 5
    assert rlc_entity_1.packets_dropped == 0
    assert rlc_entity_1.packets_expired == 0
    assert rlc_entity_1.extracted_bytes == 0

    lcid_2 = manager._get_lcid_from_bearer_id(2)
    rlc_entity_2 = ue_stack_1.rlc_entities.get(lcid_2)
    
    assert rlc_entity_2.current_tx_buffer_size == 2500
    assert rlc_entity_2.packets_added == 10
    assert rlc_entity_2.packets_dropped == 0
    assert rlc_entity_2.packets_expired == 0
    assert rlc_entity_2.extracted_bytes == 0

def test_add_packet_without_buffer(manager, packet):
    with pytest.raises(ValueError):
        manager.add_packet(1, packet())