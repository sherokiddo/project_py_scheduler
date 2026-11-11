"""
Тесты на переполнения буферов
"""
def test_per_ue_limit(manager, single_bearer, packet):
    manager.create_ue_buffer(1, None, single_bearer)
    manager.create_ue_buffer(2, None, single_bearer)
    for _ in range(5):
        assert manager.add_packet(1, packet(size=1000))

    ue_stack = manager.ue_stacks.get(1)
    lcid = manager._get_lcid_from_bearer_id(1)
    rlc_entity = ue_stack.rlc_entities.get(lcid)

    assert rlc_entity.current_tx_buffer_size == 5000
    assert rlc_entity.packets_added == 5
    assert rlc_entity.packets_dropped == 0

    assert manager.add_packet(1, packet(size=100)) is False

    assert rlc_entity.current_tx_buffer_size == 5000
    assert rlc_entity.packets_added == 5
    assert rlc_entity.packets_dropped == 1

    for _ in range(20):
        assert manager.add_packet(1, packet(size=1000)) is False

    assert rlc_entity.current_tx_buffer_size == 5000
    assert rlc_entity.packets_added == 5
    assert rlc_entity.packets_dropped == 21

def test_global_limit(manager, single_bearer, packet):
    manager.create_ue_buffer(1, None, single_bearer)
    manager.create_ue_buffer(2, None, single_bearer)
    manager.create_ue_buffer(3, None, single_bearer)
    for _ in range(5):
        assert manager.add_packet(1, packet(size=1000))

    ue_stack_1 = manager.ue_stacks.get(1)
    lcid = manager._get_lcid_from_bearer_id(1)
    rlc_entity_1 = ue_stack_1.rlc_entities.get(lcid)

    assert rlc_entity_1.current_tx_buffer_size == 5000
    assert rlc_entity_1.packets_added == 5
    assert rlc_entity_1.packets_dropped == 0
    assert manager.global_packets_dropped == 0
        
    for _ in range(5):
        assert manager.add_packet(2, packet(size=1000))

    ue_stack_2 = manager.ue_stacks.get(2)
    rlc_entity_2 = ue_stack_2.rlc_entities.get(lcid)

    assert rlc_entity_2.current_tx_buffer_size == 5000
    assert rlc_entity_2.packets_added == 5
    assert rlc_entity_2.packets_dropped == 0
    assert manager.global_packets_dropped == 0
        
    assert manager.add_packet(3, packet(size=100)) is False

    ue_stack_3 = manager.ue_stacks.get(3)
    rlc_entity_3 = ue_stack_3.rlc_entities.get(lcid)

    assert rlc_entity_3.current_tx_buffer_size == 0
    assert rlc_entity_3.packets_added == 0
    assert rlc_entity_3.packets_dropped == 0
    assert manager.global_packets_dropped == 1

    for _ in range(15):
        assert manager.add_packet(3, packet(size=1500)) is False

    assert manager.global_packets_dropped == 16

    


    