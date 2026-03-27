"""
Тесты создания буферов
"""
def test_create_ue_buffer_one_bearer(manager, single_bearer):
    manager.create_ue_buffer(1, None, single_bearer)
    assert manager.ue_has_buffer(1)
    
    ue_stack = manager.ue_stacks.get(1)
    assert ue_stack.ue_id == 1
    assert len(ue_stack.rlc_entities) == 1

    lcid = manager._get_lcid_from_bearer_id(1)
    assert lcid == 3

    rlc_entity = ue_stack.rlc_entities.get(3)
    assert rlc_entity.ue_id == 1
    assert rlc_entity.lcid == 3
    assert rlc_entity.qci == 9
    assert rlc_entity.max_tx_buffer_size == 5_000

def test_create_ue_buffer_two_bearers(manager, two_bearers):
    manager.create_ue_buffer(1, None, two_bearers)
    assert manager.ue_has_buffer(1)
    
    ue_stack = manager.ue_stacks.get(1)
    assert ue_stack.ue_id == 1
    assert len(ue_stack.rlc_entities) == 2

    lcid_1 = manager._get_lcid_from_bearer_id(1)
    lcid_2 = manager._get_lcid_from_bearer_id(2)
    assert lcid_1 == 3
    assert lcid_2 == 4

    rlc_entity_1 = ue_stack.rlc_entities.get(3)
    assert rlc_entity_1.ue_id == 1
    assert rlc_entity_1.lcid == 3
    assert rlc_entity_1.qci == 9
    assert rlc_entity_1.max_tx_buffer_size == 5_000

    rlc_entity_2 = ue_stack.rlc_entities.get(4)
    assert rlc_entity_2.ue_id == 1
    assert rlc_entity_2.lcid == 4
    assert rlc_entity_2.qci == 1
    assert rlc_entity_2.max_tx_buffer_size == 5_000

def test_create_multiple_ue_buffers(manager, single_bearer):
    manager.create_ue_buffer(1, None, single_bearer)
    manager.create_ue_buffer(2, None, single_bearer)
    assert manager.ue_has_buffer(1)
    assert manager.ue_has_buffer(2)

    ue_stack_1 = manager.ue_stacks.get(1)
    assert ue_stack_1.ue_id == 1
    assert len(ue_stack_1.rlc_entities) == 1

    ue_stack_2 = manager.ue_stacks.get(2)
    assert ue_stack_2.ue_id == 2
    assert len(ue_stack_2.rlc_entities) == 1

    lcid = manager._get_lcid_from_bearer_id(1)
    assert lcid == 3

    rlc_entity_1 = ue_stack_1.rlc_entities.get(3)
    assert rlc_entity_1.ue_id == 1
    assert rlc_entity_1.lcid == 3
    assert rlc_entity_1.qci == 9
    assert rlc_entity_1.max_tx_buffer_size == 5_000

    rlc_entity_2 = ue_stack_2.rlc_entities.get(3)
    assert rlc_entity_2.ue_id == 2
    assert rlc_entity_2.lcid == 3
    assert rlc_entity_2.qci == 9
    assert rlc_entity_2.max_tx_buffer_size == 5_000
    
def test_remove_ue_buffer(manager, single_bearer):
    manager.create_ue_buffer(1, None, single_bearer)
    assert manager.ue_has_buffer(1)
    
    manager.remove_ue_buffer(1)
    assert not manager.ue_has_buffer(1)



