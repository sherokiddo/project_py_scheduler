"""
Тесты извлечения пакетов из буферов
"""
import pytest

def test_simple_grant(manager, single_bearer, two_bearers, packet, qci, grant):
    manager.create_ue_buffer(1, None, single_bearer)
    manager.create_ue_buffer(2, None, two_bearers)

    for _ in range(7):
        assert manager.add_packet(1, packet(size=300))

    assert manager.add_packet(2, packet(size=1500, bearer_id=1, qci=qci.VOIP))
    assert manager.add_packet(2, packet(size=750, bearer_id=1, qci=qci.VOIP))

    for _ in range(10):
        assert manager.add_packet(2, packet(size=450, bearer_id=2))

    ue_stack_1 = manager.ue_stacks.get(1)
    ue_stack_2 = manager.ue_stacks.get(2)
    lcid_1 = manager._get_lcid_from_bearer_id(1)
    lcid_2 = manager._get_lcid_from_bearer_id(2)

    rlc_entity_1 = ue_stack_1.rlc_entities.get(lcid_1)
    rlc_entity_2 = ue_stack_2.rlc_entities.get(lcid_1)
    rlc_entity_3 = ue_stack_2.rlc_entities.get(lcid_2)

    assert rlc_entity_1.current_tx_buffer_size == 2100
    assert rlc_entity_2.current_tx_buffer_size == 2250
    assert rlc_entity_3.current_tx_buffer_size == 4500

    grant_1 = grant(num_bytes=600)

    packets, extracted = manager.get_packets([grant_1])
    assert extracted == 600
    assert len(packets) == 2

    assert rlc_entity_1.current_tx_buffer_size == 1500
    assert rlc_entity_1.extracted_bytes == 600

    grant_2 = grant(ue_id=2, num_bytes=1275, lcid=3)
    grant_3 = grant(ue_id=2, num_bytes=2250, lcid=4)
    grant_fake = grant(ue_id=3, num_bytes=555, lcid=3)

    # Список грантов с разными UE ID
    with pytest.raises(ValueError):
        packets, extracted = manager.get_packets([grant_2, grant_3, grant_fake])

    packets, extracted = manager.get_packets([grant_2, grant_3])

    assert extracted == 3525
    assert len(packets) == 6

    assert rlc_entity_2.current_tx_buffer_size == 975
    assert rlc_entity_2.extracted_bytes == 1275

    assert rlc_entity_3.current_tx_buffer_size == 2250
    assert rlc_entity_3.extracted_bytes == 2250
    

