"""
Тесты на проверку работы Logical Channel Multiplexing
"""

def test_logical_channel_multiplexing(manager, scheduler, packet, qci):

    scheduler.lte_grid.bs.use_simple_buffer = False

    bearers_info = {
        "bearers": {
            1: {"bearer_id": 1, "qci": 1},
            2: {"bearer_id": 2, "qci": 2},
            3: {"bearer_id": 3, "qci": 5},
        }
    }

    manager.create_ue_buffer(1, None, bearers_info)

    for _ in range(10):
        manager.add_packet(1, packet(
            size=250, ue_id=1, bearer_id=1, creation_time=0, qci=qci.VOIP
        ))

    manager.add_packet(1, packet(
        size=1250, ue_id=1, bearer_id=2, creation_time=0, qci=qci.CONV_VIDEO
        ))

    manager.add_packet(1, packet(
        size=150, ue_id=1, bearer_id=3, creation_time=0, qci=qci.IMS
    ))

    buffer_status_list = manager.get_buffer_status(1)

    assert len(buffer_status_list) == 3

    # Buffer Status для QCI 1 (LCID 3)
    assert buffer_status_list[0].ue_id == 1
    assert buffer_status_list[0].buffer_size == 2500
    assert buffer_status_list[0].timestamp == 0
    assert buffer_status_list[0].lcid == 3
    assert buffer_status_list[0].qci == 1
    assert buffer_status_list[0].priority == 2
    assert buffer_status_list[0].hol_delay == 0

    # Buffer Status для QCI 2 (LCID 4)
    assert buffer_status_list[1].ue_id == 1
    assert buffer_status_list[1].buffer_size == 1250
    assert buffer_status_list[1].timestamp == 0
    assert buffer_status_list[1].lcid == 4
    assert buffer_status_list[1].qci == 2
    assert buffer_status_list[1].priority == 4
    assert buffer_status_list[1].hol_delay == 0

    # Buffer Status для QCI 5 (LCID 5)
    assert buffer_status_list[2].ue_id == 1
    assert buffer_status_list[2].buffer_size == 150
    assert buffer_status_list[2].timestamp == 0
    assert buffer_status_list[2].lcid == 5
    assert buffer_status_list[2].qci == 5
    assert buffer_status_list[2].priority == 1
    assert buffer_status_list[2].hol_delay == 0

    tb_size = 5000

    # LCM должен поделить TB равномерно между LC (по 1666 байт на каждый LC)
    grants_list = scheduler._logical_channel_multiplexing(
        tb_size=tb_size,
        buffer_status_list=buffer_status_list,
    )

    # Grant для QCI 1 (LCID 3)
    assert grants_list[0].ue_id == 1
    assert grants_list[0].num_bytes == 1666
    assert grants_list[0].lcid == 3

    # Grant для QCI 2 (LCID 4)
    assert grants_list[1].ue_id == 1
    assert grants_list[1].num_bytes == 1666
    assert grants_list[1].lcid == 4

    # Grant для QCI 3 (LCID 5)
    assert grants_list[2].ue_id == 1
    assert grants_list[2].num_bytes == 1666
    assert grants_list[2].lcid == 5