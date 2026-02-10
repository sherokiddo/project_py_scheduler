import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

import GLOBALS
from BS_MODULE import SimpleBufferManager
from TRAFFIC_MODEL import Packet
from SCHEDULER import SchedulingGrant

def test_add_packet_without_buffer_raises():
    manager = SimpleBufferManager()
    packet = Packet(size=500, ue_id=1, creation_time=0)

    with pytest.raises(ValueError, match="buffer"):
        manager.add_packet(ue_id=1, packet=packet)

def test_add_packet_existing_buffer():
    manager = SimpleBufferManager(default_max_size=1000)
    manager.create_ue_buffer(ue_id=1)

    packet = Packet(size=100, ue_id=1, creation_time=0)
    result = manager.add_packet(ue_id=1, packet=packet)

    assert result is True
    assert manager.buffers[1].current_size == 100

def test_add_packet_overflow_existing_buffer():
    manager = SimpleBufferManager(default_max_size=1000)
    manager.create_ue_buffer(ue_id=1)

    packet = Packet(size=1500, ue_id=1, creation_time=0)
    result = manager.add_packet(ue_id=1, packet=packet)

    assert result is False
    assert manager.buffers[1].current_size == 0
    assert manager.buffers[1].total_packets_dropped == 1
    
def test_get_buffer_status():
    manager = SimpleBufferManager()
    manager.create_ue_buffer(ue_id=1)
    
    packet = Packet(size=200, ue_id=1, creation_time=0)
    manager.add_packet(ue_id=1, packet=packet)
    
    GLOBALS.CURRENT_TIME = 5

    buffer_status = manager.get_buffer_status(ue_id=1)

    assert isinstance(buffer_status, list)
    assert len(buffer_status) == 1

    status = buffer_status[0]
    assert status.ue_id == 1
    assert status.buffer_size == 200
    assert status.timestamp == 5
    assert status.lcid is None
    assert status.qci is None
    assert status.priority == 0
    assert status.hol_delay is None
    
def test_get_packets_single_grant():
    manager = SimpleBufferManager()
    manager.create_ue_buffer(ue_id=1)

    pkt1 = Packet(size=3000, ue_id=1, creation_time=0)
    pkt2 = Packet(size=1500, ue_id=1, creation_time=0)
    manager.add_packet(ue_id=1, packet=pkt1)
    manager.add_packet(ue_id=1, packet=pkt2)

    grant = SchedulingGrant(ue_id=1, num_bytes=4000)

    packets, extracted_bytes = manager.get_packets([grant])

    assert extracted_bytes == 4000
    assert len(packets) == 2
    assert manager.buffers[1].current_size == 500

def test_upd_buffers_all_expired():
    manager = SimpleBufferManager()
    manager.create_ue_buffer(ue_id=1)
    manager.create_ue_buffer(ue_id=2)

    pkt1 = Packet(size=3000, ue_id=1, creation_time=0, deadline=5)
    pkt2 = Packet(size=1500, ue_id=1, creation_time=0, deadline=15)

    manager.add_packet(ue_id=1, packet=pkt1)
    manager.add_packet(ue_id=1, packet=pkt2)
    
    pkt1 = Packet(size=3000, ue_id=1, creation_time=0, deadline=1)
    pkt2 = Packet(size=1500, ue_id=1, creation_time=0, deadline=10)
    
    manager.add_packet(ue_id=2, packet=pkt1)
    manager.add_packet(ue_id=2, packet=pkt2)

    GLOBALS.CURRENT_TIME = 10
    manager.upd_buffers_all()

    buffer1 = manager.buffers[1]
    assert buffer1.current_size == 1500
    assert buffer1.total_packets_expired == 1
    
    buffer2 = manager.buffers[2]
    assert buffer2.current_size == 0
    assert buffer2.total_packets_expired == 2
    
def test_remove_ue_buffer():
    manager = SimpleBufferManager()
    manager.create_ue_buffer(ue_id=1)

    manager.add_packet(ue_id=1, packet=Packet(size=3000, ue_id=1, creation_time=0))
    manager.remove_ue_buffer(1)

    assert 1 not in manager.buffers
