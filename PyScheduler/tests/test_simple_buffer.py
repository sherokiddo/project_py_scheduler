import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import GLOBALS
from BS_MODULE import SimpleBuffer
from TRAFFIC_MODEL import Packet


def test_add_packet_success():
    buf = SimpleBuffer(ue_id=1, max_size=10000)
    pkt = Packet(size=5430, ue_id=1, creation_time=0)

    result = buf.add_packet(pkt)

    assert result is True
    assert buf.current_size == 5430
    assert len(buf.buffer) == 1
    assert buf.total_packets_added == 1
    
def test_add_packet_overflow():
    buf = SimpleBuffer(ue_id=1, max_size=1000)
    pkt = Packet(size=5430, ue_id=1, creation_time=0)

    result = buf.add_packet(pkt)

    assert result is False
    assert buf.current_size == 0
    assert len(buf.buffer) == 0
    assert buf.total_packets_dropped == 1
    
def test_get_packets_full_packet():
    buf = SimpleBuffer(ue_id=1, max_size=10000)
    pkt = Packet(size=4000, ue_id=1, creation_time=0)
    buf.add_packet(pkt)

    packets, extracted_bytes = buf.get_packets(num_bytes=4000)

    assert extracted_bytes == 4000
    assert len(packets) == 1
    assert packets[0].size == 4000
    assert buf.current_size == 0
    assert len(buf.buffer) == 0
    
def test_get_packets_fragmentation():
    buf = SimpleBuffer(ue_id=1, max_size=1000)
    pkt = Packet(size=100, ue_id=1, creation_time=0)
    buf.add_packet(pkt)

    packets, extracted_bytes = buf.get_packets(num_bytes=40)

    assert extracted_bytes == 40
    assert len(packets) == 1

    fragment = packets[0]
    assert fragment.size == 40
    assert fragment.is_fragment is True

    # остаток пакета в буфере
    assert len(buf.buffer) == 1
    assert buf.buffer[0].size == 60
    assert buf.current_size == 60

def test_upd_buffer_expired_packets():
    buf = SimpleBuffer(ue_id=1, max_size=1000)

    pkt1 = Packet(size=100, ue_id=1, creation_time=0, deadline=5)
    pkt2 = Packet(size=200, ue_id=1, creation_time=0, deadline=15)

    buf.add_packet(pkt1)
    buf.add_packet(pkt2)

    GLOBALS.CURRENT_TIME = 10
    buf.upd_buffer()

    assert len(buf.buffer) == 1
    assert buf.buffer[0] == pkt2
    assert buf.current_size == 200
    assert buf.total_packets_expired == 1
    
def test_get_buffer_status():
    buf = SimpleBuffer(ue_id=2, max_size=10000)
    buf.add_packet(Packet(size=550, ue_id=2, creation_time=0))
    buf.add_packet(Packet(size=1450, ue_id=2, creation_time=0))
    _, _ = buf.get_packets(num_bytes=1500)

    GLOBALS.CURRENT_TIME = 7
    status = buf.get_buffer_status()

    assert status.ue_id == 2
    assert status.buffer_size == 500
    assert status.timestamp == 7
    assert status.lcid is None
    assert status.qci is None
    assert status.priority == 0
    assert status.hol_delay is None
    
def test_clear_buffer():
    buf = SimpleBuffer(ue_id=1, max_size=1000)
    buf.add_packet(Packet(size=550, ue_id=1, creation_time=0))
    buf.add_packet(Packet(size=1450, ue_id=1, creation_time=0))

    buf.clear()

    assert len(buf.buffer) == 0
    assert buf.current_size == 0
    assert buf.total_packets_added == 0
    assert buf.total_packets_dropped == 0
    assert buf.total_packets_expired == 0
