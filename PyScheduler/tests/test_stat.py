import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from TRAFFIC_MODEL import Packet, TrafficStatistics


class TestTrafficStatistics:
    def test_update_increments_counters(self):
        stats = TrafficStatistics()

        packets = [
            Packet(size=1000, ue_id=1, creation_time=100, qci=1),
            Packet(size=1500, ue_id=1, creation_time=101, qci=1),
        ]

        stats.update(packets)

        assert stats._total_packets == 2
        assert stats._total_bytes == 2500
        assert stats._ue_packets[1] == 2
        assert stats._qci_packets[1] == 2

    def test_get_ue_stats(self):
        stats = TrafficStatistics()

        packets = [
            Packet(size=1000, ue_id=1, creation_time=1000, qci=9),
            Packet(size=1000, ue_id=1, creation_time=2000, qci=9),
        ]
        stats.update(packets)

        ue_stats = stats.get_ue_stats(ue_id=1)

        assert ue_stats["packets"] == 2
        assert ue_stats["bytes"] == 2000
        assert ue_stats["avg_packet_size"] == 1000.0
        assert ue_stats["duration_ms"] == 1000.0  # 2000 - 1000

        # 2000 bytes * 8 bits / 1 sec = 16000 bps
        assert ue_stats["bitrate_bps"] == 16000.0

    def test_qci_distribution(self):
        stats = TrafficStatistics()

        packets = [
            Packet(size=1000, ue_id=1, creation_time=100, qci=1),
            Packet(size=1000, ue_id=1, creation_time=101, qci=9),
            Packet(size=2000, ue_id=2, creation_time=102, qci=9),
        ]
        stats.update(packets)

        dist = stats.get_qci_distribution()

        # QCI 1: 1000 bytes = 25%
        # QCI 9: 3000 bytes = 75%
        assert dist[1] == 25.0
        assert dist[9] == 75.0
