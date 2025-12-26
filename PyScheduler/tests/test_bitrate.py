import sys
from collections import deque
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from TRAFFIC_MODEL import BitrateController, Packet


class TestBitrateController:
    def test_set_limit(self):
        ctrl = BitrateController(window_ms=1000)
        ctrl.set_limit(ue_id=1, max_bitrate_mbps=10.0)

        assert ctrl._limits[1] == 10e6  # 10 Mbps = 10e6 bps

    def test_no_limit_passes_all(self):
        """Без лимита все пакеты проходят"""
        ctrl = BitrateController()

        packets = [
            Packet(size=1000, ue_id=1, creation_time=100),
            Packet(size=1000, ue_id=1, creation_time=101),
        ]

        result = ctrl.check_and_throttle(ue_id=1, packets=packets, current_time=200)
        assert len(result) == 2

    def test_throttling_when_limit_exceeded(self):
        """Throttling при превышении лимита"""
        ctrl = BitrateController(window_ms=1000)
        ctrl.set_limit(ue_id=1, max_bitrate_mbps=0.01)  # 10 Kbps (очень мало!)

        # Пакеты 1000 байт = 8000 бит каждый
        # Лимит 10 Kbps = 10000 bps
        # Первый пакет пройдёт (8000 < 10000)
        # Второй НЕ пройдёт (8000 + 8000 > 10000)

        packets = [
            Packet(size=1000, ue_id=1, creation_time=1000),
            Packet(size=1000, ue_id=1, creation_time=1001),
        ]

        result = ctrl.check_and_throttle(ue_id=1, packets=packets, current_time=1001)

        assert len(result) == 1  # Только первый пакет прошёл
        assert ctrl._dropped_count[1] == 1

    def test_get_current_bitrate(self):
        """Подсчёт текущего bitrate"""
        ctrl = BitrateController(window_ms=1000)

        # 10 пакетов по 1000 байт за 1 секунду
        for i in range(10):
            pkt = Packet(size=1000, ue_id=1, creation_time=1000 + i)
            ctrl._packet_history[1] = ctrl._packet_history.get(1, deque())
            ctrl._packet_history[1].append((pkt.creation_time, pkt.size))

        # 10 * 1000 * 8 = 80000 бит за 1 сек = 80 Kbps
        bitrate = ctrl.get_current_bitrate(ue_id=1, current_time=2000)
        assert bitrate == 80000  # 80 Kbps
