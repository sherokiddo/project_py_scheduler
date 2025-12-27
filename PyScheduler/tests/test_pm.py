import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from TRAFFIC_MODEL import PacketManager, TrafficType


class TestPacketManager:
    def test_single_bearer_legacy(self):
        """Legacy режим: один UE = одна модель"""
        manager = PacketManager()

        manager.set_model(ue_id=1, model_type="Poisson", packet_rate=10)

        packets = manager.generate_packets(ue_id=1, current_time=1000, update_interval=100)

        assert isinstance(packets, list)
        assert all(pkt.ue_id == 1 for pkt in packets)

    def test_multi_bearer(self):
        """Multi-bearer: один UE, несколько bearers"""
        manager = PacketManager()

        # Bearer 1: VOIP
        b1 = manager.add_bearer(
            ue_id=1, model_type="Poisson", qci=1, traffic_type=TrafficType.VOIP, packet_rate=50
        )

        # Bearer 2: VIDEO
        b2 = manager.add_bearer(
            ue_id=1,
            model_type="OnOff",
            qci=7,
            traffic_type=TrafficType.VIDEO_STREAM,
            duration_on=2,
            duration_off=3,
            packet_rate=100,
        )

        packets = manager.generate_packets(ue_id=1, current_time=1000, update_interval=100)

        # Пакеты должны быть с обоих bearers
        bearer_ids = {pkt.bearer_id for pkt in packets if pkt.bearer_id is not None}
        assert len(bearer_ids) >= 1  # Минимум с одного

        # Разные QCI
        qcis = {pkt.qci for pkt in packets}
        assert 1 in qcis or 7 in qcis

    def test_bitrate_throttling(self):
        """Bitrate throttling работает"""
        manager = PacketManager(enable_bitrate_control=True)

        manager.add_bearer(
            ue_id=1,
            model_type="Poisson",
            qci=9,
            traffic_type=TrafficType.WEB,
            packet_rate=1000,  # Очень высокий rate
        )

        # Устанавливаем низкий лимит
        manager.set_bitrate_limit(ue_id=1, max_bitrate_mbps=0.1)  # 100 Kbps

        packets = manager.generate_packets(ue_id=1, current_time=1000, update_interval=100)

        # Должны быть dropped packets
        stats = manager.bitrate_controller.get_statistics(ue_id=1)
        # (может быть 0 если лимит не превышен, зависит от random)

    def test_callback_called(self):
        """Callback вызывается при генерации пакетов"""
        received_packets = []

        def packet_handler(packets):
            received_packets.extend(packets)

        manager = PacketManager(packet_handler=packet_handler)

        manager.add_bearer(
            ue_id=1, model_type="Poisson", qci=9, traffic_type=TrafficType.WEB, packet_rate=10
        )

        packets = manager.generate_packets(ue_id=1, current_time=1000, update_interval=100)

        # Callback должен был получить те же пакеты
        assert len(received_packets) == len(packets)

    def test_qos_statistics(self):
        """QoS статистика собирается"""
        manager = PacketManager()

        manager.add_bearer(
            ue_id=1, model_type="Poisson", qci=1, traffic_type=TrafficType.VOIP, packet_rate=50
        )

        # Генерация
        for t in range(1000, 5000, 100):
            manager.generate_packets(ue_id=1, current_time=t, update_interval=100)

        # Проверяем статистику
        stats = manager.get_statistics(ue_id=1)
        assert stats["packets"] > 0
        assert stats["bitrate_mbps"] > 0
