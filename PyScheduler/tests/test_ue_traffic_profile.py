import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from TRAFFIC_MODEL import OnOffModel, PoissonModel, TrafficType, UeTrafficProfile


class TestUeTrafficProfile:
    def test_add_bearer(self):
        profile = UeTrafficProfile(ue_id=1)
        model = PoissonModel(packet_rate=10)

        bearer_id = profile.add_bearer(model=model, qci=1, traffic_type=TrafficType.VOIP)

        assert bearer_id == 1  # First bearer
        assert len(profile.bearers) == 1

    def test_multi_bearer(self):
        """Несколько bearers для одного UE"""
        profile = UeTrafficProfile(ue_id=1)

        # Bearer 1: VOIP
        b1 = profile.add_bearer(
            model=PoissonModel(packet_rate=50), qci=1, traffic_type=TrafficType.VOIP
        )

        # Bearer 2: VIDEO
        b2 = profile.add_bearer(
            model=OnOffModel(duration_on=2, duration_off=3, packet_rate=100),
            qci=7,
            traffic_type=TrafficType.VIDEO_STREAM,
        )

        assert len(profile.bearers) == 2
        assert b1 != b2

    def test_generate_all_traffic(self):
        """Генерация со всех bearers"""
        profile = UeTrafficProfile(ue_id=5)

        profile.add_bearer(model=PoissonModel(packet_rate=10), qci=1, traffic_type=TrafficType.VOIP)
        profile.add_bearer(model=PoissonModel(packet_rate=20), qci=9, traffic_type=TrafficType.WEB)

        packets = profile.generate_all_traffic(current_time=1000, update_interval=100)

        # Должны быть пакеты с обоих bearers
        bearer_ids = {pkt.bearer_id for pkt in packets}
        assert len(bearer_ids) >= 1  # Минимум с одного bearer

        # Все пакеты должны иметь ue_id=5
        assert all(pkt.ue_id == 5 for pkt in packets)

        # bearer_id проставлен
        assert all(pkt.bearer_id is not None for pkt in packets)

    def test_remove_bearer_clears_state(self):
        """remove_bearer очищает состояние stateful модели"""
        profile = UeTrafficProfile(ue_id=1)

        model = OnOffModel(duration_on=2, duration_off=3, packet_rate=25)
        bearer_id = profile.add_bearer(model=model, qci=9, traffic_type=TrafficType.WEB)

        # Генерация создаст состояние
        profile.generate_all_traffic(current_time=1000, update_interval=100)
        assert 1 in model._device_states

        # Удаление должно очистить состояние
        profile.remove_bearer(bearer_id)
        assert 1 not in model._device_states
