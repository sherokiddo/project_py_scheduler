import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from TRAFFIC_MODEL import ITrafficModel, OnOffModel, Packet, PoissonModel


class TestPoissonModel:
    def test_inherits_from_interface(self):
        """PoissonModel наследуется от ITrafficModel"""
        model = PoissonModel(packet_rate=10)
        assert isinstance(model, ITrafficModel)

    def test_generate_traffic_new_signature(self):
        """Новая сигнатура с ue_id"""
        model = PoissonModel(packet_rate=10)

        # ✅ НОВОЕ: Передаём ue_id
        packets = model.generate_traffic(ue_id=1, current_time=1000, update_interval=100)

        assert isinstance(packets, list)
        # ✅ НОВОЕ: Возвращает Packet объекты
        if packets:
            assert isinstance(packets[0], Packet)
            assert packets[0].ue_id == 1

    def test_returns_packets_not_dicts(self):
        """Возвращает List[Packet], не List[Dict]"""
        model = PoissonModel(packet_rate=20)
        packets = model.generate_traffic(ue_id=5, current_time=2000, update_interval=100)

        for pkt in packets:
            assert isinstance(pkt, Packet)
            assert hasattr(pkt, "size")
            assert hasattr(pkt, "creation_time")
            assert pkt.ue_id == 5

    def test_get_model_name(self):
        """get_model_name возвращает 'Poisson'"""
        model = PoissonModel(packet_rate=5)
        assert model.get_model_name() == "Poisson"

    def test_get_model_info(self):
        """get_model_info содержит параметры"""
        model = PoissonModel(packet_rate=15)
        info = model.get_model_info()

        assert info["name"] == "Poisson"
        assert info["packet_rate"] == 15
        assert "min_packet_size" in info


class TestOnOffModel:
    def test_inherits_from_interface(self):
        model = OnOffModel(duration_on=2, duration_off=3, packet_rate=25)
        assert isinstance(model, ITrafficModel)

    def test_generates_packets(self):
        model = OnOffModel(duration_on=2, duration_off=3, packet_rate=25)
        packets = model.generate_traffic(ue_id=1, current_time=1000, update_interval=100)

        assert isinstance(packets, list)
        for pkt in packets:
            assert isinstance(pkt, Packet)
            assert pkt.ue_id == 1

    def test_maintains_separate_states_for_ues(self):
        """Разные UE имеют разные состояния"""
        model = OnOffModel(duration_on=2, duration_off=3, packet_rate=25)

        # Генерация для UE1
        model.generate_traffic(ue_id=1, current_time=1000, update_interval=100)
        # Генерация для UE2
        model.generate_traffic(ue_id=2, current_time=1000, update_interval=100)

        # Должно быть 2 разных состояния
        assert len(model._device_states) == 2
        assert 1 in model._device_states
        assert 2 in model._device_states

    def test_clear_state_removes_ue(self):
        """clear_state удаляет состояние UE"""
        model = OnOffModel(duration_on=2, duration_off=3, packet_rate=25)

        model.generate_traffic(ue_id=1, current_time=1000, update_interval=100)
        assert 1 in model._device_states

        # ✅ НОВОЕ: Очистка
        model.clear_state(ue_id=1)
        assert 1 not in model._device_states

    def test_clear_state_nonexistent_ue_no_error(self):
        """clear_state для несуществующего UE не вызывает ошибку"""
        model = OnOffModel(duration_on=2, duration_off=3, packet_rate=25)
        # Не должно быть ошибки
        model.clear_state(ue_id=999)
