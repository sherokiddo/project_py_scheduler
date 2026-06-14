import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from INTERFACES import ChannelQualityState  # noqa: E402
from UE_MODULE import UECollection, UserEquipment  # noqa: E402


def test_user_equipment_returns_channel_quality_state():
    ue = UserEquipment(UE_ID=1)
    ue.SINR = 12.5
    ue.cqi = 9
    ue.cqi_subband = [8, 9, 10]

    state = ue.get_channel_quality_state()

    assert isinstance(state, ChannelQualityState)
    assert state.ue_id == 1
    assert state.sinr == 12.5
    assert state.cqi == 9
    assert state.cqi_subband == (8, 9, 10)


def test_channel_quality_state_is_snapshot_of_subband_cqi():
    ue = UserEquipment(UE_ID=1)
    ue.cqi_subband = [8, 9, 10]

    state = ue.get_channel_quality_state()
    ue.cqi_subband.append(11)

    assert state.cqi_subband == (8, 9, 10)


def test_ue_collection_returns_channel_quality_states_for_all_users():
    ue1 = UserEquipment(UE_ID=1)
    ue1.SINR = 10.0
    ue1.cqi = 7
    ue1.cqi_subband = [6, 7]

    ue2 = UserEquipment(UE_ID=2)
    ue2.SINR = 14.0
    ue2.cqi = 11
    ue2.cqi_subband = [10, 11]

    collection = UECollection()
    collection.ADD_USER(ue1)
    collection.ADD_USER(ue2)

    states = collection.get_channel_quality_states()

    assert states == [
        ChannelQualityState(ue_id=1, sinr=10.0, cqi=7, cqi_subband=(6, 7)),
        ChannelQualityState(ue_id=2, sinr=14.0, cqi=11, cqi_subband=(10, 11)),
    ]
