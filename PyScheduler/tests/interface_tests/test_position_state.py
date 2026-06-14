import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from INTERFACES import PositionState  # noqa: E402
from UE_MODULE import UECollection, UserEquipment  # noqa: E402


def test_user_equipment_returns_position_state():
    ue = UserEquipment(UE_ID=1, x=10.0, y=20.0)
    ue.velocity = 1.5
    ue.direction = 0.25

    state = ue.get_position_state()

    assert isinstance(state, PositionState)
    assert state.ue_id == 1
    assert state.x == 10.0
    assert state.y == 20.0
    assert state.velocity == 1.5
    assert state.direction == 0.25


def test_ue_collection_returns_position_states_for_all_users():
    collection = UECollection()
    collection.ADD_USER(UserEquipment(UE_ID=1, x=10.0, y=20.0))
    collection.ADD_USER(UserEquipment(UE_ID=2, x=30.0, y=40.0))

    states = collection.get_position_states()

    assert states == [
        PositionState(ue_id=1, x=10.0, y=20.0, velocity=0.0, direction=0.0),
        PositionState(ue_id=2, x=30.0, y=40.0, velocity=0.0, direction=0.0),
    ]
