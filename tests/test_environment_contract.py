import pytest

from environments.base import (
    Environment,
    SnapshotError,
    snapshot_hash,
    take_snapshot,
    validate_snapshot,
)


class FakeEnvironment:
    def __init__(self):
        self.live = {"session": {"cart": {"A": 1}}, "shared": {}}

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:0"

    def start(self):
        return self

    def stop(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        pass

    def snapshot(self, session_id: str):
        return self.live


def test_fake_environment_satisfies_protocol():
    assert isinstance(FakeEnvironment(), Environment)


def test_snapshot_is_a_deep_copy_not_live_state():
    env = FakeEnvironment()
    snap = take_snapshot(env, "s1")
    snap["session"]["cart"]["A"] = 99
    assert env.live["session"]["cart"]["A"] == 1


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"session": {}},
        {"session": {}, "shared": {}, "extra": {}},
        {"session": [], "shared": {}},
        {"session": {"x": (1, 2)}, "shared": {}},
        {"session": {"x": {1, 2}}, "shared": {}},
        {"session": {1: "a"}, "shared": {}},
        {"session": {"x": float("nan")}, "shared": {}},
        {"session": {"x": object()}, "shared": {}},
    ],
)
def test_malformed_snapshots_are_rejected(bad):
    with pytest.raises(SnapshotError):
        validate_snapshot(bad)


def test_hash_is_deterministic_and_order_independent():
    a = {"session": {"a": 1, "b": 2}, "shared": {}}
    b = {"session": {"b": 2, "a": 1}, "shared": {}}
    assert snapshot_hash(a) == snapshot_hash(b)


def test_hash_changes_when_state_changes():
    a = {"session": {"cart": {"A": 1}}, "shared": {}}
    b = {"session": {"cart": {"A": 2}}, "shared": {}}
    assert snapshot_hash(a) != snapshot_hash(b)