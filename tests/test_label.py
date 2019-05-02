import numpy as np
import pytest
from signalworks.tracking import Label, Track


@pytest.fixture
def var():
    p1 = Label(
        np.array([0, 3, 5, 8, 10, 12], dtype=np.int64),
        np.array(["start", "middle", "end"]),
        1,
        12,
    )
    p2 = Label(
        np.array([0, 5, 8, 10], dtype=np.int64), np.array(["start2", "end2"]), 1, 10
    )
    assert isinstance(p1, Track)
    assert isinstance(p2, Track)
    return p1, p2


def test_eq(var):
    p1, p2 = var
    assert p1 == p1
    assert not p1 == p2


def test_add(var):
    p1, p2 = var
    p = p1 + p2
    assert p.duration == 22
    assert p.time[7] == 17
    assert p.value[4] == "end2"
    p1 += p2
    p = p1
    assert p.duration == 22
    assert p.time[7] == 17
    assert p.value[4] == "end2"
