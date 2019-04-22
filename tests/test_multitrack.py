from collections import UserDict

import numpy as np
import pytest
from signalworks.tracking import Event, MultiTrack, Partition, Wave


@pytest.fixture
def var():
    w = Wave(value=np.arange(0, 50), fs=1)
    p = Partition(
        np.array([0, 15, 20, 50], dtype=np.int64),
        np.array(["start", "middle", "end"]),
        1,
    )
    e = Event(np.array([0, 10, 30]).astype(np.int64), 1, 50)
    return w, p, e


def test_select(var, benchmark):
    dict: UserDict = UserDict()
    dict["wave"], dict["partition"], dict["event"] = var
    mt = MultiTrack(dict)
    new_mt = benchmark(mt.select, 10, 24)
    assert new_mt["wave"] == Wave(value=np.arange(10, 24), fs=1)
    assert new_mt["partition"] == Partition(
        np.array([0, 5, 10, 14], dtype=np.int64),
        np.array(["start", "middle", "end"]),
        1,
    )


def test_crossfade(var, benchmark):
    w1, p1, e1 = var
    dict: UserDict = UserDict()
    dict["wave"], dict["partition"] = w1, p1
    mt1 = MultiTrack(dict)

    w2 = Wave(value=np.arange(0, 50), fs=1)
    p2 = Partition(
        np.array([0, 15, 20, 50], dtype=np.int64),
        np.array(["start", "middle", "end"]),
        1,
    )
    dict = UserDict()
    dict["wave"], dict["partition"] = w2, p2
    mt2 = MultiTrack(dict)

    mt3 = benchmark(mt1.crossfade, mt2, 5)
    assert mt3.duration == mt1.duration + mt2.duration - 5
    assert mt3["wave"] == Wave(
        value=np.r_[np.arange(45), np.array([37, 31, 24, 18, 11]), np.arange(5, 50)],
        fs=1,
    )
    assert mt3["partition"] == Partition(
        np.array([0, 15, 20, 48, 60, 65, 95], dtype=np.int64),
        np.array(["start", "middle", "end", "start", "middle", "end"]),
        1,
    )
