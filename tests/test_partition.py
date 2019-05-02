import numpy as np
import pytest
from signalworks.tracking import Partition, TimeValue, Track


@pytest.fixture
def var():
    p1 = Partition(
        np.array([0, 5, 6, 10], dtype=np.int64), np.array(["start", "middle", "end"]), 1
    )
    p2 = Partition(
        np.array([0, 5, 10], dtype=np.int64), np.array(["start2", "end2"]), 1
    )
    assert isinstance(p1, Track)
    assert isinstance(p2, Track)
    return p1, p2


def test_insert():
    p = Partition(
        np.array([0, 5, 6, 10], dtype=np.int64), np.array(["start", "middle", "end"]), 1
    )
    p.insert(7, "still the middle")
    assert (p.time == np.array([0, 5, 6, 7, 10])).all()


def test_eq(var):
    p1, p2 = var
    assert p1 == p1
    assert not p1 == p2


def test_add(var):
    p1, p2 = var
    p = p1 + p2
    assert p.duration == 20
    assert p.time[4] == 15
    assert p.value[4] == "end2"
    p1 += p2
    p = p1
    assert p.duration == 20
    assert p.time[4] == 15
    assert p.value[4] == "end2"


def test_select(var):
    p1, p2 = var
    p = p1.select(5, 6)
    assert p == Partition(np.array([0, 1], dtype=np.int64), np.array(["middle"]), 1)
    p = p1.select(5, 7)
    assert p == Partition(
        np.array([0, 1, 2], dtype=np.int64), np.array(["middle", "end"]), 1
    )
    p = p1.select(4, 6)
    assert p == Partition(
        np.array([0, 1, 2], dtype=np.int64), np.array(["start", "middle"]), 1
    )
    p = p1.select(4, 7)
    assert p == Partition(
        np.array([0, 1, 2, 3], dtype=np.int64), np.array(["start", "middle", "end"]), 1
    )


def test_from_TimeValue(benchmark):
    tv = TimeValue(
        np.arange(9, dtype=np.int64) * 10 + 10,
        np.array([0.0, 1.0, 1.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0]),
        1,
        100,
    )
    p = benchmark(Partition.from_TimeValue, tv)
    assert (p.time == np.array([0, 15, 35, 65, 100])).all()
    assert (p.value == np.array([0.0, 1.0, 4.0, 8.0])).all()


def test_merge_same(benchmark):
    p = Partition(np.array([0, 3, 6, 10], dtype=np.int64), np.array(["1", "1", "2"]), 1)
    p = benchmark(p.merge_same)
    assert p.value[1] == "2"


def test_crossfade(benchmark):
    prt1 = Partition(np.array([0, 8, 10], dtype=np.int64), np.array(["1", "2"]), 1)
    prt2 = Partition(np.array([0, 2, 10], dtype=np.int64), np.array(["3", "4"]), 1)
    length = 4
    prt = benchmark(prt1.crossfade, prt2, length)
    assert prt1.duration + prt2.duration - length == prt.duration
    assert np.allclose(prt.time, np.array([0, 8, 16]))
    assert (prt.value == np.array(["1", "4"])).all()
