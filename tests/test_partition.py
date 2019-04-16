import numpy
import pytest
from signalworks.tracking.partition import Partition


@pytest.fixture
def var():
    p1 = Partition(
        numpy.array([0, 5, 6, 10], dtype=numpy.int64),
        numpy.array(["start", "middle", "end"]),
        1,
    )
    p2 = Partition(
        numpy.array([0, 5, 10], dtype=numpy.int64), numpy.array(["start2", "end2"]), 1
    )
    return p1, p2


def test_insert():
    p = Partition(
        numpy.array([0, 5, 6, 10], dtype=numpy.int64),
        numpy.array(["start", "middle", "end"]),
        1,
    )
    p.insert(7, "still the middle")
    assert (p.time == numpy.array([0, 5, 6, 7, 10])).all()


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
    assert p == Partition(
        numpy.array([0, 1], dtype=numpy.int64), numpy.array(["middle"]), 1
    )
    p = p1.select(5, 7)
    assert p == Partition(
        numpy.array([0, 1, 2], dtype=numpy.int64), numpy.array(["middle", "end"]), 1
    )
    p = p1.select(4, 6)
    assert p == Partition(
        numpy.array([0, 1, 2], dtype=numpy.int64), numpy.array(["start", "middle"]), 1
    )
    p = p1.select(4, 7)
    assert p == Partition(
        numpy.array([0, 1, 2, 3], dtype=numpy.int64),
        numpy.array(["start", "middle", "end"]),
        1,
    )


def test_merge_same():
    p = Partition(
        numpy.array([0, 3, 6, 10], dtype=numpy.int64), numpy.array(["1", "1", "2"]), 1
    )
    p = p.merge_same()
    assert p.value[1] == "2"
