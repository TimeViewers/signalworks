import numpy
import pytest
from signalworks.tracking.timevalue import TimeValue
from signalworks.tracking.tracking import Track

TIME_TYPE = numpy.int64


@pytest.fixture
def var():
    t1 = TimeValue(
        (numpy.linspace(1, 9, 3)).astype(TIME_TYPE), numpy.array([1, 4, 2]), 1, 10
    )
    t2 = TimeValue(
        (numpy.linspace(2, 8, 4)).astype(TIME_TYPE), numpy.array([1, 4, 8, 2]), 1, 10
    )
    assert isinstance(t1, Track)
    assert isinstance(t2, Track)
    return t1, t2


def test_init():
    with pytest.raises(Exception):
        TimeValue(
            numpy.array([6, 3], dtype=TIME_TYPE), numpy.array([3, 6]), 1, 10
        )  # bad times
    with pytest.raises(Exception):
        TimeValue(
            numpy.array([3, 6], dtype=TIME_TYPE), numpy.array([3, 6]), 1, 5
        )  # duration too short


def test_eq(var):
    t1, t2, = var
    assert t1 == t1
    assert not t1 == t2


def test_add(var):
    t1, t2, = var
    t = t1 + t2
    assert t.duration == 20
    assert t.time[5] == 16
    assert t.value[5] == 8
    t1 += t2
    t = t1
    assert t.duration == 20
    assert t.time[5] == 16
    assert t.value[5] == 8


def test_select(var):
    t1, _ = var
    t = t1.select(1, 5)
    assert t == TimeValue(numpy.array([0], dtype=TIME_TYPE), numpy.array([1]), 1, 4)
    t = t1.select(1, 6)
    assert t == TimeValue(
        numpy.array([0, 4], dtype=TIME_TYPE), numpy.array([1, 4]), 1, 5
    )
    t = t1.select(1, 6)
    assert t == TimeValue(
        numpy.array([0, 4], dtype=TIME_TYPE), numpy.array([1, 4]), 1, 5
    )
    # t = t1.select(2, 5)
    # assert t == TimeValue(numpy.array([], dtype=TIME_TYPE), numpy.array([]), 1, 3) # empty value


def test_duration(var):
    t1, _ = var
    t1.duration = 11  # ok
    with pytest.raises(Exception):
        t1.set_duration(5)  # duration too short
