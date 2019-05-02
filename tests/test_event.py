import numpy as np
import pytest
from signalworks.tracking import Event, Track


@pytest.fixture
def var():
    t = Event(np.array([3, 6], dtype=np.int64), 1, 10)
    u = Event(np.array([0, 2], dtype=np.int64), 1, 3)
    assert isinstance(t, Event)
    assert isinstance(u, Event)
    assert isinstance(u, Track)
    return t, u


def test_init():
    with pytest.raises(Exception):
        Event(np.array([6, 3], dtype=np.int64), 1, 10)  # bad times
    with pytest.raises(Exception):
        Event(np.array([3, 6], dtype=np.int64), 1, 5)  # duration too shorts


def test_duration(var):
    t, _ = var
    t.duration = 8
    with pytest.raises(Exception):
        t.set_duration(5)


def test_eq(var):
    t, u = var
    assert t == t
    assert not t == u


def test_add(var):
    t, u = var
    v = t + u
    assert v == Event(np.array([3, 6, 10, 12], dtype=np.int64), 1, 13)
    t += u
    assert t == Event(np.array([3, 6, 10, 12], dtype=np.int64), 1, 13)


def test_select(var):
    t, u = var
    t_ = t.select(2, 7)
    assert t_ == Event(np.array([1, 4], dtype=np.int64), 1, 5)
    t_ = t.select(3, 7)
    assert t_ == Event(np.array([0, 3], dtype=np.int64), 1, 4)
    t_ = t.select(3, 6)
    assert t_ == Event(np.array([0], dtype=np.int64), 1, 3)
    t_ = t.select(2, 6)
    assert t_ == Event(np.array([1], dtype=np.int64), 1, 4)


def test_resample(benchmark):
    e = Event(np.array([3, 6], dtype=np.int64), 1, 10)
    # e = e.resample(2)
    e = benchmark(e.resample, 2)
    assert e.time[0] == 6


def test_crossfade(benchmark):
    evt1 = Event(np.array([1, 5, 9], dtype=np.int64), 1, 10)
    evt2 = Event(np.array([2, 5, 9], dtype=np.int64), 1, 10)
    length = 2
    # evt = evt1.crossfade(evt2, length)
    evt = benchmark(evt1.crossfade, evt2, length)
    assert evt1.duration + evt2.duration - length == evt.duration
    assert np.allclose(evt.time, np.array([1, 5, 10, 13, 17]))
