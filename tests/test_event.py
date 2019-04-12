from signalworks.tracking.event import Event

import numpy

import pytest

@pytest.fixture
def var():
    t = Event(numpy.array([3, 6], dtype=numpy.int64), 1, 10)
    u = Event(numpy.array([0, 2], dtype=numpy.int64), 1, 3)
    return t, u

def test_eq(var):
    t, u = var
    assert t == t
    assert not t == u

def test_add(var):
    t, u = var
    v = t + u
    assert v == Event(numpy.array([3, 6, 10, 12], dtype=numpy.int64), 1, 13)
    t += u
    assert t == Event(numpy.array([3, 6, 10, 12], dtype=numpy.int64), 1, 13)

def test_select(var):
    t, u = var
    t_ = t.select(2, 7)
    assert t_ == Event(numpy.array([1, 4], dtype=numpy.int64), 1, 5)
    t_ = t.select(3, 7)
    assert t_ == Event(numpy.array([0, 3], dtype=numpy.int64), 1, 4)
    t_ = t.select(3, 6)
    assert t_ == Event(numpy.array([0], dtype=numpy.int64), 1, 3)
    t_ = t.select(2, 6)
    assert t_ == Event(numpy.array([1], dtype=numpy.int64), 1, 4)
