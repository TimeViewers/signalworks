from signalworks.tracking.wave import Wave

import numpy

import pytest

@pytest.fixture
def var():
    w = Wave(value=numpy.arange(0, 16000), fs=16000)
    v = Wave(value=numpy.arange(100, 200), fs=16000)
    return w, v

def test_add(var):
    w, v = var
    t = w + v
    assert t.duration == 16100
    assert t.value[16050] == 150
    w += v
    assert w.duration == 16100
    assert w.value[16050] == 150

def test_select(var):
    w, _ = var
    w = w.select(10, 20)
    assert w == Wave(numpy.arange(10, 20, dtype=numpy.int64), 16000)
