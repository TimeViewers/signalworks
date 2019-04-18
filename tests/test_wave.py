import numpy as np
import pytest
from signalworks.tracking.wave import Wave


@pytest.fixture
def var():
    w = Wave(value=np.arange(0, 16000), fs=16000)
    v = Wave(value=np.arange(100, 200), fs=16000)
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
    assert w == Wave(np.arange(10, 20, dtype=np.int64), 16000)


def test_resample():
    w1 = Wave(value=np.arange(0, 50), fs=1)
    w2 = w1.resample(2)
    assert w1.value[0] == int(w2.value[0])


def test_crossfade():
    wav1 = Wave(np.array([1, 1, 1, 1, 1]), 1)
    wav2 = Wave(np.array([10, 10, 10, 10, 10]), 1)
    length = 3
    wav = wav1.crossfade(wav2, length)
    assert wav1.duration + wav2.duration - length, wav.duration
    assert np.allclose(wav.value, np.array([1, 1, 3, 5, 7, 10, 10]))
