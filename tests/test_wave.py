from signalworks.tracking.wave import Wave

import numpy

def test_add():
    w = Wave(numpy.arange(0, 16000), 16000)
    v = Wave(numpy.arange(100, 200), 16000)
    t = w + v
    assert t.duration == 16100
    assert t.value[16050] == 150
    w += v
    assert w.duration == 16100
    assert w.value[16050] == 150
