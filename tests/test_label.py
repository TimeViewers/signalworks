import numpy
import pytest
from signalworks.tracking.label import Label


@pytest.fixture
def var():
    p1 = Label(
        numpy.array([0, 3, 5, 8, 10, 12], dtype=numpy.int64),
        numpy.array(["start", "middle", "end"]),
        1,
        12,
    )
    p2 = Label(
        numpy.array([0, 5, 8, 10], dtype=numpy.int64),
        numpy.array(["start2", "end2"]),
        1,
        10,
    )
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


#
# def test_select(var):
#     p1, p2 = var
#     p = p1.select(5, 6)
#     assert p == Partition(numpy.array([0, 1], dtype=numpy.int64), numpy.array(['middle']), 1)
#     p = p1.select(5, 7)
#     assert p == Partition(numpy.array([0, 1, 2], dtype=numpy.int64), numpy.array(['middle', 'end']), 1)
#     p = p1.select(4, 6)
#     assert p == Partition(numpy.array([0, 1, 2], dtype=numpy.int64), numpy.array(['start', 'middle']), 1)
#     p = p1.select(4, 7)
#     assert p == Partition(numpy.array([0, 1, 2, 3], dtype=numpy.int64),
#                                    numpy.array(['start', 'middle', 'end']),
#                                    1)
#
# def test_merge_same():
#     p = Partition(numpy.array([0, 3, 6, 10], dtype=numpy.int64), numpy.array(["1", "1", "2"]), 1)
#     p = p.merge_same()
#     assert p.value[1] == "2"
