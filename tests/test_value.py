from signalworks.tracking.value import Value
import pytest

@pytest.fixture
def var():
    a = Value('this is a comment', 1, 10)
