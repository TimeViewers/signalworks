# -*- coding: utf-8 -*-
from pathlib import Path

import pytest  # noqa: F401

from signalworks.tracking import Track


@pytest.fixture(scope="function")
def speech_track():
    speech_wav = Path(__file__).parents[1] / "data" / "speech-mwm.wav"
    track = Track.read(speech_wav)
    assert isinstance(track, Track)
    return track
