# -*- coding: utf-8 -*-
from pathlib import Path

import pytest  # noqa: F401

import signalworks


@pytest.fixture(scope="function")
def speech_track():
    speech_wav = Path(signalworks.__file__).parents[1] / "data" / "speech-mwm.wav"
    track = signalworks.tracking.Track.read(speech_wav)
    return track
