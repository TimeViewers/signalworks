# -*- coding: utf-8 -*-
import pytest  # noqa: F401

import numpy as np

from signalworks.tracking import tracking
from signalworks import dsp


def test_spectrogram(speech_track):
    frame_size = 0.030
    half = frame_size * speech_track.fs // 2
    centers = np.round(np.linspace(half, speech_track.duration - half, 1000)).astype(
        np.int
    )

    NFFT = 2 ** dsp.nextpow2(512 * 2)
    assert isinstance(speech_track, tracking.Wave)
    X, f = dsp.spectrogram_centered(
        speech_track, frame_size, centers, NFFT=NFFT, normalized=False
    )
    assert X.shape == (len(centers), NFFT // 2 + 1)
