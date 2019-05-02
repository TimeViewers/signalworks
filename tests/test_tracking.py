# -*- coding: utf-8 -*-
import numpy as np
import pytest  # noqa: F401
from signalworks import dsp
from signalworks.tracking.wave import Wave


def test_spectrogram(speech_track, benchmark):
    frame_size = 0.030
    half = frame_size * speech_track.fs // 2
    centers = np.round(np.linspace(half, speech_track.duration - half, 1000)).astype(
        np.int
    )

    NFFT = 2 ** dsp.nextpow2(512 * 2)
    assert isinstance(speech_track, Wave)
    X, f = benchmark(
        dsp.spectrogram_centered,
        speech_track,
        frame_size,
        centers,
        NFFT=NFFT,
        normalized=False,
    )
    assert X.shape == (len(centers), NFFT // 2 + 1)


def test_world(speech_track, benchmark):
    sp, f0, t = benchmark(dsp.world, speech_track)
    assert sp.shape[1] == 513
