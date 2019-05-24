# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pytest
import signalworks
from signalworks.tracking import Track


def test_load_wav(benchmark):
    # read regular wav file
    path = Path(signalworks.__file__).parents[1] / "data" / "speech-mwm.wav"
    wave = benchmark(Track.read, path)
    assert np.any(wave.value > 0)
    assert wave.fs == 22050


def test_load_au(benchmark):
    # read au file
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    path = Path(signalworks.__file__).parents[1] / "data" / "test.au"
    wave = benchmark(Track.read, path)
    assert np.any(wave.value > 0)
    assert wave.fs == 44100


def test_load_TIMIT(benchmark):
    # read NIST file
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    path = Path(signalworks.__file__).parents[1] / "data" / "test.WAV"
    wave = benchmark(Track.read, path)
    assert np.any(wave.value > 0)
    assert wave.fs == 16000


def test_load_nis(benchmark):
    # read NIST file
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    path = Path(signalworks.__file__).parents[1] / "data" / "test.nis"
    wave = benchmark(Track.read, path)
    assert np.any(wave.value > 0)
    assert wave.fs == 16000


def test_load_wa1(benchmark):
    # read WA1 file
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    path = Path(signalworks.__file__).parents[1] / "data" / "test.wa1"
    wave = benchmark(Track.read, path)
    assert np.any(wave.value > 0)
    assert wave.fs == 8000


def test_load_wa2(benchmark):
    # read WA2 file
    path = Path(signalworks.__file__).parents[1] / "data" / "test.wa2"
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    wave = benchmark(Track.read, path)
    assert np.any(wave.value > 0)
    assert wave.fs == 8000


@pytest.mark.xfail(reason="We cannot support this kind of file")
def test_load_wv1(benchmark):
    # read WA2 file
    path = Path(signalworks.__file__).parents[1] / "data" / "test.WV1"
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    wave = benchmark(Track.read, path)
    assert np.any(wave.value > 0)
    assert wave.fs == 16000


@pytest.mark.xfail(reason="We cannot support this kind of file")
def test_load_wv2(benchmark):
    # read WA2 file
    path = str(Path(signalworks.__file__).parents[1] / "data" / "test.WV2")
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    wave = benchmark(Track.read, path)
    assert np.any(wave.value > 0)
    assert wave.fs == 16000
