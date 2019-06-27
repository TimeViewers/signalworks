# -*- coding: utf-8 -*-
import distutils.spawn
from pathlib import Path

import numpy as np
import pytest

from signalworks.tracking import load_audio


def has_gitlfs() -> bool:
    return distutils.spawn.find_executable("git-lfs") is not None


def is_gitlfs_pointer(path: Path) -> bool:
    return path.stat().st_blocks == 8 and path.stat().st_blksize == 4096


xfailif_no_gitlfs = pytest.mark.xfail(
    not has_gitlfs(), reason="This test requires git-lfs"
)


def test_load_wav():
    # read regular wav file
    path = Path(__file__).parents[1] / "data" / "speech-mwm.wav"
    multiTrack = load_audio(path)
    wave = multiTrack["left"]
    assert np.any(wave.value > 0)
    assert wave.fs == 22050


@xfailif_no_gitlfs
def test_load_au():
    # read au file
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    path = Path(__file__).parents[1] / "data" / "test.au"
    if is_gitlfs_pointer(path):
        pytest.skip("Audio object is a git lfs pointer")
    multiTrack = load_audio(path)
    wave = multiTrack["left"]
    assert np.any(wave.value > 0)
    assert wave.fs == 44100


@xfailif_no_gitlfs
def test_load_TIMIT():
    # read NIST file
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    path = Path(__file__).parents[1] / "data" / "test.WAV"
    if is_gitlfs_pointer(path):
        pytest.skip("Audio object is a git lfs pointer")
    multiTrack = load_audio(path)
    wave = multiTrack["left"]
    assert np.any(wave.value > 0)
    assert wave.fs == 16000


@xfailif_no_gitlfs
def test_load_nis():
    # read NIST file
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    path = Path(__file__).parents[1] / "data" / "test.nis"
    if is_gitlfs_pointer(path):
        pytest.skip("Audio object is a git lfs pointer")
    multiTrack = load_audio(path)
    wave = multiTrack["left"]
    assert np.any(wave.value > 0)
    assert wave.fs == 16000


@xfailif_no_gitlfs
def test_2channels():
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    path = Path(__file__).parents[1] / "data" / "01 Quixotic (Scattle Remix).flac"
    if is_gitlfs_pointer(path):
        pytest.skip("Audio object is a git lfs pointer")
    multiTrack = load_audio(path)
    wave_left = multiTrack["left"]
    assert np.any(wave_left.value > 0)
    assert wave_left.fs == 44100

    wave_right = multiTrack["right"]
    assert np.any(wave_right.value > 0)
    assert wave_right.fs == 44100


@xfailif_no_gitlfs
def test_load_wa1():
    # read WA1 file
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    path = Path(__file__).parents[1] / "data" / "test.wa1"
    if is_gitlfs_pointer(path):
        pytest.skip("Audio object is a git lfs pointer")
    multiTrack = load_audio(path)
    wave = multiTrack["left"]
    assert np.any(wave.value > 0)
    assert wave.fs == 8000


@xfailif_no_gitlfs
def test_load_wa2():
    # read WA2 file
    path = Path(__file__).parents[1] / "data" / "test.wa2"
    if is_gitlfs_pointer(path):
        pytest.skip("Audio object is a git lfs pointer")
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    multiTrack = load_audio(path)
    wave = multiTrack["left"]
    assert np.any(wave.value > 0)
    assert wave.fs == 8000


@xfailif_no_gitlfs
@pytest.mark.xfail(reason="We cannot support this kind of file")
def test_load_wv1():
    # read WA2 file
    path = Path(__file__).parents[1] / "data" / "test.WV1"
    if is_gitlfs_pointer(path):
        pytest.skip("Audio object is a git lfs pointer")
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    multiTrack = load_audio(path)
    wave = multiTrack["left"]
    assert np.any(wave.value > 0)
    assert wave.fs == 16000


@xfailif_no_gitlfs
@pytest.mark.xfail(reason="We cannot support this kind of file")
def test_load_wv2():
    # read WA2 file
    path = Path(__file__).parents[1] / "data" / "test.WV2"
    if is_gitlfs_pointer(path):
        pytest.skip("Audio object is a git lfs pointer")
    soundfile = pytest.importorskip(  # noqa
        "soundfile", reason="If soundfile is not installed, this test will fail"
    )
    multiTrack = load_audio(path)
    wave = multiTrack["left"]
    assert np.any(wave.value > 0)
    assert wave.fs == 16000
