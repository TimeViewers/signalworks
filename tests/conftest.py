# -*- coding: utf-8 -*-
import distutils.spawn
from pathlib import Path

import pytest  # noqa: F401

from signalworks.tracking import Track


def has_gitlfs() -> bool:
    return distutils.spawn.find_executable("git-lfs") is not None


def is_gitlfs_pointer(path: Path) -> bool:
    return path.stat().st_blocks == 8 and path.stat().st_blksize == 4096


xfailif_no_gitlfs = pytest.mark.xfail(
    not has_gitlfs(), reason="This test requires git-lfs"
)


@xfailif_no_gitlfs
@pytest.fixture(scope="function")
def speech_track():
    speech_wav = Path(__file__).parents[1] / "data" / "speech-mwm.wav"
    if is_gitlfs_pointer(speech_wav):
        return None
    else:
        track = Track.read(speech_wav)
    assert isinstance(track, Track)
    return track
