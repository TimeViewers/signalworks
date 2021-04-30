# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.io.wavfile import read as wav_read

from .error import LabreadError, MultiChannelError  # noqa
from .event import Event  # noqa
from .label import Label  # noqa
from .metatrack import MetaTrack  # noqa
from .partition import Partition  # noqa
from .timevalue import TimeValue  # noqa
from .tracking import Track  # noqa: F401
from .value import Value  # noqa
from .wave import Wave  # noqa

from .multitrack import MultiTrack  # isort:skip

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def load_audio(
    path: Path,
    channel: Optional[int] = None,
    mmap: bool = False,
    channel_names: List[str] = ["left", "right"],
) -> MultiTrack:
    """load waveform from file"""
    multiTrack = MultiTrack()
    assert 0 < len(channel_names) <= 2
    try:
        fs, value = wav_read(path, mmap=mmap)
    except ValueError:
        try:
            import soundfile as sf

            value, fs = sf.read(path, dtype="int16")
        except ImportError:
            logging.error(
                f"Scipy was unable to import {path}, "
                f"try installing soundfile python package for more compatability"
            )
            raise ImportError
        except RuntimeError:
            raise RuntimeError(f"Unable to import audio file {path}")
    if value.ndim == 1:
        if channel is not None and channel != 0:
            raise MultiChannelError(
                f"cannot select channel {channel} from monaural file {path}"
            )
        multiTrack[channel_names[0]] = Wave(value[:, np.newaxis], fs, path=path)
    if value.ndim == 2:

        if channel is None:
            multiTrack[channel_names[0]] = Wave(value[:, 0], fs, path=path)
            multiTrack[channel_names[1]] = Wave(value[:, 1], fs, path=path)
        else:
            try:
                multiTrack[channel_names[channel]] = Wave(
                    value[:, channel], fs, path=path
                )
            except IndexError:
                raise MultiChannelError(
                    f"cannot select channel {channel} from file "
                    f"{path} with {value.shape[1]} channels"
                )

    for k in multiTrack.keys():
        value = multiTrack[k].value

        if np.issubdtype(value.dtype, np.integer):
            multiTrack[k].min = np.iinfo(value.dtype).min
            multiTrack[k].max = np.iinfo(value.dtype).max
        elif np.issubdtype(value.dtype, np.floating):
            multiTrack[k].min = -1.0
            multiTrack[k].max = 1.0
        else:
            logging.error(f"Wave dtype {value.dtype} not supported")
            raise NotImplementedError
    return multiTrack
