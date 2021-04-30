# -*- coding: utf-8 -*-
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Optional

import numpy as np
from resampy import resample as rsmpy_resample
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
from signalworks.tracking.tracking import Track

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Wave(Track):
    """monaural waveform"""

    default_suffix = ".wav"

    def __init__(
        self,
        value: np.ndarray,
        fs: int,
        duration: Optional[int] = None,
        offset: int = 0,
        path: Optional[Path] = None,
    ) -> None:
        """Initializer for wave track object

        Parameters
        ----------
        value : np.ndarray
            orientation of array should be [samples, n_channels]
        fs : int
            sample rate of the signal
        duration : Optional[int], optional
            number of samples in the signal, typically automatically calculated, by default None
        offset : int, optional
            initial offset of the signal, by default 0
        path : Optional[Path], optional
            underlying path for file referenced, by default None
        """
        super().__init__(path)

        if path is None:
            self.path = (Path.home() / str(id)).with_suffix(self.default_suffix)
        else:
            self.path = Path(path).with_suffix(self.default_suffix)
        if value.ndim == 1:  # handle case of 1-d array
            value = value[:, np.newaxis]

        if np.issubdtype(value.dtype, np.integer):
            self.min = np.iinfo(value.dtype).min
            self.max = np.iinfo(value.dtype).max
        elif np.issubdtype(value.dtype, np.floating):
            # normalize floating point audio to -1:1
            max_value = np.absolute(value).max()
            # if 0 < max_value <= 1.0 then our audio is likely scaled properly as is..
            if max_value > 1.0:
                value = value / (max_value + np.finfo(value.dtype).eps)
            self.min = -1.0
            self.max = 1.0

        assert isinstance(value, np.ndarray)
        assert 2 == value.ndim, "values need to be a 2D array"
        assert isinstance(fs, int)
        assert fs > 0
        self._value = value
        self._fs = fs
        self._offset = (
            offset  # this is required to support heterogenous fs in multitracks
        )
        self.type = "Wave"
        self.label = f"amplitude-{value.dtype}"
        if duration is None:
            duration = len(self._value)
        assert len(self._value) <= duration < len(self._value) + 1, (
            "Cannot set duration of a wave to other than a number in "
            "[length, length+1), where length = len(self.value)"
        )
        self._duration = duration

    def get_offset(self):
        return self._offset

    def set_offset(self, offset):
        assert 0.0 <= offset < 1.0
        self._offset = offset

    offset = property(get_offset, set_offset)

    def get_time(self):
        return (
            np.arange(len(self._value))
            if self._offset == 0
            else np.arange(len(self._value)) + self._offset
        )

    def set_time(self, time):
        raise Exception("can't set times for Wave")

    time = property(get_time, set_time)

    def get_value(self):
        return self._value

    def set_value(self, value):
        assert isinstance(value, np.ndarray)
        assert 1 == value.ndim, "only a single channel is supported"
        self._value = value
        if not (len(self._value) <= self._duration < len(self._value) + 1):
            self._duration = len(self._value)

    value = property(get_value, set_value)

    def get_fs(self):
        return self._fs

    def set_fs(self, _value):
        raise Exception("Cannot change fs, try resample()")

    fs = property(get_fs, set_fs, doc="sampling frequency")

    def get_duration(self):
        return self._duration

    def set_duration(self, duration):
        assert len(self._value) <= duration < len(self._value) + 1, (
            "Cannot set duration of a wave to other than a number in [length, length+1) "
            "- where length = len(self.value)"
        )
        self._duration = duration

    duration = property(get_duration, set_duration)

    #  def get_channels(self):
    #      return 1 if self._value.ndim == 1 else self._value.shape[1] # by convention

    #  def set_channels(self, *args):
    #      raise Exception("Cannot change wave channels - create new wave instead")

    #  channels = property(get_channels)  #, set_channels)

    def get_dtype(self):
        return self._value.dtype

    #  def set_dtype(self, *args):
    #      raise Exception("Cannot change wave dtype - create new wave instead")
    dtype = property(get_dtype)

    def get_bitdepth(self):
        return self._value.dtype.itemsize * 8

    bitdepth = property(get_bitdepth)

    def __eq__(self, other):
        if (
            (self._fs == other._fs)
            and (self._duration == other._duration)
            and (len(self._value) == len(other._value))
            and (self._value == other._value).all()
        ):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self) -> str:
        return "\n".join(
            [
                f"value={self.value}",
                f"min={self.value.min()}",
                f"max={self.value.max()}",
                f"dtype={self.dtype}",
                f"fs={self.fs}",
                f"duration={self.duration}",
            ]
        )

    def __repr__(self) -> str:
        return str(self)

    def __len__(self):
        return len(self._value)

    def __add__(self, other):
        assert type(other) == type(self), "Cannot add Track objects of different types"
        assert self._fs == other._fs, "sampling frequencies must match"
        assert self.value.dtype == other.value.dtype, "dtypes must match"
        value = np.concatenate((self._value, other._value))
        return type(self)(value, self.fs)

    def resample(self, fs: int) -> "Wave":
        """resample to a certain fs.  wraps resampy.resample"""
        if fs == self._fs:
            return self
        return type(self)(rsmpy_resample(self._value, self._fs, fs), fs)

    def _convert_dtype(self, source, target_dtype):
        """
        return a link (if unchanged) or copy of signal in the specified dtype (often changes bit-depth as well)
        """
        assert isinstance(source, np.ndarray)
        source_dtype = source.dtype
        assert source_dtype in (
            np.int16,
            np.int32,
            np.float32,
            np.float64,
        ), "source must be a supported type"
        assert target_dtype in (
            np.int16,
            np.int32,
            np.float32,
            np.float64,
        ), "target must be a supported type"
        if source_dtype == target_dtype:
            return source
        else:  # conversion
            if source_dtype == np.int16:
                if target_dtype == np.int32:
                    return source.astype(target_dtype) << 16
                else:  # target_dtype == np.float32 / np.float64:
                    return source.astype(target_dtype) / (1 << 15)
            elif source_dtype == np.int32:
                if target_dtype == np.int16:
                    return (source >> 16).astype(target_dtype)  # lossy
                else:  # target_dtype == np.float32 / np.float64:
                    return source.astype(target_dtype) / (1 << 31)
            else:  # source_dtype == np.float32 / np.float64
                M = np.max(np.abs(source))
                limit = 1 - 1e-16
                if M > limit:
                    factor = limit / M
                    logger.warning(
                        f"maximum float waveform value {M} is beyond [-{limit}, {limit}],"
                        f"applying scaling of {factor}"
                    )
                    source *= factor
                if target_dtype == np.float32 or target_dtype == np.float64:
                    return source.astype(target_dtype)
                else:
                    if target_dtype == np.int16:
                        return (source * (1 << 15)).astype(target_dtype)  # dither?
                    else:  # target_dtype == np.int32
                        return (source * (1 << 31)).astype(target_dtype)  # dither?

    def convert_dtype(self, target_dtype):
        """returns a new wave with the waveform in the specified target_dtype"""
        # TODO: take care of setting new min and max
        return type(self)(
            self._convert_dtype(self._value, target_dtype), self._fs, path=self.path
        )

    def select(self, a, b):
        assert a >= 0
        assert a < b  # or a <= b?
        assert b <= self.duration
        # TODO: modify this for float a and b
        return type(self)(self._value[a:b], self._fs)

    @classmethod
    def read(cls, name, *_args, **_kwargs):
        """Loads object from name, adding default extension if missing."""
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            name += cls.default_suffix
        return cls.read_wav(name)

    @classmethod
    def read_wav(cls, path, channel=None, mmap=False):
        """load waveform from file"""
        try:
            fs, value = wav_read(path, mmap=mmap)
            if np.ndim(value) == 1:
                value = value.reshape(-1, 1)
        except ValueError:
            try:
                if mmap:
                    logger.warning("mmap is not supported by soundfile, ignoring")
                import soundfile as sf

                audioEncodings: DefaultDict[str, str] = defaultdict(lambda: "float64")
                audioEncodings["PCM_S8"] = "int16"  # soundfile does not support int8
                audioEncodings["PCM_U8"] = "int16"  # soundfile does not support uint16
                audioEncodings["PCM_16"] = "int16"
                audioEncodings["PCM_24"] = "int32"  # there is no np.int24
                audioEncodings["PCM_32"] = "int32"
                audioEncodings["FLOAT"] = "float32"
                audioEncodings["DOUBLE"] = "float64"
                file_info = sf.info(path)

                value, fs = sf.read(
                    path, dtype=audioEncodings[file_info.subtype], always_2d=True
                )
            except ImportError:
                logger.error("Install soundfile for greater audio file compatability")
            except RuntimeError:
                logger.error("Soundfile was unable to open file")
                return None

        if channel is not None:
            value = value[:, channel]
        wav = Wave(value, fs, path=path)
        return wav

    wav_read = read_wav

    def write(self, name, *_args, **_kwargs):
        """Saves object to name, adding default extension if missing."""
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            name += self.default_suffix
        self.write_wav(name)

    def write_wav(self, name):
        """save waveform to file
        The bits-per-sample will be determined by the data-type (mostly)"""
        wav_write(name, self._fs, self._value)

    wav_write = write_wav

    def __getitem__(self, index):
        return Wave(self._value[index], self.fs)

    # def __setitem__(self, index, value):
    #    self._value[index] = value

    # def __add__(self, other):
    #     """wave1 + wave2"""
    #     if self.fs != other.fs:
    #         raise Exception("sampling frequency of waves must match")
    #     return type(self)(np.concatenate((self.va, other.va)), self.fs)  # return correct (child) class

    # def delete(self, a, b, fade = 0):
    #     pass

    # def cut(self, a, b, fade = 0):
    #     wave = self.copy(a, b)
    #     self.delete(a, b, fade)
    #     return wave

    # def insert(self, wave, a, fade = 0):
    #     """insert wave into self at time a"""
    #     if self.fs != wave.fs:
    #         raise Exception("sampling frequency of waves must match")
    #     if fade:
    #         n = round(fade * self.fs)
    #         if n*2 > len(wave.signal):
    #             raise Exception("fade inverval is too large")
    #         up = np.linspace(0, 1, n)
    #         down = np.linspace(1, 0, n)
    #         p = wave.signal.copy()
    #         p[:n] *= up
    #         p[-n:] *= down
    #         l = self.signal[:a+n]
    #         l[-n:] *= down
    #         r = self.signal[a-n:]
    #         r[:n] *= up
    #     else:
    #         self.signal = np.concatenate((self.signal[:a], wave.signal, self.signal[a:]))

    def crossfade(self, wave, length):
        """append wave to self, using a crossfade of a specified length in samples"""
        assert type(self) == type(wave), "Cannot add Track objects of different types"
        assert self.fs == wave.fs, "sampling frequency of waves must match"
        assert isinstance(length, int)
        assert length > 0
        assert wave.duration >= length
        assert self.duration >= length
        ramp = np.linspace(1, 0, length + 2)[1:-1][
            :, np.newaxis
        ]  # don't include 0 and 1

        value = self.value.copy()
        value[-length:] = value[-length:] * ramp + wave.value[:length] * (
            1 - ramp
        )  # TODO: think about dtypes here
        value = np.concatenate((value, wave.value[length:]))
        return type(self)(value, self.fs)

    # TODO: Test / fix me!
    def time_warp(self, x, y):
        raise NotImplementedError
        logger.warning(
            "time_warping wave, most of the time this is not what is desired"
        )
        time = np.arange(len(self._value))
        # time = index / self._fs
        time = np.round(np.interp(time, x, y)).astype(np.int16)
        # index = int(time * self.fs)
        self._value = self._value[time]
