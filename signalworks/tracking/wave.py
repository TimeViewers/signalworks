import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
from scipy.signal import resample_poly
from signalworks.tracking.error import MultiChannelError
from signalworks.tracking.tracking import Track

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Wave(Track):
    """monaural waveform"""

    default_suffix = ".wav"

    def __init__(
        self,
        value: numpy.ndarray,
        fs: int,
        duration: int = None,
        offset: int = 0,
        path: Union[str, Path] = None,
    ) -> None:
        super().__init__(path)
        if path is None:
            path = str(id(self))
        # self.min = None
        # self.max = None
        if numpy.issubdtype(value.dtype, numpy.integer):
            info = numpy.iinfo(value.dtype)
            self.min = info.min
            self.max = info.max
        elif numpy.issubdtype(value.dtype, numpy.floating):
            self.min = -1.0
            self.max = 1.0
        else:
            logging.error(f"Wave dtype {value.dtype} not supported")
            raise NotImplementedError

        # TODO: what happens if path is None
        self.path = Path(path).with_suffix(self.default_suffix)
        assert isinstance(value, numpy.ndarray)
        assert 1 <= value.ndim, "only a single channel is supported"
        assert isinstance(fs, int)
        assert fs > 0
        self._value = value
        self._fs = fs
        self._offset = (
            offset
        )  # this is required to support heterogenous fs in multitracks
        self.type: Optional[str] = "Wave"
        self.label: Optional[str] = f"amplitude-{value.dtype}"
        if not duration:
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
            numpy.arange(len(self._value))
            if self._offset == 0
            else numpy.arange(len(self._value)) + self._offset
        )

    def set_time(self, time):
        raise Exception("can't set times for Wave")

    time = property(get_time, set_time)

    def get_value(self):
        return self._value

    def set_value(self, value):
        assert isinstance(value, numpy.ndarray)
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
        dtype = self._value.dtype
        if dtype == numpy.int16:
            return 16
        elif dtype == numpy.float32 or dtype == numpy.int32:
            return 32
        elif dtype == numpy.float64:
            return 64
        else:
            raise Exception("unknown dtype = %s" % dtype)

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

    def __str__(self):
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

    def __len__(self):
        return len(self._value)

    def __add__(self, other):
        assert type(other) == type(self), "Cannot add Track objects of different types"
        assert self._fs == other._fs, "sampling frequencies must match"
        assert self.value.dtype == other.value.dtype, "dtypes must match"
        value = numpy.concatenate((self._value, other._value))
        return type(self)(value, self.fs)

    def resample(self, fs):
        """resample to a certain fs"""
        assert isinstance(fs, int)
        if fs != self._fs:
            if fs > self._fs:  # upsampling
                return type(self)(resample_poly(self._value, int(fs / self._fs), 1), fs)
            else:  # downsampling
                return type(self)(resample_poly(self._value, 1, int(self._fs / fs)), fs)
        else:
            return self

    def _convert_dtype(self, source, target_dtype):
        """
        return a link (if unchanged) or copy of signal in the specified dtype (often changes bit-depth as well)
        """
        assert isinstance(source, numpy.ndarray)
        source_dtype = source.dtype
        assert source_dtype in (
            numpy.int16,
            numpy.int32,
            numpy.float32,
            numpy.float64,
        ), "source must be a supported type"
        assert target_dtype in (
            numpy.int16,
            numpy.int32,
            numpy.float32,
            numpy.float64,
        ), "target must be a supported type"
        if source_dtype == target_dtype:
            return source
        else:  # conversion
            if source_dtype == numpy.int16:
                if target_dtype == numpy.int32:
                    return source.astype(target_dtype) << 16
                else:  # target_dtype == numpy.float32 / numpy.float64:
                    return source.astype(target_dtype) / (1 << 15)
            elif source_dtype == numpy.int32:
                if target_dtype == numpy.int16:
                    return (source >> 16).astype(target_dtype)  # lossy
                else:  # target_dtype == numpy.float32 / numpy.float64:
                    return source.astype(target_dtype) / (1 << 31)
            else:  # source_dtype == numpy.float32 / numpy.float64
                M = numpy.max(numpy.abs(source))
                limit = 1 - 1e-16
                if M > limit:
                    factor = limit / M
                    logger.warning(
                        f"maximum float waveform value {M} is beyond [-{limit}, {limit}],"
                        f"applying scaling of {factor}"
                    )
                    source *= factor
                if target_dtype == numpy.float32 or target_dtype == numpy.float64:
                    return source.astype(target_dtype)
                else:
                    if target_dtype == numpy.int16:
                        return (source * (1 << 15)).astype(target_dtype)  # dither?
                    else:  # target_dtype == numpy.int32
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

    @staticmethod
    def _load_from_scipy(
        path: Path, mmap: bool = False, **kwargs: Dict[Any, Any]
    ) -> Optional[Tuple[int, numpy.ndarray]]:
        try:
            fs, value = wav_read(path, mmap=mmap)
        except ValueError as err:
            logger.error("Scipy was unable to read this file, trying other methods...")
            logger.error(err)
            return None
        return fs, value

    @staticmethod
    def _load_from_soundfile(
        path: Path, dtype: Type = numpy.int16, **kwargs: Dict[Any, Any]
    ) -> Optional[Tuple[int, numpy.ndarray]]:
        logger.info(f"Attempting to load {path} via soundfile library")
        try:
            import soundfile as sf

            value, fs = sf.read(path, dtype=dtype)
        except ImportError:
            logger.error(
                "Attempted to load file using soundfile library, library not installed"
            )
            return None
        except RuntimeError as err:
            logger.error(f"Soundfile was unable to import {path}")
            logger.error(err)
            return None
        return fs, value

    @staticmethod
    def _load_from_sphfile(
        path: Path, **kwargs: Dict
    ) -> Optional[Tuple[int, numpy.ndarray]]:
        logger.info(f"Attempting to load {path} via sphfile library")
        try:
            from sphfile import SPHFile

            audio = SPHFile(path)
            fs = audio.format["sample_rate"]
        except ImportError:
            logger.error("SPHFile library not found")
            return None
        # TODO: catch error importing file
        return fs, audio.content

    @classmethod
    def read_wav(
        cls, path: Path, channel: Optional[int] = None, mmap: bool = False
    ) -> "Wave":
        """load waveform from file"""

        methods = [
            Wave._load_from_scipy,
            Wave._load_from_soundfile,
            Wave._load_from_sphfile,
        ]

        for load_method in methods:
            output = load_method(path, mmap=mmap)  # type: ignore
            if output is not None:
                fs, value = output
                break
        else:
            logger.error(
                f"Unable to import audio file, suggest installing soundfile audio library"
            )
            raise RuntimeError(f"Unable to import audio file {path}")

        if value.ndim == 1:
            if channel is not None and channel != 0:
                raise MultiChannelError(
                    f"cannot select channel {channel} from monaural file {path}"
                )
        elif value.ndim == 2:
            if channel is None:
                raise MultiChannelError(
                    f"must select channel when loading file {path} with {value.shape[1]} channels"
                )
            try:
                value = value[:, channel]
            except IndexError:
                raise MultiChannelError(
                    f"cannot select channel {channel} from file "
                    f"{path} with {value.shape[1]} channels"
                )
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
    #     return type(self)(numpy.concatenate((self.va, other.va)), self.fs)  # return correct (child) class

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
    #         up = numpy.linspace(0, 1, n)
    #         down = numpy.linspace(1, 0, n)
    #         p = wave.signal.copy()
    #         p[:n] *= up
    #         p[-n:] *= down
    #         l = self.signal[:a+n]
    #         l[-n:] *= down
    #         r = self.signal[a-n:]
    #         r[:n] *= up
    #     else:
    #         self.signal = numpy.concatenate((self.signal[:a], wave.signal, self.signal[a:]))

    def crossfade(self, wave, length):
        """append wave to self, using a crossfade of a specified length in samples"""
        assert type(self) == type(wave), "Cannot add Track objects of different types"
        assert self.fs == wave.fs, "sampling frequency of waves must match"
        assert isinstance(length, int)
        assert length > 0
        assert wave.duration >= length
        assert self.duration >= length
        ramp = numpy.linspace(1, 0, length + 2)[1:-1]  # don't include 0 and 1
        value = self.value.copy()
        value[-length:] = value[-length:] * ramp + wave.value[:length] * (
            1 - ramp
        )  # TODO: think about dtypes here
        value = numpy.concatenate((value, wave.value[length:]))
        return type(self)(value, self.fs)

    # TODO: Test / fix me!
    def time_warp(self, x, y):
        raise NotImplementedError
        logger.warning(
            "time_warping wave, most of the time this is not what is desired"
        )
        time = numpy.arange(len(self._value))
        # time = index / self._fs
        time = numpy.round(numpy.interp(time, x, y)).astype(numpy.int)
        # index = int(time * self.fs)
        self._value = self._value[time]
