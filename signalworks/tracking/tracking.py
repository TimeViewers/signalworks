# -*- coding: utf-8 -*-
"""Tracks
Each track has a fs and a duration. There are 4 kinds of tracks:

1 Event - times
2 Wave - values
3 TimeValue - values at times, duration
4 Partition - values between times

All track intervals are of the type [), and duration points to the next unoccupied sample == length
"""

import logging
from builtins import str
from pathlib import Path
from typing import List, Optional

import numpy
from signalworks.tracking.metatrack import MetaTrack

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.ERROR)

TIME_TYPE = numpy.int64


def convert_dtype(source, target_dtype):
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


class Track(MetaTrack):
    default_suffix = ".trk"

    def __init__(self, path):
        self._fs = 0
        self.type: Optional[str] = None
        self.min: Optional[int] = None
        self.max: Optional[int] = None
        self.unit: Optional[str] = None
        self.label: Optional[str] = None
        if path is None:
            path = str(id(self))
        self.path = Path(path).with_suffix(self.default_suffix)

    def get_time(self):
        raise NotImplementedError

    def set_time(self, time):
        raise NotImplementedError

    time = property(get_time, set_time)

    def get_value(self):
        raise NotImplementedError

    def set_value(self, value):
        raise NotImplementedError

    value = property(get_value, set_value)

    def get_fs(self):
        return self._fs

    def set_fs(self, _value):
        raise Exception("Cannot change fs, try resample()")

    fs = property(get_fs, set_fs, doc="sampling frequency")

    def get_duration(self):
        pass

    def set_duration(self, duration):
        raise NotImplementedError

    duration = property(get_duration, set_duration)

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

    def __len__(self):
        pass

    def __str__(self):
        pass

    def __add__(self, other):
        raise NotImplementedError

    @classmethod
    def read(cls, path):
        # we do the imports here to avoid circular import when Wave inherits Track, and Track call Wave's function
        # we only need a function from the dependencies
        from signalworks.tracking.partition import Partition
        from signalworks.tracking.timevalue import TimeValue
        from signalworks.tracking.wave import Wave
        from signalworks.tracking.multitrack import MultiTrack

        """Loads object from name, adding default extension if missing."""
        # E = []
        suffix = Path(path).suffix
        if suffix == ".wav":
            channels = None
            mmap = False
            return Wave.wav_read(path, channels, mmap)
        elif suffix == ".tmv":
            return TimeValue.read_tmv(path)  # for now, handle nans
        elif suffix == ".lab":
            return Partition.read(path)
        elif suffix == ".edf":
            return MultiTrack.read_edf(path)
        elif suffix == ".xdf":
            return MultiTrack.read_xdf(path)
        else:
            raise Exception(f"I don't know how to read files with suffix {suffix}")

    def write(self, name, *args, **kwargs):
        """Saves object to name, adding default extension if missing."""
        raise NotImplementedError

    def resample(self, fs):
        """resample self to a certain fs"""
        raise NotImplementedError

    def select(self, a, b):
        """
        return a selection of the track from a to b. a and b are in fs units.
        Times are new objects, but values are views - idea is to make a read-only section, not a copy
        """
        raise NotImplementedError

    def insert(self, a, t):
        raise NotImplementedError

    def remove(self, a, b):
        raise NotImplementedError

    def copy(self, a, b):
        raise NotImplementedError

    def cut(self, a, b):
        t = self.copy(a, b)
        self.remove(a, b)
        return t


def get_track_classes() -> List[Track]:
    def all_subclasses(c):
        return c.__subclasses__() + [
            a for b in c.__subclasses__() for a in all_subclasses(b)
        ]

    return [obj for obj in all_subclasses(Track)]


# TODO: class NamedEvent(_Track)
#  there hasn't been a need for it yet, but may be useful in the future
#  wonder if I can extend Event itself with optional values...
# class NamedEvent(_Track):
#  def __init__(self, time, value, fs, duration)


# class HetMultiTrack(MultiTrack):  # may want to define common abstract class instead
#     """
#     A dictionary containing time-synchronous tracks of equal duration, but HETEROGENOUS fs
#     """

#     # this fs relates to the manner by which we time-index (possibly with float) into the multitrack object.
#     # Use 1.0 for seconds.
#     def __init__(self, mapping=dict(), fs=1.0):
#         dict.__init__(self, mapping)
#         if __debug__:  # long assert - TODO: do this on mapping, and then assign
#             self.check()
#         self._fs = fs

#     def check(self):
#         if len(self) > 1:
#             duration = None
#             for i, (key, track) in enumerate(self.items()):
#                 if duration is None:
#                     duration = track.duration / track.fs
#                 if track.duration / track.fs != duration:
#                     raise AssertionError(
#                         f"all durations must be equal, track #{i} ('{key}') does not match track #1"
#                     )

#     def get_fs(self):
#         if len(self):
#             return self._fs
#         else:
#             return 0  # or raise?

#     def set_fs(self, fs):
#         self._fs = fs

#     fs = property(get_fs, set_fs, doc="sampling frequency of time-index")

#     def select(self, a, b, keys=None):
#         assert a >= 0
#         assert a < b  # or a <= b?
#         assert b <= self.duration
#         """return a new object with all track views from time a to b"""
#         if keys is None:
#             keys = self.keys()
#         obj = type(self)()
#         for key in keys:
#             trk = self[key]
#             obj[key] = trk.select(
#                 a / self._fs * trk._fs, b / self._fs * trk._fs
#             )  # untested
#         return obj

# def test_pml(self):
#     import tempfile
#     tmp = tempfile.NamedTemporaryFile(prefix='test_pml_')
#     filename = tmp.name
#     tmp.close()
#     self.t.pmlwrite(filename)
#     s = Event.pmlread(filename)
#     os.unlink(filename)
#     # duration CANNOT be encoded in the file (or can it?)
#     s.duration = int(numpy.round(self.t.duration * s.fs / self.t.fs))
#     s = s.resample(self.t.fs)
#     self.assertTrue(numpy.allclose(s.time, self.t.time))
