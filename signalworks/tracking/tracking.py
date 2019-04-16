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
from typing import List

import numpy
from signalworks.tracking.metatrack import MetaTrack
from signalworks.tracking.partition import Partition
from signalworks.tracking.timevalue import TimeValue
from signalworks.tracking.wave import Wave

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
        self.type = None
        self.min = None
        self.max = None
        self.unit = None
        self.label = None
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


# class MultiTrack(dict):
#     """
#     A dictionary containing time-synchronous tracks of equal duration and fs
#     """

#     def __init__(self, mapping={}):
#         dict.__init__(self, mapping)
#         if __debug__:  # long assert - TODO: do this on mapping, and then assign
#             self.check()

#     def check(self):
#         if len(self) > 1:
#             for i, (key, track) in enumerate(self.items()):
#                 if track.fs != self.fs:
#                     raise AssertionError(
#                         f"all fs' must be equal, track #{i} ('{key}) does not match track #1"
#                     )
#                 if track.duration != next(iter(self.values())).duration:
#                     raise AssertionError(
#                         f"all durations must be equal, track #{i} ('{key}'') does not match track #1"
#                     )

#     def get_fs(self):
#         if len(self):
#             return next(iter(self.values())).fs
#         else:
#             return 0  # or raise?

#     def set_fs(self, fs):
#         raise Exception("Cannot change fs, try resample()")

#     fs = property(get_fs, set_fs, doc="sampling frequency")

#     def get_duration(self):
#         if len(self):
#             if __debug__:  # long assert - TODO: do this on mapping, and then assign
#                 self.check()
#             return next(iter(self.values())).duration
#         else:
#             return 0

#     def set_duration(self, duration):
#         raise Exception("The duration cannot be set, it is derived from its conents")

#     duration = property(
#         get_duration, set_duration, doc="duration, as defined by its content"
#     )

#     def __eq__(self, other):
#         # excluding wav from comparison as long as wav writing/reading is erroneous
#         if (set(self.keys()) - {"wav"}) != (set(other.keys()) - {"wav"}):
#             return False
#         for k in self.keys():
#             if k != "wav" and self[k] != other[k]:
#                 return False
#         return True

#     def __ne__(self, other):
#         return not self.__eq__(other)

#     def __setitem__(self, key, value):
#         if len(self):
#             if value.duration != self.duration:
#                 raise AssertionError("duration does not match")
#             if value.fs != self.fs:
#                 raise AssertionError("fs does not match")
#         dict.__setitem__(self, key, value)

#     def __str__(self):
#         s = ""
#         for key, track in self.items():
#             s += "%s: %s\n" % (key, track)
#         return s

#     def __add__(self, other):
#         if self is other:
#             other = copy.deepcopy(other)
#         obj = type(self)()
#         for k in self:  # .iterkeys():
#             obj[k] = self[k] + other[k]
#         return obj

#     def resample(self, fs):
#         multiTrack = type(self)()
#         for key, track in self.items():
#             multiTrack[key] = track.resample(fs)
#         return multiTrack

#     def crossfade(self, other, length):
#         """
#         append multiTrack to self, using a crossfade of a specified length in samples
#         """
#         assert type(self) == type(other)
#         assert self.keys() == other.keys()
#         assert self.fs == other.fs
#         assert isinstance(length, int)
#         assert length > 0
#         assert other.duration >= length
#         assert self.duration >= length
#         multiTrack = type(self)()
#         for key, _ in self.items():
#             multiTrack[key] = self[key].crossfade(other[key], length)
#         return multiTrack

#     def select(self, a, b, keys=None):
#         assert a >= 0
#         assert a < b  # or a <= b?
#         assert b <= self.duration
#         """return a new multitrack object with all track views from time a to b"""
#         if keys is None:
#             keys = self.keys()
#         multiTrack = type(self)()
#         for key in keys:
#             multiTrack[key] = self[key].select(a, b)
#         return multiTrack

#     # TODO: should this be deprecated in favor of / should this call - the more general time_warp function?
#     def scale_duration(self, factor):
#         if factor != 1:
#             for t in self.values():
#                 if isinstance(t, Partition):
#                     t.time *= (
#                         factor
#                     )  # last time parameter IS duration, so no worries about duration
#                 elif isinstance(t, TimeValue) or isinstance(t, Event):
#                     if factor > 1:  # make room for expanded times
#                         t.duration = int(t.duration * factor)
#                         t.time *= factor
#                     else:
#                         t.time *= factor
#                         t.duration = int(t.duration * factor)
#                 else:
#                     raise NotImplementedError  # wave?

#     def time_warp(self, x, y):
#         """in-place"""
#         for track in iter(self.values()):
#             track.time_warp(x, y)

#     default_suffix = ".mtt"

#     @classmethod
#     def read(cls, name):
#         """Loads info about stored tracks from name, adding extension if missing,
#         and loads tracks by calling read(<name without extension>) for them.
#         """
#         name_wo_ext = os.path.splitext(name)[0]
#         if name == name_wo_ext:
#             name += cls.default_suffix
#         with open(name, "rb") as mtt_file:
#             track_infos = json.load(mtt_file)
#         self = cls()
#         for track_type_name, track_info_list in track_infos:
#             track_type = globals()[track_type_name]
#             track_info = dict(track_info_list)
#             track = track_type.read(name_wo_ext, **track_info)
#             self[track_info["track_name"]] = track
#         return self

#     def write(self, name):
#         """Saves info about stored tracks to name, adding extension if missing,
#         and calls write(<name without extension>) for the contained tracks.
#         Note!: not saving wav as long as wav writing/reading is erroneous
#         """
#         name_wo_ext = os.path.splitext(name)[0]
#         if name == name_wo_ext:
#             name += self.default_suffix
#         track_infos = []  # list of dicts storing track info
#         for track_name, track in sorted(self.items()):
#             if track_name == "wav":
#                 continue
#             track_info = {
#                 "track_name": track_name,
#                 "fs": int(track.get_fs()),
#                 "duration": int(track.get_duration()),
#             }
#             if type(track) == Value:
#                 track_info.update({"value_type": type(track.get_value()).__name__})
#             track.write(name_wo_ext, **track_info)
#             track_infos.append((type(track).__name__, sorted(track_info.items())))
#         with open(name, "wb") as mtt_file:
#             json.dump(track_infos, mtt_file)


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


# class TestTimeValue(unittest.TestCase):
#     def setUp(self):
#         self.t1 = TimeValue((numpy.linspace(1, 9, 3)).astype(TIME_TYPE), numpy.array([1, 4, 2]), 1, 10)
#         self.t2 = TimeValue((numpy.linspace(2, 8, 4)).astype(TIME_TYPE), numpy.array([1, 4, 8, 2]), 1, 10)
#         self.s1 = TimeValue(numpy.array([0, 1, 2], dtype=TIME_TYPE),
#                             numpy.array(['.pau', 'aI', '.pau'], dtype='S8'),
#                             1,
#                             3)
#         self.s2 = TimeValue(numpy.array([0, 1, 2], dtype=TIME_TYPE),
#                             numpy.array(['.pau', 'oU', '.pau'],
#                             dtype='S8'),
#                             1,
#                             3)
#         desc = numpy.dtype({"names": ['string', 'int'], "formats": ['S30', numpy.uint8]})  # record arrays
#         self.r1 = TimeValue(numpy.array([0, 1], dtype=TIME_TYPE),
#                             numpy.array([('abc', 3), ('def', 4)], dtype=desc),
#                             1,
#                             2)
#         self.f1 = TimeValue(numpy.array([0, 1], dtype=TIME_TYPE),
#                             numpy.array([numpy.arange(3), numpy.arange(4)], dtype=numpy.ndarray),
#                             1,
#                             2)

#     def test_init(self):
#         TimeValue(numpy.empty(0, dtype=TIME_TYPE), numpy.empty(0), 1, 0)  # empty
#         TimeValue(numpy.empty(0, dtype=TIME_TYPE), numpy.empty(0), 1, 10)  # empty
#         self.assertRaises(AssertionError,
#                           TimeValue,
#                           numpy.array([6, 3], dtype=TIME_TYPE),
#                           numpy.array([3, 6]), 1, 10)  # bad times
#         self.assertRaises(Exception,
#                           TimeValue,
#                           numpy.array([3, 6], dtype=TIME_TYPE),
#                           numpy.array([3, 6]), 1, 5)  # duration too short

#     def test_duration(self):
#         self.t1.duration = 11  # ok
#         self.assertRaises(Exception, self.t1.set_duration, 5)  # duration too short
#         self.s1.duration = 3  # ok
#         self.r1.duration = 3  # ok

#     def test_eq(self):
#         self.assertTrue(self.t1 == self.t1)
#         self.assertFalse(self.t1 == self.t2)

#     def test_add(self):
#         t = self.t1 + self.t2
#         self.assertTrue(t.duration == 20)
#         self.assertTrue(t.time[5] == 16)
#         self.assertTrue(t.value[5] == 8)
#         self.t1 += self.t2
#         t = self.t1
#         self.assertTrue(t.duration == 20)
#         self.assertTrue(t.time[5] == 16)
#         self.assertTrue(t.value[5] == 8)
#         self.s1 + self.s2
#         self.r1 + self.r1

#     def test_select(self):
#         t = self.t1.select(1, 5)
#         self.assertTrue(t == TimeValue(numpy.array([0], dtype=TIME_TYPE), numpy.array([1]), 1, 4))
#         t = self.t1.select(1, 6)
#         self.assertTrue(t == TimeValue(numpy.array([0, 4], dtype=TIME_TYPE), numpy.array([1, 4]), 1, 5))
#         t = self.t1.select(1, 6)
#         self.assertTrue(t == TimeValue(numpy.array([0, 4], dtype=TIME_TYPE), numpy.array([1, 4]), 1, 5))
#         t = self.t1.select(2, 5)
#         self.assertTrue(t == TimeValue(numpy.array([], dtype=TIME_TYPE), numpy.array([]), 1, 3))


# class TestMultiTrack(unittest.TestCase):
#     def setUp(self):
#         self.e = Event(numpy.array([3, 6], dtype=TIME_TYPE), 1, 10)
#         self.w = Wave(numpy.arange(0, 10, dtype=numpy.int16), 1)
#         self.t = TimeValue((numpy.linspace(1, 9, 3)).astype(TIME_TYPE), numpy.array([1, 4, 2]), 1, 10)
#         self.p = Partition(numpy.array([0, 5, 6, 10], dtype=TIME_TYPE), numpy.array(["start", "middle", "end"]), 1)
#         self.m = MultiTrack({"e": self.e, "w": self.w, "t": self.t, "p": self.p})

#     def test_str(self):
#         str(self.m)

#     # def test_add(self):
#     #     answer = self.multiTrack1 + self.multiTrack2
#     #     self.assertTrue(self.mResult == answer)
#     #     answer = copy.copy(self.multiTrack1)
#     #     answer += self.multiTrack2
#     #     self.assertTrue(answer == self.mResult)

#     def test_resample(self):
#         m = self.m.resample(2)
#         self.assertTrue(m["e"].duration == m["w"].duration == m["t"].duration == m["p"].duration == 20)
#         self.assertTrue(m["e"].time[0] == 6)

#     # def test_select1(self):
#     #     ws = self.wave.select(0, self.wave.duration)
#     #     self.assertTrue(ws == self.wave)

#     # def test_select2(self):
#     #     tv = TimeValue(numpy.array([1, 5, 9]), numpy.array([1., 4., 2.]), 1, duration=10)
#     #     tv1 = tv.select(0, 2, interpolation="linear")
#     #     tv2 = tv.select(2, 10, interpolation="linear")
#     #     tvs = tv1 + tv2
#     #     v1 = tv.get(numpy.linspace(0., 10., 11), interpolation="linear").transpose()
#     #     v2 = tvs.get(numpy.linspace(0., 10., 11), interpolation="linear").transpose()
#     #     print v1
#     #     print v2
#     #     self.assertAlmostEqual( numpy.sum(numpy.abs(v1 - v2)), 0)

#     # def test_select3(self):
#     #     pt = Partition(fs=1, time=numpy.array([0., 6, 12, 18]), value=['.pau','h', 'E'], duration=18)
#     #     pt1 = pt.select(4.5, 12.5)
#     #     #print pt1


# class TestCrossfade(unittest.TestCase):
#     def test_wave(self):
#         wav1 = Wave(numpy.array([1,  1,  1,  1,  1]), 1)
#         wav2 = Wave(numpy.array([10, 10, 10, 10, 10]), 1)
#         length = 3
#         wav = wav1.crossfade(wav2, length)
#         self.assertEqual(wav1.duration + wav2.duration - length, wav.duration)
#         self.assertTrue(numpy.allclose(wav.value, numpy.array([1, 1, 3, 5, 7, 10, 10])))

#     def test_event(self):
#         evt1 = Event(numpy.array([1, 5, 9], dtype=TIME_TYPE), 1, 10)
#         evt2 = Event(numpy.array([2, 5, 9], dtype=TIME_TYPE), 1, 10)
#         length = 2
#         evt = evt1.crossfade(evt2, length)
#         self.assertEqual(evt1.duration + evt2.duration - length, evt.duration)
#         self.assertTrue(numpy.allclose(evt.time, numpy.array([1, 5, 10, 13, 17])))

#     def test_partition(self):
#         prt1 = Partition(numpy.array([0, 8, 10], dtype=TIME_TYPE), numpy.array(['1', '2']), 1)
#         prt2 = Partition(numpy.array([0, 2, 10], dtype=TIME_TYPE), numpy.array(['3', '4']), 1)
#         length = 4
#         prt = prt1.crossfade(prt2, length)
#         self.assertEqual(prt1.duration + prt2.duration - length, prt.duration)
#         self.assertTrue(numpy.allclose(prt.time, numpy.array([0, 8, 16])))
#         self.assertTrue((prt.value == numpy.array(['1', '4'])).all())

#     def test_timeValue(self):
#         pass  # TODO: implement me!
