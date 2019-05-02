import contextlib
import logging
import os
from pathlib import Path

import numpy
from signalworks.tracking.tracking import Track

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TIME_TYPE = numpy.int64


class Event(Track):
    def __init__(self, time, fs, duration, path=None):
        super().__init__(path)
        if path is None:
            path = str(id(self))
        self.path = Path(path).with_suffix(".trk")
        assert isinstance(time, numpy.ndarray)
        assert time.ndim == 1
        assert time.dtype == TIME_TYPE
        assert (
            numpy.diff(time.astype(numpy.float)) > 0
        ).all(), "times must be strictly monotonically increasing"
        assert isinstance(fs, int)
        assert fs > 0
        # assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        assert not (len(time) and duration <= time[-1]), "duration is not > times"
        self._fs = fs
        self._time = time
        self._duration = TIME_TYPE(duration)

    def get_time(self):
        assert (
            numpy.diff(self._time.astype(numpy.float)) > 0
        ).all(), "times must be strictly monotonically increasing"
        # in case the user messed with .time[index] directly
        return self._time

    def set_time(self, time):
        assert isinstance(time, numpy.ndarray)
        assert time.ndim == 1
        assert time.dtype == TIME_TYPE
        assert (
            numpy.diff(time.astype(numpy.float)) > 0
        ).all(), "times must be strictly monotonically increasing"
        # assert (numpy.diff(time.astype(numpy.float)) >= 0).all(), "times must be strictly monotonically increasing"
        assert not (len(time) and self._duration <= time[-1]), "duration is not > times"
        self._time = time

    def get_value(self):
        raise Exception("No values exist for Events")

    def set_value(self, value):
        raise Exception("can't set values for Events")

    def get_duration(self):
        return self._duration

    def set_duration(self, duration):
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        assert not (
            len(self._time) and duration <= self._time[-1]
        ), "duration is not > times"
        self._duration = duration

    def get_fs(self):
        return self._fs

    def set_fs(self, _value):
        raise Exception("Cannot change fs, try resample()")

    time = property(get_time, set_time)
    value = property(get_value, set_value)
    duration = property(get_duration, set_duration, doc="duration of track")
    fs = property(get_fs, set_fs, doc="sampling frequency")

    def __len__(self):
        return len(self._time)

    def __str__(self):
        return "%s, fs=%i, duration=%i." % (self._time, self.fs, self.duration)

    def __add__(self, other):
        assert type(other) == type(self), "Cannot add Track objects of different types"
        assert self.fs == other.fs, "sampling frequencies must match"
        time = numpy.concatenate(
            (self.time, (other.time + self.duration).astype(other.time.dtype))
        )
        duration = self.duration + other.duration
        return type(self)(time, self.fs, duration)

    def __eq__(self, other):
        if (
            (self._fs == other._fs)
            and (self._duration == other._duration)
            and (len(self._time) == len(other._time))
            and (self._time == other._time).all()
        ):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def crossfade(self, event, length):
        """append wave to self, using a crossfade of a specified length in samples"""
        assert type(self) == type(event), "Cannot add Track objects of different types"
        assert self.fs == event.fs, "sampling frequency of waves must match"
        assert isinstance(length, int)
        assert length > 0
        assert event.duration >= length
        assert self.duration >= length
        # cut left, cut right, concatenate
        a = self.select(0, self.duration - length // 2)
        b = event.select(length - length // 2, event.duration)
        return a + b

    def resample(self, fs):
        if fs != self._fs:
            factor = fs / self._fs
            # need to use numpy.round for consistency - it's different from the built-in round
            duration = int(numpy.ceil(factor * self._duration))
            if len(self._time):
                time = numpy.round(factor * self._time).astype(TIME_TYPE)
                if (numpy.diff(time) == 0).any():
                    logger.warning(
                        "new fs causes times to fold onto themselves due to lack in precision, "
                        "eliminating duplicates"
                    )
                    time = numpy.unique(time)
                if duration <= time[-1]:  # try to fix this situation
                    if len(time) > 1:
                        if (
                            time[-2] == time[-1] - 1
                        ):  # is the penultimate point far enough away?
                            raise Exception(
                                "cannot adjust last time point to be smaller "
                                "than the duration of the track"
                            )
                    logger.warning(
                        "new fs causes last time point to be == duration, "
                        "retarding last time point by one sample"
                    )
                    time[-1] -= 1
            else:
                time = self._time
            return type(self)(time, fs, duration)
        else:
            return self

    def select(self, a, b):
        assert a >= 0
        assert b > a
        assert b <= self._duration
        ai = self.time.searchsorted(a)
        bi = self.time.searchsorted(b)
        time = self._time[ai:bi] - a
        return type(self)(time, self.fs, b - a)

    default_suffix = ".evt"

    @classmethod
    def read(cls, name, fs, duration, *_args, **_kwargs):
        """Loads object from name, adding default extension if missing."""
        if name == os.path.splitext(name)[0]:
            ext = cls.default_suffix
            name += ext
        else:
            ext = os.path.splitext(name)[1].lower()
        if ext == ".pml" or ext == cls.default_suffix:
            self = cls.read_pml(name, fs)
        elif ext == ".pp":
            self = cls.read_PointProcess(name, fs)
        else:
            raise ValueError("file '{}' has unknown format".format(name))
        if duration:
            self.set_duration(duration)
        return self

    @classmethod
    def read_pml(cls, name, fs=48000):
        with open(name, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            logger.warning("pmlread(): empty file")
            return Event(numpy.empty(0, dtype=TIME_TYPE), fs, 0)
        time = numpy.zeros(len(lines), TIME_TYPE)
        for i, line in enumerate(lines):
            token = line.split(" ")
            # t1 = token[0]
            t2 = token[1]
            time[i] = numpy.round(float(t2) * fs)
        if (numpy.diff(time) <= 0).any():
            logger.error(
                "events are too close (for fs=%i) in file: %s, merging events"
                % (fs, name)
            )
            time = numpy.unique(time)
        # we cannot truly know the duration, so we are giving it the minimum duration
        return Event(time, fs, int(time[-1] + 1))

    pmlread = read_pml

    @classmethod
    def read_pm(cls, name, fs, _duration):
        # suited for loading .pm files (pitch mark) that exist in CMU-ARCTIC
        with open(name, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            logger.warning("pmread(): empty file")
            return Event(numpy.empty(0, dtype=TIME_TYPE), fs, 0)
        time = numpy.zeros(len(lines), TIME_TYPE) - 1
        for i, line in enumerate(lines):
            token = line.split(" ")
            t1 = token[0]
            with contextlib.suppress(IndexError):
                time[i] = numpy.round(float(t1) * fs)
            # try:
            #     time[i] = numpy.round(float(t1) * fs)
            # except IndexError:
            #     continue
            # else:
            #     t2 = token[1]

        time = time[time != -1]
        if (numpy.diff(time) <= 0).any():
            logger.error(
                "events are too close (for fs=%i) in file: %s, merging events"
                % (fs, name)
            )
            time = numpy.unique(time)
        # if int(time[-1] + 1) >= duration:
        # time = time[:-1]
        # we cannot truly know the duration, so we are giving it the minimum duration
        return Event(time, fs, int(time[-1] + 1))

    def write(self, name, *_args, **_kwargs):
        """Saves object to name, adding default extension if missing."""
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            name += self.default_suffix
        self.write_pml(name)

    def write_pml(self, name):
        f = open(name, "w")
        t1 = 0.0
        for t in self.time:
            t2 = t / self.fs
            f.write("%f %f .\n" % (t1, t2))
            t1 = t2
        f.close()

    pmlwrite = write_pml

    # def __getitem__(self, index):
    #     return self._time[index]

    # def __setitem__(self, index, value):
    #     self._time[index] = value

    def get(self, t):
        if t in self._time:
            return True
        else:
            return False

    def draw_pg(self, **kwargs):
        raise NotImplementedError

    def time_warp(self, X, Y):
        assert X[0] == 0
        assert Y[0] == 0
        assert X[-1] == self.duration
        time = numpy.interp(self.time, X, Y).astype(self.time.dtype)
        if 0:
            # from matplotlib import pylab
            # pylab.plot(X, Y, 'rx-')
            # for x, y in zip(self.time, time):
            #     pylab.plot([x, x, 0], [0, y, y])
            # pylab.show()
            raise NotImplementedError
        # may have to remove some collapsed items
        assert len(numpy.where(numpy.diff(time) == 0)[0]) == 0
        self._time = time  # [index]
        self.duration = Y[-1]  # changed this from _duration

    #  TODO: NEEDS TESTING!
    def insert(self, a, t):  #
        assert isinstance(t, type(self))
        index = numpy.where((self.time >= a))[0][0]  # first index of the "right side"
        self._time = numpy.hstack(
            (self._time[:index], t._time + a, self._time[index:] + t.duration)
        )
        self._duration += t.duration
