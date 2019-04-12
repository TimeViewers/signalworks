import logging
from pathlib import Path

from signalworks.tracking.metatrack import MetaTrack
from signalworks.tracking.error import LabreadError

import numpy

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TIME_TYPE = numpy.int64

class Label(MetaTrack):
    """Like Partition, but label regions do NOT have to be contiguous"""

    default_suffix = ".lab"

    def check(self):
        """value[k] has beginning and ending times time[2*k] and time[2*k+1]"""
        assert isinstance(self._time, numpy.ndarray)
        assert self._time.ndim == 1
        assert self._time.dtype == TIME_TYPE
        # assert (numpy.diff(self._time.astype(numpy.float)) >= 0).all(),\
        # "times must be (non-strictly) monotonically increasing"
        if not (numpy.diff(self._time.astype(numpy.float)) >= 0).all():
            logger.warning(
                "Label times must be (non-strictly) monotonically increasing"
            )
        assert isinstance(self._value, numpy.ndarray)
        # assert self._value.ndim == 1 # TODO: can I remove this?
        assert isinstance(self._fs, int)
        assert self._fs > 0
        assert (
            numpy.diff(self._time[::2]) > 0
        ).all(), (
            "zero-duration labels are not permitted (but abutting labels are permitted)"
        )
        # this means an empty partition contains one time value at 0!!!
        assert len(self._time) == 2 * len(
            self._value
        ), "length of time and value *2 must match"
        assert isinstance(self._duration, TIME_TYPE) or isinstance(self._duration, int)
        assert not (
            len(self._time) and self._duration < self._time[-1]
        ), "duration is not >= times"
        return True

    def __init__(self, time, value, fs, duration, path=None):
        super().__init__()
        if path is None:
            path = str(id(self))
        self.path = Path(path).with_suffix(self.default_suffix)
        self._time = time
        self._value = value
        self._fs = fs
        self._duration = duration
        self.type = "Label"
        assert self.check()

    def get_time(self):
        if __debug__:
            self.check()
        return self._time

    def set_time(self, time):
        self._time = time
        if __debug__:
            self.check()

    time = property(get_time, set_time)

    def get_value(self):
        if __debug__:
            self.check()
        return self._value

    def set_value(self, value):
        self._value = value
        if __debug__:
            self.check()

    value = property(get_value, set_value)

    def get_duration(self):
        return self._duration

    def set_duration(self, duration):
        assert isinstance(duration, (TIME_TYPE, int, numpy.int64))
        if len(self.time) > 0:
            assert duration > self.time[-1]
        self._duration = duration

    duration = property(get_duration, set_duration, doc="duration of track")

    def __str__(self):
        # s = [u"0"]
        # for i in range(len(self._value)):
        #    s.append(':%s:%i/%.3f' % (self._value[i], self._time[i + 1], self._time[i + 1] / self._fs))
        s = [""]
        for i in range(len(self._value)):
            s.append(
                f"#{i}: %i/%.3f: %s :%i/%.3f\n"
                % (
                    self._time[2 * i],
                    self._time[2 * i] / self._fs,
                    self._value[i],
                    self._time[2 * i + 1],
                    self._time[2 * i + 1] / self._fs,
                )
            )
        s = "".join(s)
        return "%sfs=%i\nduration=%i" % (s, self._fs, self._duration)

    def __add__(self, other):
        if self._fs != other._fs:
            raise Exception("sampling frequencies must match")
        time = numpy.hstack(
            (self._time, self.duration + other._time)
        )  # other._time[0] == 0
        value = numpy.hstack((self._value, other._value))
        duration = self.duration + other.duration
        return Label(time, value, self._fs, duration)

    def __len__(self):
        return len(self._value)

    def resample(self, fs):
        """resample to a certain fs"""
        if fs != self._fs:
            factor = fs / self._fs
            time = numpy.round(factor * self._time).astype(TIME_TYPE)
            time[-1] = numpy.ceil(factor * self._time[-1])
            assert (
                numpy.diff(time) > 0
            ).all(), (
                "new fs causes times to fold onto themselves due to lack in precision"
            )
            duration = numpy.round(factor * self._duration).astype(TIME_TYPE)
            return type(self)(time, self._value, fs, duration)
        else:
            return self

    def select(self, a, b):
        assert 0 <= a
        assert a < b
        assert b <= self.duration
        if __debug__:  # consistency check
            self.check()
        ai = self._time.searchsorted(a)
        bi = self._time.searchsorted(b)
        if ai % 2 == 1:  # inside label
            ai -= 1
        if bi % 2 == 1:  # inside label
            bi += 1
        value = self._value[(ai // 2) : (bi // 2)]
        time = self.time[ai:bi]
        time = time - a
        duration = b - a
        if time[0] < 0:
            time[0] = 0
        if time[-1] > duration:
            time[-1] = duration
        return type(self)(time.astype(TIME_TYPE), value, self.fs, duration)

    @classmethod
    def read_lbl(cls, path, fs=300000, encoding="UTF8"):
        """load times, values from a .lbl label file"""
        with open(path, "r", encoding=encoding) as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise ValueError("file '{}' is empty".format(path))
        lines += "\n"  # make sure it's terminated
        time = []
        value = []
        # label_type = numpy.float64
        for i, line in enumerate(lines):
            if line != "\n":
                try:
                    t1, t2, label = line.split()
                except ValueError:
                    logger.warning(
                        'ignoring line "%s" in file %s at line %i' % (line, path, i + 1)
                    )
                    continue
                t1 = float(t1)  # / 1000  # this particular
                t2 = float(t2)  # / 1000  # file format
                if label[-1] == "\r":
                    label = label[:-1]
                dur = t2 - t1
                if dur > 0:
                    time.extend([t1, t2])
                    value.append(label)
                elif dur == 0:
                    logger.warning(
                        'zero-length label "%s" in file %s at line %i, ignoring'
                        % (label, path, i + 1)
                    )
                else:
                    raise LabreadError(
                        "label file contains times that are not monotonically increasing"
                    )
        if len(time) == 0:
            raise (Exception("file is empty or all lines were ignored"))
        time = numpy.round(numpy.array(time) * fs).astype(TIME_TYPE)
        # assert labels are not longer than X characters
        value = numpy.array(value)
        # best guess at duration (+1 sec)
        lab = Label(time, value, fs=fs, duration=TIME_TYPE(time[-1] + fs), path=path)
        return lab

    @classmethod
    def read(cls, *args, **kwargs):
        return cls.read_lbl(cls, *args, **kwargs)
