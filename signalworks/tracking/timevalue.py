import logging
import os
from collections import Iterable

import numpy
from signalworks.tracking import Partition
from signalworks.tracking.tracking import Track

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TIME_TYPE = numpy.int64


class TimeValue(Track):
    def __init__(self, time, value, fs, duration, path=None):
        super().__init__(path)
        assert isinstance(time, numpy.ndarray)
        assert time.ndim == 1
        assert time.dtype == TIME_TYPE
        assert (
            numpy.diff(time.astype(numpy.float)) > 0
        ).all(), "times must be strictly monotonically increasing"
        assert isinstance(value, numpy.ndarray)
        assert isinstance(fs, int)
        assert fs > 0
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        assert len(time) == len(value), "length of time and value must match"
        assert not (len(time) and duration <= time[-1]), "duration is not > times"
        self._time = time
        self._value: numpy.ndarray = value
        self._fs = fs
        self._duration = duration
        self.min = numpy.nanmin(value)
        self.max = numpy.nanmax(value)
        self.unit = ""
        self.label = ""
        self.path = path

    def get_time(self):
        assert (
            numpy.diff(self._time.astype(numpy.float)) > 0
        ).all(), (
            "times must be strictly monotonically increasing"
        )  # in case the user messed with .time[index] directly
        return self._time

    def set_time(self, time):
        assert isinstance(time, numpy.ndarray)
        assert time.ndim == 1
        assert time.dtype == TIME_TYPE
        assert not (len(time) and self._duration <= time[-1]), "duration is not > times"
        assert (
            numpy.diff(time.astype(numpy.float)) > 0
        ).all(), "times must be strictly monotonically increasing"
        assert len(time) == len(self._value), "length of time and value must match"
        self._time = time

    time = property(get_time, set_time)

    def get_value(self):
        return self._value

    def set_value(self, value):
        assert isinstance(value, numpy.ndarray)
        assert len(self._time) == len(value), "length of time and value must match"
        self._value = value

    value = property(get_value, set_value)

    def get_duration(self):
        return self._duration

    def set_duration(
        self, duration
    ):  # assume times are available, if not, this must be overridden
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        assert not (
            len(self._time) and duration <= self._time[-1]
        ), "duration is not > times"
        self._duration = duration

    duration = property(get_duration, set_duration, doc="duration")

    def get_fs(self):
        return self._fs

    def set_fs(self, _value):
        raise Exception("Cannot change fs, try resample()")

    fs = property(get_fs, set_fs, doc="sampling frequency")

    def __eq__(self, other):
        if (
            (self._fs == other._fs)
            and (self._duration == other._duration)
            and (len(self._time) == len(other._time))
            and (len(self._value) == len(other._value))
            and (self._time == other._time).all()
            and numpy.allclose(
                numpy.round(self._value, 3), numpy.round(other._value, 3)
            )
        ):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._time)

    def __str__(self):
        return f"{list(zip(self._time, self._value))}, fs={self._fs}, duration={self._duration}, path={self.path}"

    def __add__(self, other):
        assert type(other) == type(self), "Cannot add Track objects of different types"
        assert self.fs == other.fs, "sampling frequencies must match"
        time = numpy.concatenate(
            (self.time, (other.time + self.duration).astype(other.time.dtype))
        )
        value = numpy.concatenate((self.value, other.value))
        duration = self.duration + other.duration
        return type(self)(time, value, self.fs, duration)

    # def __iadd__(self, other):
    #     assert type(other) == type(self), "Cannot add Track objects of different types"
    #     assert self.fs == other.fs, "sampling frequencies must match"
    #     self._time = numpy.concatenate((self._time, (other._time + self._duration).astype(other.time.dtype)))
    #     self._value = numpy.concatenate((self._value, other.value))
    #     duration = self.duration + other.duration
    #     return self

    def resample(self, fs):
        if fs != self._fs:
            factor = fs / self._fs
            time = numpy.round(factor * self._time).astype(TIME_TYPE)
            assert (
                numpy.diff(time) > 0
            ).all(), (
                "new fs causes times to fold onto themselves due to lack in precision"
            )
            # need to use numpy.round for consistency - it's different from the built-in round
            duration = int(numpy.ceil(factor * self._duration))
            return type(self)(time, self._value, fs, duration)
        else:
            return self

    def select(self, a, b):
        """(SHOULD!) return a copy"""
        # assert isinstance(a, TIME_TYPE)  # this doesn't seem necessary
        # assert isinstance(b, TIME_TYPE)
        assert b > a
        ai = self.time.searchsorted(a)
        bi = self.time.searchsorted(b)
        time = self._time[ai:bi] - a  # not possible to make this a view
        value = self._value[ai:bi]  # this is a view - should probably be a copy.
        return type(self)(time, value, self.fs, TIME_TYPE(b - a))

    default_suffix = ".tmv"

    @classmethod
    def read(cls, name, track_name=None, fs=300_000, duration=None, *_args, **_kwargs):
        """Loads object from name, adding default extension if missing."""
        if name == os.path.splitext(name)[0]:
            if track_name:
                ext = "." + track_name
            else:
                ext = cls.default_suffix
            name += ext
        else:
            ext = os.path.splitext(name)[1].lower()
        if ext in (".pit", ".nrg", cls.default_suffix):
            self = cls.read_tmv(name, fs)
        elif ext == ".f0":
            self = cls.read_pitchtier(name, fs)[0]
        elif ext == ".pitchtier":
            self = cls.read_pitchtier(name, fs)
        else:
            self = None
            try:
                if self is None:
                    self = cls.read_f0(name, fs)[0]
            except FileNotFoundError:  # TODO: need more exceptions?
                pass
            try:
                if self is None:
                    self = cls.read_pitchtier(name, fs)
            except FileNotFoundError:  # TODO: need more exceptions?
                pass
            if self is None:
                raise ValueError("file '{}' has unknown format".format(name))
        if duration:
            self.set_duration(duration)
        self.path = name
        return self

    # @classmethod
    # def read_f0(cls, name, frameRate=0.01, frameSize=0.0075, fs=48000):
    #     # return TimeValue(numpy.ndarray([1]).astype(TIME_TYPE), numpy.ndarray([1]), fs, 1)
    #     # 4 fields for each frame, pitch, probability of voicing, local root mean squared measurements,
    #     # and the peak normalized cross-correlation value frame arguments are in seconds
    #     f = open(name, "r")
    #     lines = f.readlines()
    #     f.close()
    #     F = len(lines)
    #     time = numpy.round((numpy.arange(F) * frameRate + frameSize / 2) * fs).astype(
    #         TIME_TYPE
    #     )
    #     duration = numpy.round((F * frameRate + frameSize) * fs).astype(TIME_TYPE)
    #     # time = numpy.arange(len(lines)) * frameRate + frameSize / 2
    #     value_f0 = numpy.zeros(F, numpy.float32)
    #     value_vox = numpy.zeros(F, numpy.float32)
    #     import re
    #
    #     form = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$")
    #     for i, line in enumerate(lines):
    #         match = form.search(line)
    #         if not match:
    #             # logger.error('badly formatted f0 file: {}'.format("".join(lines)))
    #             raise Exception
    #             # return (TimeValue(numpy.array([0]), numpy.array([100]), fs, duration),
    #             #         Partition(numpy.array([0, duration]), numpy.array([0]), fs))
    #         f0, voicing, _energy, _xcorr = match.groups()
    #         value_f0[i] = float(f0)
    #         value_vox[i] = float(voicing)
    #     index = numpy.where(value_f0 > 0)[0]  # keep only nonzeros F0 values
    #     pit = TimeValue(time[index], value_f0[index], fs, duration)
    #     vox = Partition.from_TimeValue(
    #         TimeValue(time, value_vox, fs, duration)
    #     )  # return a Partition
    #     return pit, vox
    #
    # f0read = read_f0

    @classmethod
    def read_tmv(cls, name, fs=300000):
        obj = numpy.loadtxt(name)
        time = obj[:, 0]
        value = obj[:, 1]
        time = numpy.round(time * fs).astype(TIME_TYPE)
        duration = (time[-1] + 1).astype(TIME_TYPE)
        return TimeValue(time, value, fs, duration, path=name)

    @classmethod
    def read_frm(cls, name, fs=48000):
        f = open(name, "r")
        lines = f.readlines()
        f.close()
        frameRate = 0.01
        frameSize = 0.049
        time = numpy.round(
            (numpy.arange(len(lines)) * frameRate + frameSize / 2) * fs
        ).astype(TIME_TYPE)
        duration = (time[-1] + 1).astype(TIME_TYPE)
        for i, line in enumerate(lines):
            fb = numpy.array(line.split()).astype(numpy.float)
            if i == 0:
                value = numpy.zeros((len(lines), len(fb)), numpy.float64)
            value[i, :] = fb
        return TimeValue(time, value, fs, duration, path=name)

    frmread = read_frm

    def write(self, name, track_name=None, *_args, **_kwargs):
        """Saves object to name, adding default extension if missing."""
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            if track_name:
                ext = "." + track_name
            else:
                ext = self.default_suffix
            name += ext
        self.write_tmv(name)

    def write_f0(self, name, frameRate=0.01, frameSize=0.0075, fs=48000):
        raise NotImplementedError("TimeValue.write_f0: not yet implemented")

    def write_tmv(self, name):
        obj = numpy.c_[self._time / self._fs, self._value]
        numpy.savetxt(name, obj)
        # with open(name, 'w') as f:
        #     for tv in zip(self._time / self._fs, self._value):
        #         f.write('\t'.join(str(round(x, 16)) for x in tv) + '\n')

    def write_frm(self, name):
        with open(name, "w") as f:
            if isinstance(self._value[0], Iterable):
                for fb in self._value:
                    f.write("\t".join(str(round(f, 3)) for f in fb) + "\n")
            else:
                for fb in self._value:
                    f.write(str(round(fb, 3)) + "\n")

    @classmethod
    def from_Partition(cls, p):
        """convert a partition track into a time-value track"""
        assert isinstance(p, Partition)
        time = numpy.r_[p.time[:-1], p.time[-1] - 1, p.time[1:-1] - 1]
        time.sort()
        # double this to keep it constant
        value = numpy.array(
            [p.value[int(i)] for i in numpy.arange(0, len(p.value), 0.5)]
        )
        return TimeValue(time, value, p.fs, p.duration)

    # def __getitem__(self, index):  # should I make the default just the value?
    #     return (self._time[index], self._value[index])
    #     return self._value[index] # ??

    # def __setitem__(self, index, value):
    #     self._time[index] = value[0]
    #     self._value[index] = value[1]

    def get_index(self, t):
        """return the index of the nearest available data"""
        if len(self._time) == 0:
            raise Exception
        if t <= 0:
            return 0
        if t >= self._time[-1]:
            return len(self._value) - 1
        # t is within bounds
        ri = self._time.searchsorted(t)
        li = ri - 1
        rt = self._time[ri]
        lt = self._time[li]
        if (t - lt) < (rt - t):
            return li
        else:
            return ri

    def _get_value(self, t, interpolation="nearest"):
        # check bounds
        if len(self._time) == 0:
            raise Exception
        if t < self._time[0]:
            # raise Exception('out of bounds left')
            # logger.warning('out of bounds left')
            v = self._value[0]  # better? yes!
        elif t > self._time[-1]:
            # raise Exception('out of bounds right')
            # logger.warning('out of bounds right')
            v = self._value[-1]  # better? yes!
        else:  # t is within bounds
            ri = self._time.searchsorted(t)  # a numpy function
            rt = self._time[ri]
            rv = self._value[ri]
            if rt == t:  # exact hit
                v = rv
            else:
                li = ri - 1
                lt = self._time[li]
                lv = self._value[li]
                if interpolation == "nearest":
                    if (t - lt) < (rt - t):
                        v = lv
                    else:
                        v = rv
                else:  # linear
                    a = float(t - lt) / float(rt - lt)
                    v = a * rv + (1.0 - a) * lv
        return v

    def get(
        self, T, interpolation="nearest"
    ):  # start using interp_* methods, which are faster
        """return the values at times T"""
        if isinstance(T, numpy.ndarray):
            assert T.ndim == 1
            n = (
                self._value.shape[1] if self._value.ndim > 1 else 1
            )  # this can be done much better I'm sure
            V = numpy.empty((T.shape[0], n), dtype=self._value.dtype)
            for i, t in enumerate(T):
                # TODO: speed this up by using a cached version of the interpolate function
                V[i] = self._get_value(t, interpolation)
            return V
        else:
            return self._get_value(T, interpolation)

    def interp_linear(self, T):
        assert numpy.all(numpy.diff(self.time) > 0)
        return numpy.interp(T, self.time, self.value)

    def interp_nearest(self, T):
        assert numpy.all(numpy.diff(self.time) > 0)
        index = numpy.interp(T, self.time, numpy.arange(T))
        return self.value[numpy.round(index)]  # TODO: test me!

    def interpolate(self, T, kind="linear"):
        """kind : str or int, optional
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
        where 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of first, second or third order) or as an integer
        specifying the order of the spline interpolator to use.
        Default is 'linear'."""
        from scipy.interpolate import interp1d

        if len(self.time):
            f = interp1d(
                self.time,
                self.value,
                kind=kind,
                axis=0,
                copy=True,
                bounds_error=True,
                fill_value=numpy.nan,
            )
            return f(T)
        else:
            logger.warning("interpolating without data")
            return numpy.ones(len(T)) * numpy.nan

    # def select_index(self, a, b):
    #     """return indeces such that a <= time < b"""
    #     assert b > a
    #     return range(self.time.searchsorted(a), self.time.searchsorted(b))

    # def select(self, a, b):
    #     """copy a section of the track, interval [)"""
    #     #indexa = numpy.where((self.ti - a) >= 0)[0][0] # first inside
    #     #indexb = numpy.where((self.ti - b) <= 0)[0][-1] # last inside
    #     index = self.select_index(a, b)
    #     if len(index) == 0:
    #         if type(self._value) == numpy.ndarray:
    #             value = numpy.array([])
    #         else:
    #             value = []
    #         return type(self)(time=numpy.empty(0, dtype=TIME_TYPE), value=value, fs=self.fs, duration=b-a)
    #     # on the line above, I first had [0], which cause all subsequent += to be in int, causing a bad bug.
    #     # Perhaps this is the reason to choose all times as int
    #     ai = index[0]
    #     bi = index[-1]
    #     time = self._time[ai:bi] #.copy() - don't think that's needed here ...
    #     value = self._value[ai:bi] #.copy()
    #     ## now, to copy the track accurately, we have to go all the way to the edges, and capture the values there,
    #     ## in accordance to the underlying interpolation function
    #     ### limit selection
    #     ##if a < self._time[0]:
    #         ##a = self.ti[0]
    #     ##if b > self.ti[-1]:
    #         ##b = self.ti[-1]
    #     ###if a < self._time[index[0]]:  # prepend - not sure if this is needed
    #         ###time  = numpy.concatenate(([a], time))
    #         ###value = numpy.concatenate(([self.get(a, interpolation)], value))
    #     ## can't append in any way that make sense, I _think_ TODO: Check this
    #     ##if self.time[index[-1]] <= b:
    #         ##time = numpy.concatenate((time, [b]))
    #         ##value = numpy.concatenate((value, f(b).T))
    #     ##time -= time[0]
    #     time -= a # TODO: Check this!
    #     tv = type(self)(time=time, value=value, fs=self.fs, duration=b-a)
    #     assert type(self._value) == type(tv.value)
    #     return tv

    def time_warp(self, X, Y):
        assert X[0] == 0
        assert Y[0] == 0
        assert X[-1] == self.duration
        time = numpy.interp(self.time, X, Y).astype(self.time.dtype)
        if 0:
            from matplotlib import pylab

            pylab.plot(X, Y, "rx-")
            for x, y in zip(self.time, time):
                pylab.plot([x, x, 0], [0, y, y])
            pylab.show()
        assert len(numpy.where(numpy.diff(time) <= 0)[0]) == 0  # no collapsed items
        self._time = time  # don't do duration check yet
        self.duration = Y[-1]  # changed this from _duration

    def interp(self, time):  # TODO: unify with get()
        if self._value.ndim == 1:
            return numpy.interp(
                time, self._time, self._value, self._value[0], self._value[-1]
            )
        elif self._value.ndim == 2:
            value = numpy.empty(
                (len(time), self._value.shape[1]), dtype=self._value.dtype
            )
            for j in range(self._value.shape[1]):
                value[:, j] = numpy.interp(
                    time,
                    self._time,
                    self._value[:, j],
                    self._value[0, j],
                    self._value[-1, j],
                )
            return value
        else:
            raise Exception
