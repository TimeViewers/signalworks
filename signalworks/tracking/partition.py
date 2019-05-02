import codecs
import logging
import os
from pathlib import Path

import numpy
from signalworks.tracking import LabreadError
from signalworks.tracking.tracking import Track

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

TIME_TYPE = numpy.int64


class Partition(Track):
    default_suffix = ".lab"

    def check(self):
        assert isinstance(self._time, numpy.ndarray)
        assert self._time.ndim == 1
        assert self._time.dtype == TIME_TYPE
        # assert (numpy.diff(self._time.astype(numpy.float)) > 0).all(),\
        #     "times must be strictly monotonically increasing"
        if not (numpy.diff(self._time.astype(numpy.float)) > 0).all():
            logger.warning("Partition: times must be strictly monotonically increasing")
        assert isinstance(self._value, numpy.ndarray)
        # assert self._value.ndim == 1 # TODO: can I remove this?
        assert isinstance(self._fs, int)
        assert self._fs > 0
        # if len(self._time):
        assert self._time[0] == 0, "partition must begin at time 0"
        # assert (numpy.diff(self._time) > 0).all(), "zero-duration labels are not permitted"
        if not (numpy.diff(self._time) > 0).all():
            logger.warning("Partition: zero-duration labels are not permitted")
        # this means an empty partition contains one time value at 0!!!
        assert (
            len(self._time) == len(self._value) + 1
        ), "length of time and value+1 must match"
        # else:
        #    assert len(self._value) == 0
        return True

    def __init__(self, time, value, fs, path=None):
        super().__init__(path)
        if path is None:
            path = str(id(self))
        self.path = Path(path).with_suffix(self.default_suffix)
        self._time = time
        self._value = value
        self._fs = fs
        assert self.check()

    def get_time(self):
        # assert (numpy.diff(self._time.astype(numpy.float)) > 0).all(),\
        # "times must be strictly monotonically increasing" # in case the user messed with .time[index] directly
        if not (numpy.diff(self._time.astype(numpy.float)) > 0).all():
            logger.warning("get_time times must be strictly monotonically increasing")
        return self._time

    def set_time(self, time):
        assert isinstance(time, numpy.ndarray)
        assert time.ndim == 1
        assert time.dtype == TIME_TYPE
        if len(time):
            assert time[0] == 0, "partition must begin at time 0"
            # assert (numpy.diff(time) > 0).all(), "zero-duration labels are not permitted"
            # assert (numpy.diff(time) >= 0).all(), "zero-duration labels are not permitted"
            if not (numpy.diff(time) > 0).all():
                logger.warning("encorter zero-duration labels are not permitted")
            assert (
                len(time) == len(self._value) + 1
            ), "length of time and value+1 must match"
        else:
            assert len(self._value) == 0
        self._time = time

    time = property(get_time, set_time)

    def get_value(self):
        return self._value

    def set_value(self, value):
        assert isinstance(value, numpy.ndarray)
        # assert value.ndim == 1
        assert (
            len(self._time) == len(self._value) + 1
        ), "length of time and value must match"
        self._value = value

    value = property(get_value, set_value)

    def get_duration(self):
        if len(self._value):
            return self._time[-1]
        else:
            return 0  # or None, or raise Exception

    def set_duration(self, duration):
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        if len(self._value):
            assert (
                duration > self._time[-2]
            ), "can't set duration to a smaller or equal value than the next-to-last boundary (this would result in losing the last value)"
            self._time[-1] = duration
        else:
            if duration != 0:
                raise Exception(
                    "cannot set duration of an empty Partition to anything but 0"
                )

    duration = property(get_duration, set_duration, doc="duration of track in fs")

    def get_fs(self):
        return self._fs

    def set_fs(self, _value):
        raise Exception("Cannot change fs, try resample()")

    fs = property(get_fs, set_fs, doc="sampling frequency")

    def __eq__(self, other):
        try:
            if (
                (self._fs == other._fs)
                and (len(self._time) == len(other._time))
                and (len(self._value) == len(other._value))
                and (self._time == other._time).all()
                and (self._value == other._value).all()
            ):
                return True
        except AttributeError:
            pass
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return len(self._value)

    # def resample(self, fs):
    #     if fs != self._fs:
    #         factor = fs / self._fs
    #         self._time = numpy.round(factor * self._time).astype(TIME_TYPE)
    #         assert (numpy.diff(self._time) > 0).all(),\
    #             "new fs causes times to fold onto themselves due to lack in precision"
    #         self._fs = fs

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
            return type(self)(time, self._value, fs)
        else:
            return self

    def select(self, a, b):
        assert 0 <= a
        assert a < b
        assert b <= self._time[-1]
        if __debug__:  # consistency check
            self.check()
        ai = self._time.searchsorted(a)
        bi = self._time.searchsorted(b)
        if a < self._time[ai]:  # prepend
            ai -= 1
        if self._time[bi] < b:  # append
            bi += 1
        value = self._value[ai:bi]
        time = self.time[
            ai : bi + 1
        ].copy()  # otherwise we are going to modify the original below!
        # if a < self._time[ai]:  # prepend
        time[0] = a
        # if self._time[bi] < b:  # append
        time[-1] = b
        time = time - a  # must make a copy!
        return type(self)(time.astype(TIME_TYPE), value, self.fs)

    @classmethod
    def read(cls, name, fs=300_000, track_name=None, *_args, **_kwargs):
        """Loads object from name, adding default extension if missing.
        It figures out which read function is appropriate based on extension.
        """
        if name == os.path.splitext(name)[0]:
            assert track_name, "neither track name nor extension specified"
            ext = "." + track_name
            name += ext
        else:
            ext = os.path.splitext(name)[1].lower()[1:]
        if ext in ("lab", "dem", "rec"):
            return cls.read_lab(name, fs)
        elif ext == "textgrid":
            return cls.read_textgrid(name, fs)
        else:
            try:
                return cls.read_lab(name, fs)
            except FileNotFoundError:  # TODO: need to catch more exceptions?
                pass
            try:
                return cls.read_textgrid(name, fs)
            except FileNotFoundError:  # TODO: need to catch more exceptions?
                pass
            raise ValueError("file '{}' has unknown format".format(name))

    @classmethod
    def read_lab(
        cls, name: str, fs: int = 300_000, encoding: str = "UTF8"
    ) -> "Partition":
        """load times, values from a label file"""
        with codecs.open(name, "r", encoding=encoding) as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise ValueError("file '{}' is empty".format(name))
        time: list = []
        value: list = []
        # label_type = numpy.float64
        for i, line in enumerate(lines):
            try:
                tmp1, tmp2, label = line[:-1].split()
            except ValueError:
                logger.warning(
                    'ignoring line "%s" in file %s at line %i' % (line, name, i + 1)
                )
                continue
            t1 = float(tmp1)
            t2 = float(tmp2)
            if label[-1] == "\r":
                label = label[:-1]
            if len(time) == 0:
                time.append(t1)
            else:
                if time[-1] != t1:
                    logger.warning(
                        'noncontiguous label "%s" in file %s at line %i, fixing'
                        % (label, name, i + 1)
                    )
            dur = t2 - time[-1]
            if dur > 0:
                time.append(t2)
                value.append(label)
            elif dur == 0:
                logger.warning(
                    'zero-length label "%s" in file %s at line %i, ignoring'
                    % (label, name, i + 1)
                )
            else:
                raise LabreadError(
                    "label file contains times that are not monotonically increasing"
                )
        if len(time) == 0:
            raise (Exception("file is empty or all lines were ignored"))
        if time[0] != 0:
            logger.warning("moving first label boundary to zero")
            time[0] = 0
            # or insert a first label
            # time.insert(0, 0)
            # value.insert(0, default_label)
        time = numpy.round(numpy.array(time) * fs).astype(TIME_TYPE)
        # assert labels are not longer than 8 characters
        # value = numpy.array(value, dtype='U16' if label_type is str else numpy.float64)
        value = numpy.array(
            value, dtype="U16"
        )  # if label_type is str else numpy.float64)
        return Partition(
            time, value, fs=fs, path=name
        )  # u1p to 16 characters (labels could be words)

    @classmethod
    def read_partition(cls, name, fs=48000, encoding="UTF8"):
        """load times, values from a label file"""
        with codecs.open(name, "r", encoding=encoding) as f:
            lines = f.readlines()
        if len(lines) == 0:
            raise ValueError("file '{}' is empty".format(name))
        time = [0.0]
        value = []
        label_type = numpy.float64
        for i, line in enumerate(lines):
            try:
                temp, _t, label = line[:-1].split()
            except ValueError:
                logger.warning(
                    'ignoring line "%s" in file %s at line %i' % (line, name, i + 1)
                )
                continue
            t = float(temp)
            if label[-1] == "\r":
                label = label[:-1]
            try:
                label = label_type(label)
            except ValueError:
                if len(time) == 1:
                    label_type = str
                    label = label_type(label)
                else:
                    raise
            if len(time) == 1:
                dur = t
            else:
                dur = t - time[-1]

            time.append(t)
            if dur > 0:
                value.append(label)
            elif dur == 0:
                logger.warning(
                    'zero-length label "%s" in file %s at line %i, ignoring'
                    % (label, name, i + 1)
                )
            else:
                raise LabreadError(
                    "label file contains times that are not monotonically increasing"
                )
        if len(time) == 1:
            raise (Exception("file is empty or all lines were ignored"))
        if time[0] != 0:
            logger.warning("moving first label boundary to zero")
            time[0] = 0
            # or insert a first label
            # time.insert(0, 0)
            # value.insert(0, default_label)
        time = numpy.round(numpy.array(time) * fs).astype(TIME_TYPE)
        # assert labels are not longer than 8 characters
        value = numpy.array(value, dtype="U16" if label_type is str else numpy.float64)
        return Partition(
            time, value, fs=fs
        )  # u1p to 16 characters (labels could be words)

    def write(self, name):
        """Saves object to name, adding default extension if missing."""
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            assert name, "neither track name nor extension specified"
            name += self.default_suffix
        self.write_lab(name)

    def write_lab(self, file):
        if hasattr(file, "read"):
            f = file
        else:
            f = open(file, "w")
        for i, v in enumerate(self._value):
            f.write(
                "%f %f %s\n" % (self._time[i] / self.fs, self._time[i + 1] / self.fs, v)
            )
        f.close()

    def write_partition(self, name):
        # written to generate a format exactly like
        # CMU-ARCTIC lab/*.lab files
        f = open(name, "w")
        f.write("#\n")
        for i, v in enumerate(self._value):
            f.write(f"{self._time[i + 1] / self.fs} 125 {v}\n")
        f.close()

    @classmethod
    def from_TimeValue(cls, tv):
        """convert a time value track with repeating values into a partition track"""
        # import TimeValue here to avoid circular dependencies
        from signalworks.tracking.timevalue import TimeValue

        assert isinstance(tv, TimeValue)
        boundary = numpy.where(numpy.diff(tv.value))[0]
        time = numpy.empty(len(boundary), dtype=tv.time.dtype)
        value = numpy.empty(len(boundary) + 1, dtype=tv.value.dtype)
        duration = tv.duration
        for i in range(len(time)):
            index = boundary[i]
            time[i] = (tv.time[index] + tv.time[index + 1]) / 2
        time = numpy.concatenate(([0], time, [duration])).astype(TIME_TYPE)
        value[0] = tv.value[0]
        for i in range(len(boundary)):
            value[i + 1] = tv.value[boundary[i] + 1]
        par = Partition(time, value, tv.fs, path=tv.path)
        assert par.check()
        return par

    def __getitem__(self, index):
        return self._time[index], self._value[index], self._time[index + 1]

    def __setitem__(self, index, value):
        raise NotImplementedError  # this would be some work to support

    def __str__(self):
        s = [""]
        for i in range(len(self._value)):
            s.append(
                "#%i: %i/%.3f: %s :%i/%.3f\n"
                % (
                    i,
                    self._time[i],
                    self._time[i] / self._fs,
                    self._value[i],
                    self._time[i + 1],
                    self._time[i + 1] / self._fs,
                )
            )
        return f"{''.join(s)}\nfs={self._fs}\nduration={self.duration}"

    def __add__(self, other):
        if self._fs != other._fs:
            raise Exception("sampling frequencies must match")
        time = numpy.hstack(
            (self._time, self._time[-1] + other._time[1:])
        )  # other._time[0] == 0
        value = numpy.hstack((self._value, other._value))
        # duration = self.duration + other.duration
        return Partition(time, value, self.fs)

    def crossfade(self, partition, length):
        """append to self, using a crossfade of a specified length in samples"""
        assert type(self) == type(
            partition
        ), "Cannot add Track objects of different types"
        assert self.fs == partition.fs, "sampling frequency of waves must match"
        assert isinstance(length, int)
        assert length > 0
        assert partition.duration >= length
        assert self.duration >= length
        # cut left, cut right, concatenate, could be sped up
        a = self.select(0, self.duration - length // 2)
        b = partition.select(length - length // 2, partition.duration)
        return a + b

    def get(self, t):
        """returns current label at time t"""
        # return self.value[numpy.where((self.time - t) <= 0)[0][-1]] # last one that is <= 0
        return self._value[
            (self._time.searchsorted(t + 1) - 1).clip(0, len(self._value) - 1)
        ]

    def append(self, time, value):
        """appends a value at the end, time is new endtime"""
        assert time > self._time[-1]
        self._time = numpy.hstack((self._time, time))
        self._value = numpy.hstack(
            (self._value, value)
        )  # so values must be numpy objects after all ?

    def insert(self, time: float, value: str) -> None:
        """modifies partition object to include value at time - other times are unchanged"""
        assert not (time == self._time).any(), "this boundary exists already"
        index = numpy.searchsorted(self._time, numpy.array([time]))[
            0
        ]  # move _this_ index to the right
        self._time = numpy.hstack((self._time[:index], time, self._time[index:]))
        # so values must be numpy objects after all ?
        self._value = numpy.hstack((self._value[:index], value, self._value[index:]))

    def delete_merge_left(self, index):
        # TODO write unittests for me and my other half /
        # or replace by just CUT if it's possible to uniquely specify the latter
        """deletes a phoneme, leaves duration as is"""
        assert len(self._value) > 1
        assert 0 <= index < len(self._value)
        self._time = numpy.hstack((self._time[:index], self._time[index + 1 :]))
        self._value = numpy.hstack((self._value[:index], self._value[index + 1 :]))

    def delete_merge_right(self, index):
        """deletes a phoneme, leaves duration as is"""
        assert len(self._value) > 1
        assert 0 <= index < len(self._value)
        self._time = numpy.hstack((self._time[: index + 1], self._time[index + 2 :]))
        self._value = numpy.hstack((self._value[:index], self._value[index + 1 :]))

    def merge_same(self):  # rename to uniq?
        """return a copy of myself, except identical values will be merged"""
        if len(self._value) <= 1:
            return Partition(self._time, self._value, self.fs)
        else:
            time = [self._time[0]]
            value = [self._value[0]]
            for i, v in enumerate(self._value[1:]):
                if v != value[-1]:  # new value
                    time.append(self._time[i + 1])  # close last one, begin new one
                    value.append(v)
            time.append(self.duration)
            return Partition(
                numpy.array(time, dtype=TIME_TYPE), numpy.array(value), self.fs
            )

    def time_warp(self, X, Y):
        assert X[0] == 0
        assert Y[0] == 0
        assert X[-1] == self.duration
        time = numpy.interp(self.time, X, Y).astype(self.time.dtype)
        # assert len(numpy.where(numpy.diff(time) <= 0)[0]) == 0,\
        #     'some segment durations are non-positive' # Must allow this after all
        # self._time = time
        # may have to remove some collapsed items
        self._time = time
        while 1:
            index = numpy.where(numpy.diff(self._time) == 0)[0]
            if len(index):
                logger.warning(
                    'removing collapsed item #{}: "{}"'.format(
                        index[0], self.value[index[0]]
                    )
                )
                self.delete_merge_right(index[0])
            else:
                break
