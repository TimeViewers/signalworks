import os
from pathlib import Path

import numpy
from signalworks.tracking.tracking import Track

TIME_TYPE = numpy.int64


class Value(Track):
    """can store a singular value (e.g. comment string) of any type"""

    default_suffix = ".trk"

    def __init__(self, value, fs, duration, path=None):
        super().__init__(path)
        if path is None:
            path = str(id(self))
        self.path = Path(path).with_suffix(self.default_suffix)
        assert isinstance(fs, int)
        assert fs > 0
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        self._value = value
        self._fs = fs
        self._duration = duration

    def __getitem__(self, k):
        return self.value[k]

    def __setitem__(self, k, v):
        self.value[k] = v

    def __eq__(self, other):
        if (
            (self._fs == other._fs)
            and (self._duration == other._duration)
            and isinstance(other.value, type(self.value))
        ):
            return self.value == other.value
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __len__(self):
        return 1

    def __str__(self):
        return self._value

    def get_duration(self):
        return self._duration

    def set_duration(self, duration):
        assert isinstance(duration, TIME_TYPE) or isinstance(duration, int)
        self._duration = duration

    duration = property(get_duration, set_duration, doc="duration of track")

    def get_fs(self):
        return self._fs

    def set_fs(self, _value):
        raise Exception("Cannot change fs, try resample()")

    fs = property(get_fs, set_fs, doc="sampling frequency")

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    value = property(get_value, set_value)

    def select(self, a, b):
        raise NotImplementedError

    def time_warp(self, _x, y):
        self.duration = y[-1]  # changed this from _duration

    @classmethod
    def identify_ext(cls, name, track_name=None):
        if name == os.path.splitext(name)[0]:
            if track_name is None:
                raise ValueError("neither track name nor extension specified")
            elif track_name == "input":
                ext = ".txt"
            elif track_name == "xml":
                ext = ".xml"
            elif track_name == "search":
                ext = ".search"
            elif track_name == "pronunciation":
                ext = ".pron"
            else:
                raise ValueError("unknown track name '{}'".format(track_name))
        else:
            ext = os.path.splitext(name)[1]
        return ext

    @classmethod
    def read(cls, name, fs, duration, track_name=None, *_args, **_kwargs):
        """Loads object from name, adding extension if missing."""
        ext = cls.identify_ext(name, track_name)
        name = os.path.splitext(name)[0] + ext
        if ext in (".txt", ".xml", ".search"):
            with open(name, "r") as f:
                value = f.read()
                if ext == ".search":
                    value = str(value)  # , "UTF-8")
                self = Value(value, fs, duration)
        elif ext == ".pron":
            with open(name, "r") as f:
                pron = [tuple(line.split("\t")) for line in f.read().split("\n")]
                self = Value(pron, fs, duration)
        else:
            raise ValueError("unknown extension '{}'".format(ext))
        return self

    def write(self, name, track_name=None, *_args, **_kwargs):
        """
        Saves object to name, adding extension if missing.
        """
        ext = self.identify_ext(name, track_name)
        name = os.path.splitext(name)[0] + ext
        if ext in (".txt", ".xml", ".search"):
            with open(name, "w") as f:
                f.write(self.get_value())
        elif ext == ".pron":
            with open(name, "w") as f:
                f.write(
                    "\n".join(word + "\t" + pron for word, pron in self.get_value())
                )
        else:
            raise ValueError("unknown extension '{}'".format(ext))
