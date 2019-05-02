import copy
import json
import os
from collections import UserDict

from signalworks.tracking import Event, Partition, TimeValue, Value, Wave


class MultiTrack(UserDict):
    """
    A dictionary containing time-synchronous tracks of equal duration and fs
    """

    def __init__(self, mapping=None):
        if mapping is None:
            mapping = UserDict()
        UserDict.__init__(self, mapping)
        if __debug__:  # long assert - TODO: do this on mapping, and then assign
            self.check()

    def check(self):
        if len(self) > 1:
            for i, (key, track) in enumerate(self.items()):
                if track.fs != self.fs:
                    raise AssertionError(
                        f"all fs' must be equal, track #{i} ('{key}) does not match track #1"
                    )
                if track.duration != next(iter(self.values())).duration:
                    raise AssertionError(
                        f"all durations must be equal, track #{i} ('{key}'') does not match track #1"
                    )

    def get_fs(self):
        if len(self):
            return next(iter(self.values())).fs
        else:
            return 0  # or raise?

    def set_fs(self, fs):
        raise Exception("Cannot change fs, try resample()")

    fs = property(get_fs, set_fs, doc="sampling frequency")

    def get_duration(self):
        if len(self):
            if __debug__:  # long assert - TODO: do this on mapping, and then assign
                self.check()
            return next(iter(self.values())).duration
        else:
            return 0

    def set_duration(self, duration):
        raise Exception("The duration cannot be set, it is derived from its conents")

    duration = property(
        get_duration, set_duration, doc="duration, as defined by its content"
    )

    def __eq__(self, other):
        # excluding wav from comparison as long as wav writing/reading is erroneous
        if (set(self.keys()) - {"wav"}) != (set(other.keys()) - {"wav"}):
            return False
        for k in self.keys():
            if k != "wav" and self[k] != other[k]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __setitem__(self, key, value):
        if len(self):
            if value.duration != self.duration:
                raise AssertionError("duration does not match")
            if value.fs != self.fs:
                raise AssertionError("fs does not match")
        UserDict.__setitem__(self, key, value)

    def __str__(self):
        s = ""
        for key, track in self.items():
            s += "%s: %s\n" % (key, track)
        return s

    def __add__(self, other):
        if self is other:
            other = copy.deepcopy(other)
        obj = type(self)()
        for k in self:  # .iterkeys():
            obj[k] = self[k] + other[k]
        return obj

    def resample(self, fs):
        multiTrack = type(self)()
        for key, track in self.items():
            multiTrack[key] = track.resample(fs)
        return multiTrack

    def crossfade(self, other, length):
        """
        append multiTrack to self, using a crossfade of a specified length in samples
        """
        assert type(self) == type(other)
        assert self.keys() == other.keys()
        assert self.fs == other.fs
        assert isinstance(length, int)
        assert length > 0
        assert other.duration >= length
        assert self.duration >= length
        multiTrack = type(self)()
        for key, _ in self.items():
            multiTrack[key] = self[key].crossfade(other[key], length)
        return multiTrack

    def select(self, a, b, keys=None):
        assert a >= 0
        assert a < b  # or a <= b?
        assert b <= self.duration
        """return a new multitrack object with all track views from time a to b"""
        if keys is None:
            keys = self.keys()
        multiTrack = type(self)()
        for key in keys:
            multiTrack[key] = self[key].select(a, b)
        return multiTrack

    # TODO: should this be deprecated in favor of / should this call - the more general time_warp function?
    def scale_duration(self, factor):
        if factor != 1:
            for t in self.values():
                if isinstance(t, Partition):
                    t.time *= (
                        factor
                    )  # last time parameter IS duration, so no worries about duration
                elif isinstance(t, TimeValue) or isinstance(t, Event):
                    if factor > 1:  # make room for expanded times
                        t.duration = int(t.duration * factor)
                        t.time *= factor
                    else:
                        t.time *= factor
                        t.duration = int(t.duration * factor)
                else:
                    raise NotImplementedError  # wave?

    def time_warp(self, x, y):
        """in-place"""
        for track in iter(self.values()):
            track.time_warp(x, y)

    default_suffix = ".mtt"

    @classmethod
    def read(cls, name):
        """Loads info about stored tracks from name, adding extension if missing,
        and loads tracks by calling read(<name without extension>) for them.
        """
        name_wo_ext = os.path.splitext(name)[
            0
        ]  # TODO: upgrade all path stuff to pathlib
        if name == name_wo_ext:
            name += cls.default_suffix
        with open(name, "rb") as mtt_file:
            track_infos = json.load(mtt_file)
        self = cls()
        for track_type_name, track_info_list in track_infos:
            track_type = globals()[track_type_name]
            track_info: UserDict = UserDict(track_info_list)
            track = track_type.read(name_wo_ext, **track_info)
            self[track_info["track_name"]] = track
        return self

    @classmethod
    def read_edf(cls, path):
        raise NotImplementedError
        # TODO: adapt
        # the following is copied from elsewhere and won't work as is
        import pyedflib

        with pyedflib.EdfReader(str(path)) as f:
            labels = f.getSignalLabels()
            for label in labels:
                index = labels.index(label)
                wav = Wave(f.readSignal(index), f.getSampleFrequency(index))
                wav.label = label
                wav.path = f.with_name(f.stem + "-" + label + ".wav")
                wav.min = f.getPhysicalMinimum(index)
                wav.max = f.getPhysicalMaximum(index)
                wav.unit = f.getPhysicalDimension(index)
                # self.add_view(wav, panel_index=panel_index, y_min=wav.min, y_max=wav.max)

    @classmethod
    def read_xdf(cls, path):
        raise NotImplementedError
        import openxdf

        # TODO: below is a place holder and needs to be finalize
        xdf = openxdf.OpenXDF(path)
        signals = openxdf.Signal(xdf, path.with_suffix(".nkamp"))
        # TODO: automate this, why are the xdf.header names different from signals.list_channels?
        for label in ["ECG", "Chin"]:
            # logger.info(f'reading {label} channel')
            sig = signals.read_file(label)[label]
            wav = Wave(sig.ravel(), 200)
            wav.label = label
            # wav.path = file.with_name(file.stem + '-' + label + '.wav')
            wav.min = -3200
            wav.max = 3200
            wav.unit = "1"
            # self.add_view(wav, panel_index=panel_index, y_min=wav.min, y_max=wav.max)

    def write(self, name):
        """Saves info about stored tracks to name, adding extension if missing,
        and calls write(<name without extension>) for the contained tracks.
        Note!: not saving wav as long as wav writing/reading is erroneous
        """
        name_wo_ext = os.path.splitext(name)[0]
        if name == name_wo_ext:
            name += self.default_suffix
        track_infos = []  # list of dicts storing track info
        for track_name, track in sorted(self.items()):
            if track_name == "wav":
                continue
            track_info = {
                "track_name": track_name,
                "fs": int(track.get_fs()),
                "duration": int(track.get_duration()),
            }
            if type(track) == Value:
                track_info.update({"value_type": type(track.get_value()).__name__})
            track.write(name_wo_ext, **track_info)
            track_infos.append((type(track).__name__, sorted(track_info.items())))
        with open(name, "wt") as mtt_file:
            json.dump(track_infos, mtt_file)
