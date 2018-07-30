# -*- coding: utf-8 -*-
from typing import NamedTuple, Tuple

from scipy import signal
from numpy import np

from signalworks import tracking
from processing import Processor


class Filter(Processor):
    name = "Linear Filter"
    # acquire = {'wave': tracking.Wave}
    acquire = NamedTuple("acquire", [("wave", tracking.Wave)])

    def __init__(self):
        super().__init__()
        # default is pre-emphasis
        self.parameters = {"B": np.array([1., -.95]), "A": np.array([1.])}

    def process(self, progressTracker=None, **kwargs) -> Tuple[tracking.Wave]:
        # Processor.process(self, **kwargs)
        if progressTracker is not None:
            self.progressTracker = progressTracker
        wav = self.data.wave
        x = wav.value
        self.progressTracker.update(10)
        y = signal.lfilter(self.parameters["B"], self.parameters["A"], x).astype(
            x.dtype
        )
        self.progressTracker.update(90)
        new_track = (
            tracking.Wave(
                y,
                fs=wav.fs,
                path=wav.path.with_name(wav.path.stem + "-filtered").with_suffix(
                    tracking.Wave.default_suffix
                ),
            ),
        )
        return new_track


class ZeroPhaseFilter(Filter):
    name = "Zero-phase Linear Filter"
    acquire = NamedTuple("acquire", [("wave", tracking.Wave)])

    def process(self, progressTracker=None, **kwargs) -> Tuple[tracking.Wave]:
        # Processor.process(self, **kwargs)
        if progressTracker is not None:
            self.progressTracker = progressTracker
        wav = self.data.wave
        x = wav.value
        self.progressTracker.update(10)
        y = signal.filtfilt(self.parameters["B"], self.parameters["A"], x).astype(
            x.dtype
        )
        self.progressTracker.update(90)
        return (
            tracking.Wave(
                y,
                fs=wav.fs,
                path=wav.path.with_name(wav.path.stem + "-0phasefiltered").with_suffix(
                    wav.path.suffix
                ),
            ),
        )
