# -*- coding: utf-8 -*-
from typing import NamedTuple, Optional, Tuple

import numpy as np
from scipy import signal
from signalworks.processors.processing import DefaultProgressTracker, Processor
from signalworks.tracking import Wave


class Filter(Processor):
    name = "Linear Filter"
    # acquire = {'wave': tracking.Wave}
    acquire = NamedTuple("acquire", [("wave", Wave)])

    def __init__(self):
        super().__init__()
        # default is pre-emphasis
        self.parameters = {"B": np.array([1.0, -0.95]), "A": np.array([1.0])}

    def process(
        self, progressTracker: Optional[DefaultProgressTracker] = None
    ) -> Tuple[Wave]:
        # Processor.process(self, **kwargs)
        if progressTracker is not None:
            self.progressTracker = progressTracker
        wav = self.data.wave
        assert isinstance(wav, Wave)
        x = wav.value
        self.progressTracker.update(10)
        y = signal.lfilter(self.parameters["B"], self.parameters["A"], x).astype(
            x.dtype
        )
        self.progressTracker.update(90)
        new_track = (
            Wave(
                y,
                fs=wav.fs,
                path=str(
                    wav.path.with_name(wav.path.stem + "-filtered").with_suffix(
                        Wave.default_suffix
                    )
                ),
            ),
        )
        return new_track


class ZeroPhaseFilter(Filter):
    name = "Zero-phase Linear Filter"
    acquire = NamedTuple("acquire", [("wave", Wave)])

    def process(
        self, progressTracker: Optional[DefaultProgressTracker] = None
    ) -> Tuple[Wave]:
        # Processor.process(self, **kwargs)
        if progressTracker is not None:
            self.progressTracker = progressTracker
        wav = self.data.wave
        assert isinstance(wav, Wave)
        x = wav.value
        self.progressTracker.update(10)
        y = signal.filtfilt(self.parameters["B"], self.parameters["A"], x).astype(
            x.dtype
        )
        self.progressTracker.update(90)
        assert isinstance(wav, Wave)
        return (
            Wave(
                y,
                fs=wav.fs,
                path=str(
                    wav.path.with_name(wav.path.stem + "-0phasefiltered").with_suffix(
                        wav.path.suffix
                    )
                ),
            ),
        )
