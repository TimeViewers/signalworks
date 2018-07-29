# -*- coding: utf-8 -*-
from typing import NamedTuple, Tuple

import numpy as np
from scipy import signal

from signalworks import tracking, dsp
from processing import Processor


class EnergyEstimator(Processor):
    name = "RMS-Energy (dB)"
    acquire = NamedTuple("acquire", [("wave", tracking.Wave)])

    def __init__(self):
        super().__init__()
        self.parameters = {
            "frame_size": 0.020,  # in seconds
            "frame_rate": 0.010,
        }  # in seconds

    def process(self, progressTracker=None, **kwargs) -> Tuple[tracking.TimeValue]:
        # Processor.process(self, **kwargs)
        if progressTracker is not None:
            self.progressTracker = progressTracker
        wav = self.data.wave
        assert isinstance(wav, tracking.Wave)
        wav = wav.convert_dtype(np.float64)
        self.progressTracker.update(10)
        frame = dsp.frame(
            wav, self.parameters["frame_size"], self.parameters["frame_rate"]
        )
        self.progressTracker.update(70)
        frame.value *= signal.hann(frame.value.shape[1])
        value = 20 * np.log10(np.mean(frame.value ** 2.0, axis=1) ** 0.5)
        self.progressTracker.update(90)
        nrg = tracking.TimeValue(
            frame.time,
            value,
            wav.fs,
            wav.duration,
            path=wav.path.with_name(wav.path.stem + "-energy").with_suffix(
                tracking.TimeValue.default_suffix
            ),
        )
        nrg.min = value.min()
        nrg.max = value.max()
        nrg.unit = "dB"
        return (nrg,)
