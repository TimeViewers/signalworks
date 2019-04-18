# -*- coding: utf-8 -*-
from typing import NamedTuple, Optional, Tuple

import numpy as np
from signalworks import dsp, viterbi
from signalworks.processors.processing import DefaultProgressTracker, Processor
from signalworks.tracking import Partition, TimeValue, Wave


class ActivityDetector(Processor):
    name = "Activity Detector"
    acquire = NamedTuple("acquire", [("wave", Wave)])

    def __init__(self):
        super().__init__()
        self.parameters = {
            "threshold": -30.0,
            "smooth": 1.0,
            "frame_size": 0.020,  # in seconds
            "frame_rate": 0.01,
        }

    def process(
        self, progressTracker: Optional[DefaultProgressTracker] = None
    ) -> Tuple[Partition, TimeValue]:
        if progressTracker is not None:
            self.progressTracker = progressTracker
        wav = self.data.wave
        assert isinstance(wav, Wave)
        wav = wav.convert_dtype(np.float64)
        self.progressTracker.update(10)
        assert isinstance(wav, Wave)
        M, time, frequency = dsp.spectrogram(
            wav, self.parameters["frame_size"], self.parameters["frame_rate"]
        )
        self.progressTracker.update(20)
        # Emax = np.atleast_2d(np.max(M, axis=1)).T
        Emax = 20 * np.log10(np.mean((10 ** (M / 10)), axis=1) ** 0.5)
        P = np.empty((len(Emax), 2))
        P[:, 0] = 1 / (1 + np.exp(Emax - self.parameters["threshold"]))
        P[:, 1] = 1 - P[:, 0]  # complement
        self.progressTracker.update(30)
        seq, _ = viterbi.search_smooth(P, self.parameters["smooth"])
        self.progressTracker.update(90)
        tmv = TimeValue(
            time,
            seq,
            wav.fs,
            wav.duration,
            wav.path.with_name(wav.path.stem + "-act").with_suffix(
                TimeValue.default_suffix
            ),
        )
        par = Partition.from_TimeValue(tmv)
        par.value = np.char.mod("%d", par.value)
        emax = TimeValue(
            time,
            Emax,
            wav.fs,
            wav.duration,
            wav.path.with_name(wav.path.stem + "-emax").with_suffix(
                TimeValue.default_suffix
            ),
        )
        emax.min = Emax.min()
        emax.max = Emax.max()
        emax.unit = "dB"
        emax.label = "maximum frequency magnitude"
        return par, emax
