# -*- coding: utf-8 -*-
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
from signalworks import dsp, viterbi
from signalworks.processors.processing import (
    DefaultProgressTracker,
    InvalidParameterError,
    Processor,
)
from signalworks.tracking import Partition, TimeValue, Wave


class PeakTracker(Processor):
    name = "Peak Tracker"
    acquire = NamedTuple("acquire", [("wave", Wave)])

    def __init__(self):
        super().__init__()
        self.parameters = {
            "smooth": 1.0,
            "freq_min": 100,
            "freq_max": 1000,
            "frame_size": 0.02,  # seconds, determines freq res.
            "frame_rate": 0.01,
            "NFFT": 512,
        }

    def get_parameters(self):
        if "wave" in self.acquire._field_types.keys():
            if self.data.wave is not None:
                self.parameters["freq_max"] = self.data.wave.fs / 2
        return super().get_parameters()

    def set_parameters(self, parameter: Dict[str, str]) -> None:
        super().set_parameters(parameter)
        if not self.parameters["freq_min"] < self.parameters["freq_max"]:
            raise InvalidParameterError("freq_min must be < freq_max")

    def process(
        self, progressTracker: Optional[DefaultProgressTracker] = None
    ) -> Tuple[TimeValue]:
        if progressTracker is not None:
            self.progressTracker = progressTracker
        wav = self.data.wave
        self.progressTracker.update(10)
        assert isinstance(wav, Wave)
        ftr, time, frequency = dsp.spectrogram(
            wav,
            self.parameters["frame_size"],
            self.parameters["frame_rate"],
            NFFT=self.parameters["NFFT"],
        )
        self.progressTracker.update(50)
        a = frequency.searchsorted(self.parameters["freq_min"])
        b = frequency.searchsorted(self.parameters["freq_max"])
        # import time as timer
        # print('searching')
        # tic = timer.time()
        seq, _ = viterbi.search_smooth(ftr[:, a:b], self.parameters["smooth"])
        self.progressTracker.update(90)
        # toc = timer.time()
        # print(f'done, took: {toc-tic}')
        trk = TimeValue(
            time,
            frequency[a + seq],
            wav.fs,
            wav.duration,
            wav.path.with_name(wav.path.stem + "-peak").with_suffix(
                TimeValue.default_suffix
            ),
        )
        trk.min = 0
        trk.max = wav.fs / 2
        trk.unit = "Hz"
        trk.label = "frequency"
        return (trk,)


class PeakTrackerActiveOnly(PeakTracker):
    name = "Peak Tracker (active regions only)"
    acquire = NamedTuple("acquire", [("wave", Wave), ("active", Partition)])

    def process(
        self, progressTracker: Optional[DefaultProgressTracker] = None
    ) -> Tuple[TimeValue]:
        if progressTracker is not None:
            self.progressTracker = progressTracker
        peak = super().process()[0]
        active = self.data.active
        assert isinstance(active, Partition)
        for i in range(len(active.time) - 1):
            if active.value[i] in ["0", 0]:
                a = np.searchsorted(peak.time / peak.fs, active.time[i] / active.fs)
                b = np.searchsorted(peak.time / peak.fs, active.time[i + 1] / active.fs)
                peak.value[a:b] = np.nan
        return (peak,)
