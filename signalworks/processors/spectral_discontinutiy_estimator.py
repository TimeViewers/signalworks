# -*- coding: utf-8 -*-
from typing import NamedTuple, Optional, Tuple

import numpy as np
from signalworks import dsp
from signalworks.processors.processing import DefaultProgressTracker, Processor
from signalworks.tracking import TimeValue, Wave


class SpectralDiscontinuityEstimator(Processor):
    name = "Spectral Discontinuity Estimator"
    acquire = NamedTuple("acquire", [("wave", Wave)])

    def __init__(self):
        super().__init__()
        self.parameters = {
            "frame_size": 0.005,  # seconds, determines freq res.
            "NFFT": 256,
            "normalized": 1,
            "delta_order": 1,
        }

    def process(
        self, progressTracker: Optional[DefaultProgressTracker] = None
    ) -> Tuple[TimeValue]:
        if progressTracker is not None:
            self.progressTracker = progressTracker
        wav = self.data.wave
        assert isinstance(wav, Wave)
        self.progressTracker.update(10)
        ftr, time, frequency = dsp.spectrogram(
            wav,
            self.parameters["frame_size"],
            self.parameters["frame_size"],  # frame_rate = frame_size
            NFFT=self.parameters["NFFT"],
            normalized=self.parameters["normalized"],
        )
        if self.parameters["normalized"]:
            ftr = ftr - np.mean(ftr, axis=1).reshape(-1, 1)

        time = (time[:-1] + time[1:]) // 2
        assert self.parameters["delta_order"] > 0
        dynamic_win = np.arange(
            -self.parameters["delta_order"], self.parameters["delta_order"] + 1
        )

        win_width = self.parameters["delta_order"]
        win_length = 2 * win_width + 1
        den = 0
        for s in range(1, win_width + 1):
            den += s ** 2
        den *= 2
        dynamic_win = dynamic_win / den

        N, D = ftr.shape
        print(N)
        temp_array = np.zeros((N + 2 * win_width, D))
        delta_array = np.zeros((N, D))
        self.progressTracker.update(90)
        temp_array[win_width : N + win_width] = ftr
        for w in range(win_width):
            temp_array[w, :] = ftr[0, :]
            temp_array[N + win_width + w, :] = ftr[-1, :]

        for i in range(N):
            for w in range(win_length):
                delta_array[i, :] += temp_array[i + w, :] * dynamic_win[w]
        value = np.mean(np.diff(delta_array, axis=0) ** 2, axis=1) ** 0.5
        dis = TimeValue(
            time,
            value,
            wav.fs,
            wav.duration,
            path=wav.path.with_name(wav.path.stem + "-discont").with_suffix(
                TimeValue.default_suffix
            ),
        )
        dis.min = 0
        dis.max = value.max()
        dis.unit = "dB"
        dis.label = "spectral discontinuity"
        self.progressTracker.update(100)
        return (dis,)
