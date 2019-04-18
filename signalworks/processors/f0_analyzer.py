# -*- coding: utf-8 -*-
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
from scipy import signal
from signalworks import dsp, viterbi
from signalworks.processors.processing import DefaultProgressTracker, Processor
from signalworks.tracking import Partition, TimeValue, Wave


class F0Analyzer(Processor):
    name = "F0 Analysis"
    acquire = NamedTuple("acquire", [("wave", Wave)])

    def __init__(self):
        super().__init__()
        self.t0_min = 0
        self.t0_max = 0
        self.parameters = {
            "smooth": 0.01,
            "f0_min": 51,  # in Hertz
            "f0_max": 300,  # in Hertz
            "frame_size": 0.040,  # in seconds
            "frame_rate": 0.010,  # in seconds
            "dop threshold": 0.7,
            "energy threshold": 0.1,
        }

    def set_parameters(self, parameter: Dict[str, str]) -> None:
        super().set_parameters(parameter)
        assert (
            self.parameters["f0_min"] < self.parameters["f0_max"]
        ), "f0_min must be < f0_max"
        assert self.parameters["frame_size"] > (
            2 / self.parameters["f0_min"]
        ), "frame_size must be > 2 / f0_min"

    def process(
        self, progressTracker: Optional[DefaultProgressTracker] = None
    ) -> Tuple[TimeValue, TimeValue, Partition]:
        if progressTracker is not None:
            self.progressTracker = progressTracker
        wav = self.data.wave
        assert isinstance(wav, Wave)
        wav = wav.convert_dtype(np.float64)
        self.progressTracker.update(10)
        assert isinstance(wav, Wave)
        R, time, frequency = dsp.correlogram(
            wav, self.parameters["frame_size"], self.parameters["frame_rate"]
        )

        self.progressTracker.update(30)
        assert isinstance(wav, Wave)
        t0_min = int(round(wav.fs / self.parameters["f0_max"]))
        t0_max = int(round(wav.fs / self.parameters["f0_min"]))
        index = np.arange(t0_min, t0_max + 1, dtype=np.int)
        E = R[:, 0]  # energy
        R = R[:, index]  # only look at valid candidates
        # normalize
        R -= R.min()
        R /= R.max()
        # find best sequence
        seq, _ = viterbi.search_smooth(R, self.parameters["smooth"])
        self.progressTracker.update(80)

        f0 = wav.fs / (t0_min + seq)
        # degree of periodicity
        dop = R[np.arange(R.shape[0]), seq]
        # voicing
        v = (
            (dop > self.parameters["dop threshold"])
            & (E > self.parameters["energy threshold"])
            #  (seq > 0) & (seq < len(index) - 1)
        ).astype(np.int)
        v = signal.medfilt(v, 5)  # TODO: replace by a 2-state HMM
        f0[v == 0] = np.nan
        # prepare tracks
        f0 = TimeValue(
            time,
            f0,
            wav.fs,
            wav.duration,
            wav.path.with_name(wav.path.stem + "-f0").with_suffix(
                TimeValue.default_suffix
            ),
        )
        f0.min = self.parameters["f0_min"]
        f0.max = self.parameters["f0_max"]
        f0.unit = "Hz"
        f0.label = "F0"
        dop = TimeValue(
            time,
            dop,
            wav.fs,
            wav.duration,
            wav.path.with_name(wav.path.stem + "-dop").with_suffix(
                TimeValue.default_suffix
            ),
        )
        dop.min = 0
        dop.max = 1
        dop.label = "degree of periodicity"
        vox = TimeValue(
            time,
            v,
            wav.fs,
            wav.duration,
            wav.path.with_name(wav.path.stem + "-vox").with_suffix(
                TimeValue.default_suffix
            ),
        )
        vox = Partition.from_TimeValue(vox)
        vox.label = "voicing"
        assert isinstance(f0, TimeValue)
        assert isinstance(dop, TimeValue)
        assert isinstance(vox, Partition)
        return f0, dop, vox
