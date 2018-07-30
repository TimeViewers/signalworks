# -*- coding: utf-8 -*-
from typing import NamedTuple, Tuple

import numpy as np

from processing import Processor
from signalworks import tracking, dsp


class NoiseReducer(Processor):
    name = "Noise Reducer"
    acquire = NamedTuple("acquire", [("wave", tracking.Wave)])

    def __init__(self):
        super().__init__()
        self.parameters = {"silence_percentage": 10, "frame_rate": 0.01}  # in seconds

    def process(self, progressTracker=None, **kwargs) -> Tuple[tracking.Wave]:
        # Processor.process(self, **kwargs)
        if progressTracker is not None:
            self.progressTracker = progressTracker
        inp = self.data.wave
        inp = inp.convert_dtype(np.float64)
        self.progressTracker.update(20)
        # TODO: pull this up into here
        out = dsp.spectral_subtract(
            inp, self.parameters["frame_rate"], self.parameters["silence_percentage"]
        )
        self.progressTracker.update(90)
        out.path = inp.path.with_name(inp.path.stem + "-denoised").with_suffix(
            tracking.Wave.default_suffix
        )
        return (out,)
