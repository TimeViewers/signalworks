# -*- coding: utf-8 -*-
from typing import NamedTuple, Optional, Tuple

import numpy as np
from signalworks import dsp
from signalworks.processors.processing import DefaultProgressTracker, Processor
from signalworks.tracking import Wave


class NoiseReducer(Processor):
    name = "Noise Reducer"
    acquire = NamedTuple("acquire", [("wave", Wave)])

    def __init__(self):
        super().__init__()
        self.parameters = {"silence_percentage": 10, "frame_rate": 0.01}  # in seconds

    def process(
        self, progressTracker: Optional[DefaultProgressTracker] = None
    ) -> Tuple[Wave]:
        # Processor.process(self, **kwargs)
        if progressTracker is not None:
            self.progressTracker = progressTracker
        inp = self.data.wave
        assert isinstance(inp, Wave)
        inp = inp.convert_dtype(np.float64)
        self.progressTracker.update(20)
        # TODO: pull this up into here
        assert isinstance(inp, Wave)
        out = dsp.spectral_subtract(
            inp, self.parameters["frame_rate"], self.parameters["silence_percentage"]
        )
        self.progressTracker.update(90)
        assert isinstance(inp, Wave)
        out.path = inp.path.with_name(inp.path.stem + "-denoised").with_suffix(
            Wave.default_suffix
        )
        return (out,)
