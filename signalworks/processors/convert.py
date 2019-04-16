# -*- coding: utf-8 -*-
from typing import NamedTuple, Optional, Tuple

import numpy as np
from signalworks.processors.processing import DefaultProgressTracker, Processor
from signalworks.tracking import Wave

# @abstractmethod
# def process(self, progressTracker: Optional[DefaultProgressTracker] = None) -> Tuple[Tracks, ...]:


class ConverterToFloat64(Processor):
    name = "Conversion to Float64"
    acquire = NamedTuple("acquire", [("wave", Wave)])

    def process(
        self, progressTracker: Optional[DefaultProgressTracker] = None
    ) -> Tuple[Wave]:
        _ = Processor.process(self)
        wav = self.data.wave
        assert isinstance(wav, Wave)
        wav = wav.convert_dtype(np.float64)
        assert isinstance(wav, Wave)
        wav.path = wav.path.with_name(wav.path.stem + "-float64").with_suffix(
            Wave.default_suffix
        )
        return (wav,)


class ConverterToInt16(Processor):
    name = "Conversion to Int16"
    acquire = NamedTuple("acquire", [("wave", Wave)])

    def process(
        self, progressTracker: Optional[DefaultProgressTracker] = None
    ) -> Tuple[Wave]:  # , **kwargs
        Processor.process(self)
        wav = self.data.wave
        assert isinstance(wav, Wave)
        wav = wav.convert_dtype(np.int16)
        assert isinstance(wav, Wave)
        wav.path = wav.path.with_name(wav.path.stem + "-int16").with_suffix(
            Wave.default_suffix
        )
        return (wav,)
