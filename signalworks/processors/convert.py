# -*- coding: utf-8 -*-
from typing import NamedTuple, Tuple

import numpy as np

from processing import Processor
from signalworks import tracking


class ConverterToFloat64(Processor):
    name = "Conversion to Float64"
    acquire = NamedTuple("acquire", [("wave", tracking.Wave)])

    def process(self, **kwargs) -> Tuple[tracking.Wave]:
        Processor.process(self, **kwargs)
        wav = self.data.wave
        wav = wav.convert_dtype(np.float64)
        wav.path = wav.path.with_name(wav.path.stem + "-float64").with_suffix(
            tracking.Wave.default_suffix
        )
        return (wav,)


class ConverterToInt16(Processor):
    name = "Conversion to Int16"
    acquire = NamedTuple("acquire", [("wave", tracking.Wave)])

    def process(self, **kwargs) -> Tuple[tracking.Wave]:
        Processor.process(self, **kwargs)
        wav = self.data.wave
        wav = wav.convert_dtype(np.int16)
        wav.path = wav.path.with_name(wav.path.stem + "-int16").with_suffix(
            tracking.Wave.default_suffix
        )
        return (wav,)
