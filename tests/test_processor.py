# -*- coding: utf-8 -*-
import numpy as np
import pytest  # noqa: F401
from signalworks.processors import (
    ConverterToFloat64,
    EnergyEstimator,
    F0Analyzer,
    Filter,
    NoiseReducer,
    PeakTracker,
    ZeroPhaseFilter,
)


def test_filter(speech_track):
    processor = Filter()
    processor.set_data({"wave": speech_track})
    results, = processor.process()
    assert speech_track.duration == results.duration
    assert isinstance(results.get_value(), np.ndarray)

    zp_processor = ZeroPhaseFilter()
    zp_processor.set_data({"wave": speech_track})
    results, = zp_processor.process()
    assert speech_track.duration == results.duration
    assert isinstance(results.get_value(), np.ndarray)


def test_f0(speech_track):
    processor = F0Analyzer()
    processor.set_data({"wave": speech_track})
    f0, dop, vox = processor.process()
    assert f0.duration == dop.duration


def test_convert(speech_track):
    processor = ConverterToFloat64()
    processor.set_data({"wave": speech_track})
    results, = processor.process()
    assert speech_track.duration == results.duration
    assert isinstance(results.get_value(), np.ndarray)


def test_energy(speech_track):
    processor = EnergyEstimator()
    processor.set_data({"wave": speech_track})
    results, = processor.process()
    assert len(results.get_time()) == len(results.get_value())


def test_peaktrack(speech_track):
    processor = PeakTracker()
    processor.set_data({"wave": speech_track})
    results, = processor.process()
    assert len(results.get_time()) == len(results.get_value())


def test_noisereducer(speech_track):
    processor = NoiseReducer()
    processor.set_data({"wave": speech_track})
    results, = processor.process()
    assert len(results.get_time()) == len(results.get_value())
