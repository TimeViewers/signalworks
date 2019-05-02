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
from signalworks.tracking import TimeValue, Track, Wave


def test_filter(speech_track, benchmark):
    processor = Filter()
    processor.set_data({"wave": speech_track})
    results, = benchmark(processor.process)
    assert speech_track.duration == results.duration
    assert isinstance(results.get_value(), np.ndarray)
    assert isinstance(results, Wave)
    assert isinstance(results, Track)


def test_zerofilter(speech_track, benchmark):
    zp_processor = ZeroPhaseFilter()
    zp_processor.set_data({"wave": speech_track})
    results, = benchmark(zp_processor.process)
    assert speech_track.duration == results.duration
    assert isinstance(results.get_value(), np.ndarray)
    assert isinstance(results, Wave)
    assert isinstance(results, Track)


def test_f0(speech_track, benchmark):
    processor = F0Analyzer()
    processor.set_data({"wave": speech_track})
    f0, dop, vox = benchmark(processor.process)
    assert f0.duration == dop.duration
    assert isinstance(f0, TimeValue)


def test_convert(speech_track, benchmark):
    processor = ConverterToFloat64()
    processor.set_data({"wave": speech_track})
    results, = benchmark(processor.process)
    assert speech_track.duration == results.duration
    assert isinstance(results.get_value(), np.ndarray)


def test_energy(speech_track, benchmark):
    processor = EnergyEstimator()
    processor.set_data({"wave": speech_track})
    results, = benchmark(processor.process)
    assert len(results.get_time()) == len(results.get_value())
    assert isinstance(results, TimeValue)


def test_peaktrack(speech_track, benchmark):
    processor = PeakTracker()
    processor.set_data({"wave": speech_track})
    results, = benchmark(processor.process)
    assert len(results.get_time()) == len(results.get_value())
    assert isinstance(results, TimeValue)


def test_noisereducer(speech_track, benchmark):
    processor = NoiseReducer()
    processor.set_data({"wave": speech_track})
    results, = benchmark(processor.process)
    assert len(results.get_time()) == len(results.get_value())
    assert isinstance(results, Wave)
