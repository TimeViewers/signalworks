# -*- coding: utf-8 -*-
"""
Digital Signal Processing
"""
import logging
from typing import Callable, Optional, Tuple

import numba
import numpy as np
from numpy.fft import fft, ifft, irfft, rfft
from scipy import signal, stats

from signalworks.tracking import TimeValue, Wave

logger = logging.getLogger(__name__)

# TODO: make this "tracking"-free (?), and all times are in samples

# def segment_talkbox(a: np.ndarray,
#                     length: int,
#                     overlap: int = 0) -> np.ndarray:
#     # originally from talkbox.segmentaxis
#     a = np.ravel(a)  # may copy
#     l = a.shape[0]
#     if overlap >= length:
#         raise ValueError("frames cannot overlap by more than 100%")
#     if overlap < 0 or length <= 0:
#         raise ValueError("overlap must be nonnegative and length must be positive")
#     if l < length or (l - length) % (length - overlap):
#         if l > length:
#             roundup = length + (1 + (l - length) //
#                                 (length - overlap)) * (length - overlap)
#             # TODO: further optimization possible
#             rounddown = length + ((l - length) // (length - overlap)) * (length - overlap)
#         else:
#             roundup = length
#             rounddown = 0
#         assert rounddown < l < roundup
#         assert roundup == rounddown + (length - overlap) or (roundup == length and rounddown == 0)
#         a = a.swapaxes(-1, 0)
#         a = a[..., :rounddown]
#         a = a.swapaxes(-1, 0)
#     l = a.shape[0]
#     if l == 0:
#         raise ValueError("Not enough data points to segment array")
#     assert l >= length
#     assert (l - length) % (length - overlap) == 0
#     n = 1 + (l - length) // (length - overlap)
#     s = a.strides[0]
#     newshape = a.shape[:0] + (n, length) + a.shape[1:]
#     newstrides = a.strides[:0] + ((length - overlap) * s, s) + a.strides[1:]
#     try:
#         return np.ndarray.__new__(np.ndarray, strides=newstrides,
#                                   shape=newshape, buffer=a, dtype=a.dtype)
#     except TypeError:
#         import warnings
#         warnings.warn("Problem with ndarray creation forces copy.")
#         a = a.copy()
#         # Shape doesn't change but strides does
#         newstrides = a.strides[:0] + ((length - overlap) * s, s) + a.strides[1:]
#         return np.ndarray.__new__(np.ndarray,
#                                   strides=newstrides,
#                                   shape=newshape,
#                                   buffer=a,
#                                   dtype=a.dtype)


# @numba.jit((numba.int16[:], numba.int64, numba.int32), nopython=True, cache=True)
@numba.jit(nopython=True, cache=True)  # we need polymorphism here
def segment(x, nsize, nrate):
    if len(x) < nsize:
        F = 0
    else:
        F = (len(x) - nsize) // nrate + 1  # the number of full frames
    assert F >= 0
    X = np.empty((F, nsize), dtype=x.dtype)
    a = 0
    for f in range(F):
        X[f, :] = x[a : a + nsize]
        a += nrate
    return X


def frame(wav: Wave, frame_size: float, frame_rate: float) -> TimeValue:
    """
    Given a waveform, return a timeValue track with each frame as the value and times of the center of each frame.
    times point to the center of the frame.
    Each frame will have the specified size, and t[i+1] = t[i] + rate.
    this will return as much of the signal as possible in full frames
    """
    # def unsigned int a, f, nrate, nsize
    assert wav.duration > 0
    nsize = int(round(frame_size * wav.fs))
    nrate = int(round(frame_rate * wav.fs))
    # import time
    # tic = time.time()
    # print("frame timing...")
    # if 0:  # TODO: unfortunately segment doesn't allow for negative overlap, i.e. jumps
    #     value = segment_talkbox(wav.value, nsize, nsize - nrate)  # because overlap = nsize - nrate
    value = segment(wav.value[:, -1], nsize, nrate)
    # print(f"frame took time: {time.time() - tic}")
    assert value.shape[1] == nsize
    time = np.array(np.arange(value.shape[0]) * nrate, dtype=np.int64) + nsize // 2
    return TimeValue(
        time, value, wav.fs, wav.duration, path=wav.path
    )  # adjust path name here?


# @numba.jit(nopython=True, cache=True)  # we need polymorphism here
def frame_centered(signal: np.ndarray, time: np.ndarray, frame_size: int) -> np.ndarray:
    assert time.ndim == 1
    # no further assumptions on time - doesn't have to be sorted or inside signal
    value = np.zeros((len(time), frame_size), dtype=signal.dtype)
    left_frame_size = frame_size // 2
    right_frame_size = frame_size - left_frame_size
    S = len(signal)
    for f, center in enumerate(time):
        left = center - left_frame_size
        right = center + right_frame_size
        if left >= 0 and right <= S:  # make the common case fast
            value[f, :] = signal[left:right]
        else:  # deal with edges on possibly both sides
            # left
            if left < 0:
                left_avail = left_frame_size + left
            else:
                left_avail = left_frame_size
            # right
            right_over = right - S
            if right_over > 0:
                right_avail = right_frame_size - right_over
            else:
                right_avail = right_frame_size
            if 0 <= center <= S:
                value[
                    f, left_frame_size - left_avail : left_frame_size + right_avail
                ] = signal[center - left_avail : center + right_avail]
    assert value.shape[0] == len(time)
    assert value.shape[1] == frame_size
    return value  # adjust path name here?


@numba.jit(nopython=True, cache=True)  # we need polymorphism here
def ola(
    frame: np.ndarray, fs: int, duration: int, frame_size: float, frame_rate: float
) -> np.ndarray:
    nsize = int(round(frame_size * fs))
    nrate = int(round(frame_rate * fs))
    y = np.zeros(duration, dtype=np.float64)
    a = 0
    for f in range(len(frame)):
        y[a : a + nsize] += frame[f]
        a += nrate
    return y


def spectral_subtract(inp: Wave, frame_rate: int, silence_percentage: int) -> Wave:
    assert 0 < silence_percentage < 100
    ftr = frame(inp, frame_rate * 2, frame_rate)
    x = ftr.value * signal.hann(ftr.value.shape[1])
    X = fft(x, 2 ** nextpow2(x.shape[1]))
    M = np.abs(X)
    E = np.mean(M ** 2, axis=1)
    threshold = stats.scoreatpercentile(E, silence_percentage)
    index = np.where(E < threshold)[0]
    noise_profile = np.median(M[index], axis=0)
    M -= noise_profile
    np.clip(
        M, 0, None, out=M
    )  # limit this to a value greater than 0 to avoid -inf due to the following log
    Y = M * np.exp(1j * np.angle(X))  # DEBUG
    y = ifft(Y).real
    s = ola(y, inp.fs, inp.duration, frame_rate * 2, frame_rate)
    return Wave(s, inp.fs)


def spectrogram(
    wav: Wave,
    frame_size: float,
    frame_rate: float,
    window: Callable[[int], np.ndarray] = signal.hann,
    NFFT: Optional[int] = None,
    normalized: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """return log-magnitude spectrogram in dB"""
    ftr = frame(wav, frame_size, frame_rate)
    x = ftr.value * window(ftr.value.shape[1])
    if NFFT is None:
        NFFT = 2 ** nextpow2(x.shape[1])
    M = np.abs(rfft(x, NFFT))
    np.clip(M, 1e-12, None, out=M)
    M = np.log10(M) * 20
    if normalized:
        M = (M.T - np.min(M, axis=1)).T
        M = (M.T / np.max(M, axis=1)).T
        assert np.all(M.min(axis=1) == 0)
        assert np.all(M.max(axis=1) == 1)
    frequency = np.arange(M.shape[1]) / M.shape[1] * wav.fs / 2
    return M, ftr.time, frequency


def spectrogram_centered(
    wav: Wave,
    frame_period: float,
    time: np.ndarray,
    window: Callable[[int], np.ndarray] = signal.hann,
    NFFT: Optional[int] = None,
    normalize_signal: bool = False,
    normalize_output: bool = False,
    pre_emphasis: Optional[float] = None,
    channel: int = 0,
    channel_aggregate: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """return log-magnitude spectrogram in dB

    Parameters
    ----------
    wav : Wave
        Wave object that contains the information you want to determine the
        log-magnitude spectrogram of
    frame_period : float
        The length of the frame in seconds
    time : np.ndarray
        A numpy array representing the time values (in samples, not seconds) of
        where you want frames centered about, typically evenly spaced over a range
    window : Callable[[int], np.ndarray], optional
        The Windowing function to apply to each frame
        Possible pre-made functions are shown in scipy.signal.windows
        by default signal.hann
    NFFT : Optional[int], optional
        Number of FFT points to use when for the rfft calculation
        If None, the NFFT is computed based on the number of samples on each frame
        It is suggested that this value be a power of 2
        by default None
    normalize_signal : bool, optional
        scales the signal so that it ranges between -1 and 1
        by default False
    normalize_output : bool, optional
        Adjusts the value of the spectrogram so that it's between 0 and 1
        by default False
    pre_emphasis : Optional[float], optional
        if a value is provided, it is a constant to apply for a pre-emphasis
        filter where x´[t] = x[t] - α * x[t-1] where α is the pre-emphasis value
        values 0.0 <= α <= 1.0,
        by default None
    channel : int
        Specifies which channel in the Wave object to use

        The value here is ignored if a channel_aggregate method is provided,
        or if Wave track has only one channel
        by default 0
    channel_aggregate : Optional[Callable[[np.ndarray], np.ndarray]], optional
        If the Wave track has multiple channels, use this method to aggregate
        the channels together.  A typical usecase would be to pass np.average;
        atlernate use-cases may be np.maximum or np.minimum. The callable method
        must support the passing of an axis argument.

        This method will supersede the channel argument, but it is ignored if
        Wave object has only one channel
        by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Spectrogram matrix and a corresponding array of frequency values
    """

    if normalize_signal:
        s = wav.value / np.abs(np.max(wav.value))
    else:
        s = wav.value.astype(np.float64)

    # When dealing with multi-channel audio
    if wav.value.shape[1] > 1:
        # if an aggregation method is provided, use that to squash data together
        if channel_aggregate is not None:
            s = channel_aggregate(s, axis=1)  # type: ignore

        # if no aggregation method is provided, and no channel is provided, grab the first channel
        else:
            s = s[:, channel]

    if pre_emphasis is not None:
        s -= pre_emphasis * np.roll(s, -1)

    ftr = frame_centered(s.flatten(), time, int(round(frame_period * wav.fs)))

    if NFFT is None:
        NFFT = 2 ** ftr.shape[1].bit_length()

    ftr *= window(ftr.shape[1])
    M = np.absolute(rfft(ftr, n=NFFT))
    np.clip(M, np.finfo(M.dtype).eps, None, out=M)

    M[:] = 1 + np.log10(M) * 20
    # faster than np.linspace
    frequency = np.arange(M.shape[1]) / M.shape[1] * wav.fs / 2

    if normalize_output:
        M[:] = (M.T - np.min(M, axis=1)).T
        M[:] = (M.T / np.max(M, axis=1)).T

    return M, frequency


def correlate_fft(X: np.ndarray) -> np.ndarray:
    """correlation for feature matrix"""
    assert X.ndim == 2
    D = X.shape[1]
    R = irfft(np.abs(rfft(X, 2 ** nextpow2(2 * D - 1))) ** 2)[:, :D]
    # show relationship to related methods
    assert np.allclose(R[0], np.correlate(X[0], X[0], mode="full")[D - 1 :])
    # assert np.allclose(r, np.convolve(x, x[::-1], mode='full')[n - 1:])
    # from scipy.signal import fftconvolve
    # assert np.allclose(r, fftconvolve(x, x[::-1], mode='full')[n - 1:])
    return R


def correlogram(
    wav: Wave, frame_size: float, frame_rate: float, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert wav.dtype == np.float64
    # t, x = frame(wav, frame_size, frame_rate)
    ftr = frame(wav, frame_size, frame_rate)
    M, D = ftr.value.shape
    R = correlate_fft(ftr.value)
    # if 1:
    #     # FFT order must be at least 2*len(x)-1 and should be a power of 2
    #     R = irfft(np.abs(rfft(ftr.value, 2 ** nextpow2(2 * D - 1))) ** 2)[:, D]
    # else:
    #     index = np.arange(int(np.round(D / 2)), D)
    #     R = np.empty((M, len(index)), dtype=np.float64)
    #     for m in range(M):
    #         signal = ftr.value[m]
    #         R[m, :] = np.correlate(signal, signal, mode='same')[index]  # TODO: use fft2 here instead
    if normalize:
        R[:, 1:] /= np.tile(
            R[:, 0], (R.shape[1] - 1, 1)
        ).T  # keep energy in zero-th coeff?
    frequency = np.r_[np.nan, wav.fs / np.arange(1, R.shape[1])]
    return R, ftr.time, frequency


def nextpow2(i: int) -> int:
    """returns the first P such that 2**P >= abs(N)"""
    return i.bit_length()


#
#
# def world(wave: Wave) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     x = wave.value / (2 ** 15)
#     fs = wave.fs
#     _f0, t = pw.dio(x, fs)  # raw pitch extractor
#     f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
#     sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
#     return sp, f0, t
