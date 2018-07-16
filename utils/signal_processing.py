import decimal
import math

import numpy as np
import pywt
from scipy.io import wavfile


def units_to_sample(x, unit, sr):
    unit = unit.lower()

    if unit == "ms":
        return ms_to_sample(x, sr)
    if unit == "s":
        in_ms = float(x) * 1000
        return ms_to_sample(in_ms, sr)
    if unit == "sample":
        return int(x)

    raise Exception("Unknown units %s. Expected one of [ms, s, sample]" % unit)



def ms_to_sample(ms, sr):
    return int((float(ms) / 1000) * sr)

def sample_to_ms(sample, sr):
    return (sample / float(sr)) * 1000


def read_wave(path, first_channel=False):
    sample_rate, data = wavfile.read(path)

    if first_channel and len(data.shape) > 1:
        data = data[:,0]

    return sample_rate, data

def write_wav(out_file_path, sample_rate, wav_data):
    wavfile.write(out_file_path, sample_rate, wav_data)


# def _read_wave(path, expected_sample_rate=None):
#     if not os.path.isfile(path):
#         raise ValueError("File does not exist: %s" % path)
#
#     with contextlib.closing(wave.open(path, 'rb')) as wf:
#         # num_channels = wf.getnchannels()
#         #
#         # if num_channels != 1:
#         #     raise ValueError("Wrong number of channels (%i) for %s" % (num_channels, path))
#
#         sample_width = wf.getsampwidth()
#
#         if sample_width != 2:
#             raise ValueError("Wrong sample width (%i) for %s" % (sample_width, path))
#
#         sample_rate = wf.getframerate()
#
#         if sample_rate not in (8000, 16000, 32000, 44100):
#             raise ValueError("Unsupported sample rate (%i) for %s" % (sample_rate, path))
#
#         if expected_sample_rate is not None:
#             if sample_rate != expected_sample_rate:
#                 raise ValueError("Sample rate (%i) for %s does not match expected rate of %i" % (
#                     sample_rate, path, expected_sample_rate))
#
#         pcm_data = wf.readframes(wf.getnframes())
#         return pcm_data, sample_rate


# https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/sigproc.py
def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

# https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/sigproc.py
def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    """Frame a signal into overlapping frames.
    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))

    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = padsignal[indices]
    win = np.tile(winfunc(frame_len), (numframes, 1))
    return frames * win


# https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/sigproc.py
def deframesig(frames, siglen, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    """Does overlap-add procedure to undo the action of framesig.
    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = np.shape(frames)[0]
    assert np.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'

    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0: siglen = padlen

    rec_signal = np.zeros((padlen,))
    window_correction = np.zeros((padlen,))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i:]] = window_correction[indices[i:]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i:]] = rec_signal[indices[i:]] + frames[i:]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def wavelet_analysis(data, wavelet="db1"):
    c_approx, c_detail = pywt.dwt(data, wavelet)
    return np.asarray([c_approx, c_detail])

def wavelet_analysis_framed(data, sr, frame_ms, step_ms, wavelet="db1"):
    frame_len_samples = ms_to_sample(frame_ms, sr)
    frame_step_samples = ms_to_sample(step_ms, sr)

    signal = framesig(data, frame_len_samples, frame_step_samples)

    res = []
    for s in signal:
        c_approx, c_detail = pywt.dwt(s, wavelet)
        res.append([c_approx, c_detail])

    return np.asarray(res)