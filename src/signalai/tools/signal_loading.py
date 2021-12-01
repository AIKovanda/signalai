from pathlib import Path
import numpy as np
import pydub
from pydub import AudioSegment
from signalai.config import DTYPE_BYTES


def pydub2numpy(audio: pydub.AudioSegment) -> (np.ndarray, int):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate


def audio_file2numpy(file):
    file = Path(file)
    suffix = file.suffix
    file = str(file.absolute())
    if suffix == '.wav':
        audio = AudioSegment.from_wav(file)
    elif suffix == '.mp3':
        audio = AudioSegment.from_mp3(file)
    elif suffix == '.aac':
        audio = AudioSegment.from_file(file, "aac")
    else:
        raise TypeError(f"Suffix '{suffix}' is not supported yet!")

    return pydub2numpy(audio)[0].T


def from_audio(filename, file_sample_interval=None, interval=None) -> np.ndarray:
    real_start, interval_length = get_interval_values(file_sample_interval, interval)
    if not real_start:
        return audio_file2numpy(filename)
    if not file_sample_interval:
        return audio_file2numpy(filename)[:, real_start:]
    return audio_file2numpy(filename)[:, real_start: real_start + interval_length]


def from_bin(filename, file_sample_interval=None, interval=None, dtype='float32') -> np.ndarray:
    real_start, interval_length = get_interval_values(file_sample_interval, interval)
    with open(filename, "rb") as f:
        start_byte = int(DTYPE_BYTES[dtype] * real_start)
        assert start_byte % DTYPE_BYTES[dtype] == 0, "Bytes are not loading properly."
        f.seek(start_byte, 0)
        return np.expand_dims(np.fromfile(f, dtype=dtype, count=interval_length or -1), axis=0)


def from_npy(filename, file_sample_interval=None, interval=None) -> np.ndarray:
    real_start, interval_length = get_interval_values(file_sample_interval, interval)
    signal = np.load(filename)
    if len(signal.shape) == 1:
        signal = np.expand_dims(signal, axis=0)
    return signal[:, real_start: real_start + interval_length]


def get_interval_values(file_sample_interval, interval):
    real_start = 0
    interval_length = None
    if file_sample_interval is not None:
        real_start += file_sample_interval[0]
        interval_length = file_sample_interval[1] - file_sample_interval[0]
    if interval is not None:
        real_start += interval[0]
        if interval_length is not None:
            interval_length = min(interval_length, interval[1] - interval[0])
        else:
            interval_length = interval[1] - interval[0]

    if interval_length is not None:
        interval_length = int(interval_length)
    return int(real_start), interval_length
