import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from signalai.tools.utils import join_dicts


DTYPE_BYTES = {'float32': 4, 'float16': 2}


class TimeSeries:
    """
    Stores either 1D signal or a 2D time-frequency transformation of a signal.
    First axis represents channel axis, the last one time axis.
    Operators:
    a + b - summing a and b
    a | b - joining channels of a and b
    a & b = concatenation of a and b
    """
    full_dimensions = None

    def __init__(self, data_arr: np.ndarray = None, meta=None, time_map=None, fs: float = None):
        if meta is None:
            meta = {}

        if data_arr is not None and not isinstance(data_arr, np.ndarray):
            raise TypeError(f"Unknown signal type {type(data_arr)}.")

        if len(data_arr.shape) == (self.full_dimensions or 2) - 1:  # channel axis missing
            data_arr = np.expand_dims(data_arr, axis=0)

        if self.full_dimensions is not None:
            if len(data_arr.shape) != self.full_dimensions:
                raise ValueError(f"Type {type(self)} must have {self.full_dimensions} dimensions, not "
                                 f"{len(data_arr.shape)}.")

        self.data_arr = data_arr

        if time_map is None:
            if self.data_arr is None:
                raise ValueError("Time map must be set when data_arr is None.")
            self.time_map = np.ones((self.data_arr.shape[0], self.data_arr.shape[-1]), dtype=bool)
        else:
            self.time_map = time_map.astype(bool)

        if len(self.time_map.shape) == 1:
            self.time_map = np.expand_dims(self.time_map, axis=0)

        if len(self.time_map.shape) > 2:
            raise ValueError(f"Data map must have one or two axes, not {len(self.time_map.shape)}.")

        self.meta = meta.copy()
        self.fs = fs

    def crop(self, interval: tuple[int, int] = None):
        if interval is None:
            data_arr = self.data_arr
            time_map = self.time_map
        else:
            data_arr = self.data_arr[..., interval[0]:interval[1]]
            time_map = self.time_map[..., interval[0]:interval[1]]

        return type(self)(
            data_arr=data_arr,
            time_map=time_map,
            meta=self.meta,
            fs=self.fs,
        )

    def __len__(self):
        if self.data_arr is not None:
            return self.data_arr.shape[-1]

        return self.time_map.shape[-1]

    def astype_(self, dtype):
        self.data_arr = self.data_arr.astype(dtype)

    @property
    def channels_count(self):
        if self.data_arr is not None:
            return self.data_arr.shape[0]
        return self.time_map.shape[0]

    def take_channels(self, channels: list[list[int] | int] = None):
        if channels is None:
            return self

        data_arrays = []
        time_maps = []
        for channel_gen in channels:
            if isinstance(channel_gen, int):
                data_arrays.append(self.data_arr[[channel_gen], ...])
                time_maps.append(self.time_map[[channel_gen], ...])
            elif isinstance(channel_gen, list):
                data_arrays.append(np.sum(self.data_arr[channel_gen, ...], axis=0))
                time_maps.append(np.any(self.time_map[channel_gen, ...], axis=0))
            else:
                raise TypeError(f"Channel cannot be generated using type '{type(channel_gen)}'.")

        return type(self)(
            data_arr=np.vstack(data_arrays),
            time_map=np.vstack(time_maps),
            meta=self.meta,
            fs=self.fs,
        )

    def trim(self, threshold=1e-5):
        first = np.min(np.argmax(np.abs(self.data_arr) > threshold, axis=1))
        last = len(self) - np.min(np.argmax(np.abs(self.data_arr[:, ::-1]) > threshold, axis=1))
        new_data_arr = self.data_arr[:, first: last]
        new_time_map = self.time_map[:, first: last]
        return type(self)(
            data_arr=new_data_arr.copy(),
            time_map=new_time_map.copy(),
            meta=self.meta.copy(),
            fs=self.fs,
        )

    def trim_(self, threshold=1e-5):
        first = np.min(np.argmax(np.abs(self.data_arr) > threshold, axis=1))
        last = len(self) - np.min(np.argmax(np.abs(self.data_arr[:, ::-1]) > threshold, axis=1))
        self.data_arr = self.data_arr[:, first: last]
        self.time_map = self.time_map[:, first: last]

    def margin_interval(self, interval_length: int = None, start_id=0):
        if interval_length is None:
            interval_length = len(self)

        if interval_length == len(self) and (start_id == 0 or start_id is None):
            return self

        new_data_arr = np.zeros((*self.data_arr.shape[:-1], interval_length), dtype=self.data_arr.dtype)
        new_time_map = np.zeros((self.data_arr.shape[0], interval_length), dtype=bool)

        if 0 < len(self) + start_id and start_id < interval_length:
            new_data_arr[..., max(0, start_id):min(interval_length, start_id + len(self))] = \
                self.data_arr[..., max(0, -start_id):min(len(self), interval_length - start_id)]

            new_time_map[..., max(0, start_id):min(interval_length, start_id + len(self))] = \
                self.time_map[..., max(0, -start_id):min(len(self), interval_length - start_id)]

        return type(self)(
            data_arr=new_data_arr,
            meta=self.meta,
            time_map=new_time_map,
            fs=self.fs,
        )

    def apply(self, func):
        return type(self)(
            data_arr=func(self.data_arr),
            meta=self.meta,
            time_map=self.time_map,
            fs=self.fs,
        )

    def __add__(self, other):
        if isinstance(other, type(self)):
            if len(self) != len(other):
                raise ValueError(f"Adding signals with different lengths is forbidden (for a good reason). "
                                 f"{len(self)}, {len(other)}")

            if (not (self.fs is None and other.fs is None)) and self.fs != other.fs:
                raise ValueError("Adding signals with different fs is forbidden (for a good reason).")

            new_data_arr = self.data_arr + other.data_arr
            new_info = join_dicts(self.meta, other.meta)
            new_time_map = self.time_map | other.time_map

        else:
            new_data_arr = self.data_arr + other
            new_info = self.meta
            new_time_map = self.time_map.copy()

        return type(self)(
            data_arr=new_data_arr,
            meta=new_info,
            time_map=new_time_map,
            fs=self.fs,
        )

    def __and__(self, other):
        if isinstance(other, type(self)):
            other_ts = other
            new_info = join_dicts(self.meta, other.meta)
        else:
            other_ts = type(self)(other)
            new_info = self.meta

        if (not (self.fs is None and other.fs is None)) and self.fs != other.fs:
            raise ValueError("Joining signals with different fs is forbidden (for a good reason).")

        new_data_arr = np.concatenate([self.data_arr, other_ts.data_arr], axis=-1)
        new_time_map = np.concatenate([self.time_map, other_ts.time_map], axis=-1)

        return type(self)(
            data_arr=new_data_arr,
            meta=new_info,
            time_map=new_time_map,
            fs=self.fs,
        )

    def __or__(self, other):
        if isinstance(other, type(self)):
            other_ts = other
            new_info = join_dicts(self.meta, other.meta)
        else:
            other_ts = type(self)(other)
            new_info = self.meta

        if len(self) != len(other):
            raise ValueError(f"Joining signals with different lengths is forbidden (for a good reason). "
                             f"{len(self)}, {len(other)}")

        if (not (self.fs is None and other.fs is None)) and self.fs != other.fs:
            raise ValueError("Joining signals with different fs is forbidden (for a good reason).")

        new_data_arr = np.concatenate([self.data_arr, other_ts.data_arr], axis=0)
        new_time_map = np.concatenate([self.time_map, other_ts.time_map], axis=0)

        return type(self)(
            data_arr=new_data_arr,
            meta=new_info,
            time_map=new_time_map,
            fs=self.fs,
        )

    def __mul__(self, other):
        return type(self)(
            data_arr=self.data_arr * other,
            meta=self.meta,
            time_map=self.time_map,
            fs=self.fs,
        )

    def __truediv__(self, other):
        return type(self)(
            data_arr=self.data_arr / other,
            meta=self.meta,
            time_map=self.time_map,
            fs=self.fs,
        )

    def __eq__(self, other):
        return (np.all(self.data_arr == other.data_arr) and
                np.all(self.data_arr.shape == other.data_arr.shape) and
                self.meta == other.meta and
                np.all(self.time_map == other.time_map) and
                ((self.fs is None and other.fs is None) or self.fs == other.fs))

    def __repr__(self):
        return str(pd.DataFrame.from_dict(
            self.meta | {'length': len(self), 'channels': self.channels_count, 'fs': self.fs},
            orient='index', columns=['value'],
        ))

    # def __iter__(self):
    #     self.n = 0
    #     return self
    #
    # def __next__(self):
    #     if self.n < self.channels_count:
    #         sub_arr = self._data_arr[self.n: self.n+1]
    #         self.n += 1
    #         return type(self)(
    #             data_arr=sub_arr,
    #             meta=self.meta,
    #             time_map=self.time_map,
    #             fs=self.fs,
    #         )
    #     else:
    #         raise StopIteration

    def update_meta(self, dict_):
        self.meta = self.meta.copy()
        self.meta.update(dict_)


class Signal(TimeSeries):
    full_dimensions = 2

    def show(self, channels=0, figsize=(16, 3)):
        plt.figure(figsize=figsize)
        plt.plot(self.data_arr[channels])
        plt.show()

    def spectrogram(self, figsize=(16, 9), save_as=None, show=True):
        from scipy import signal as scipy_signal
        plt.figure(figsize=figsize)
        f, t, sxx = scipy_signal.spectrogram(self.data_arr[0], self.fs)
        sxx = np.sqrt(sxx)
        plt.pcolormesh(t, f, sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()

    def spectrogram3(self, figsize=(16, 9), save_as=None, show=True):
        import librosa
        plt.figure(figsize=figsize)
        sxx = np.abs(librosa.stft(self.data_arr[0, ::2], center=False, n_fft=2048, hop_length=1024))
        sxx = sxx / np.max(sxx)
        print(sxx.shape, np.max(sxx), np.min(sxx))
        plt.pcolormesh(sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()

    def spectrogram2(self):
        import librosa
        from librosa.display import specshow
        s = np.abs(librosa.stft(self.data_arr[0]))
        fig, ax = plt.subplots()

        img = specshow(librosa.amplitude_to_db(s, ref=np.max),
                       y_axis='log', x_axis='time', sr=self.fs, ax=ax)
        ax.set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    def play(self, channel_id=0, fs: float = None, volume=32):
        import sounddevice as sd
        fs = fs or self.fs
        sd.play(volume * self.data_arr[channel_id].astype('float32'), fs)
        sd.wait()

    def to_mp3(self, file, normalized=True, fs=None):
        import pydub
        """
        set fs to 44100 if you want
        """
        if self.data_arr.ndim not in [1, 2]:
            raise ValueError(f"This signal has '{self.data_arr.ndim}' channels. Allowed values are 1 or 2.")
        if normalized:  # normalized array - each item should be a float in [-1, 1)
            y = np.int16(self.data_arr.T * 2 ** 15)
        else:
            y = np.int16(self.data_arr.T)
        if fs in None:
            fs = self.fs

        song = pydub.AudioSegment(y.tobytes(), frame_rate=fs, sample_width=2, channels=self.data_arr.ndim)
        song.export(file, format="mp3", bitrate="320k")


class Signal2D(TimeSeries):
    full_dimensions = 3

    def show(self):
        ...


def sum_time_series(series: list[TimeSeries]) -> TimeSeries:
    if len(series) == 0:
        raise ValueError
    elif len(series) == 1:
        return series[0]

    ts0 = series[0]
    for ts in series[1:]:
        ts0 = ts0 + ts

    return ts0


def stack_time_series(series: list[TimeSeries]) -> TimeSeries:
    if len(series) == 0:
        raise ValueError
    elif len(series) == 1:
        return series[0]

    ts0 = series[0]
    for ts in series[1:]:
        ts0 = ts0 | ts

    return ts0


def join_time_series(series: list[TimeSeries]) -> TimeSeries:
    if len(series) == 0:
        raise ValueError
    elif len(series) == 1:
        return series[0]

    ts0 = series[0]
    for ts in series[1:]:
        ts0 = ts0 & ts

    return ts0


def audio_file2numpy(file) -> tuple[np.ndarray, int]:
    from pydub import AudioSegment
    from pydub.utils import mediainfo
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

    np_audio = np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate
    return np_audio[0].T, int(mediainfo(file)['sample_rate'])


def read_audio(filename, interval: tuple[int, int] = None, dtype=None, fs: float = None, meta=None) -> Signal:
    
    data_arr, fs_ = audio_file2numpy(filename)
    if fs is not None and fs != fs_:
        raise ValueError(f'fs is wrong in config (from config) {fs}!={fs_} (from audio)')
    if dtype is not None:
        data_arr = data_arr.astype(dtype)

    s = Signal(data_arr=data_arr, fs=fs_, meta=meta)
    if interval is not None:
        return s.crop(interval)

    return s


def read_bin(filename, interval: tuple[int, int] = None, source_dtype='float32', dtype=None, meta=None,
             fs: float = None) -> Signal:
    with open(filename, "rb") as f:
        start_byte = int(DTYPE_BYTES[source_dtype] * interval[0]) if interval is not None else 0
        assert start_byte % DTYPE_BYTES[source_dtype] == 0, "Bytes are not loading properly."
        f.seek(start_byte, 0)
        count = (interval[1] - interval[0]) if interval is not None else -1
        data_arr = np.expand_dims(np.fromfile(f, dtype=source_dtype, count=count), axis=0)
        if dtype is not None:
            data_arr = data_arr.astype(dtype)
        return Signal(data_arr=data_arr, meta=meta, fs=fs)


def from_numpy(data_arr: np.ndarray, interval: tuple[int, int] = None, dtype=None, fs: float = None, meta=None) -> TimeSeries:
    assert isinstance(data_arr, np.ndarray), f"data_arr must be a numpy.ndarray, not '{type(data_arr)}'"
    if len(data_arr.shape) == 1:
        data_arr = np.expand_dims(data_arr, axis=0)
    if interval is not None:
        data_arr = data_arr[..., interval[0]: interval[1]]
    if dtype is not None:
        data_arr = data_arr.astype(dtype)

    if len(data_arr.shape) == 2:
        return Signal(data_arr=data_arr, meta=meta, fs=fs)
    elif len(data_arr.shape) == 3:
        return Signal2D(data_arr=data_arr, meta=meta, fs=fs)
    else:
        raise ValueError(f"Loaded array has {len(data_arr.shape)} channels, maximum is 3.")


def read_npy(filename: str | pathlib.PosixPath, interval: tuple[int, int] = None, dtype=None, fs: float = None, meta=None) -> TimeSeries:
    full_data_arr = np.load(filename)
    return from_numpy(
        data_arr=full_data_arr,
        interval=interval,
        dtype=dtype,
        fs=fs,
        meta=meta,
    )
