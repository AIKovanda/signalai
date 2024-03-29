from typing import Callable

import librosa
import numpy as np
from librosa import resample
from pedalboard import \
    Chorus as PBChorus  # in the magnitude frequency response
from pedalboard import \
    Gain as \
        PBGain  # Increase or decrease the volume of a signal by applying a gain value (in decibels).
from pedalboard import \
    Phaser as \
        PBPhaser  # A 6 stage phaser that modulates first order all-pass filters to create sweeping notches
from pedalboard import \
    Reverb as \
        PBReverb  # Performs a simple reverb effect on a stream of audio data.

from signalai.time_series import Signal, Signal2D
from signalai.time_series import TimeSeries
from signalai.time_series_gen import Transformer
from signalai.tools.filters import butter_bandpass_filter


def by_channel(transform: Callable):
    def by_channel_wrapper(self, *args, **kwargs):
        arg_len = set([len(arg) for arg in args])
        assert len(arg_len) == 1, f"Inputs must be the same length."
        processed_channels = []
        for i in range(list(arg_len)[0]):
            processed_channels.append(
                transform(self, *[arg[i: i+1] for arg in args], **kwargs)
            )
        return np.concatenate(processed_channels, axis=0)

    return by_channel_wrapper


def get_parameter_uniform(param_value: float | list[float] | None, default: float | list[float] | None) -> float:
    if param_value is None:
        param_value = default
    if isinstance(param_value, list) or isinstance(param_value, tuple):
        if len(param_value) == 1:
            return param_value[0]
        elif len(param_value) == 2:
            return np.random.uniform(*param_value)
        else:
            raise ValueError(f'Parameter is a list/tuple with more than 2 values ({len(param_value)}).')

    return param_value


class Resampler(Transformer):
    takes = 'time_series'

    def _build(self):
        assert 'fs_ratio' in self.config, 'output_fs / input_fs'

    @by_channel
    def process_numpy(self, x: np.ndarray, input_fs: float, output_fs: float) -> np.ndarray:
        return np.expand_dims(resample(
            x[0].astype('float32'), input_fs, output_fs, res_type='linear',
        ).astype(x.dtype), 0)

    def _process(self, x: TimeSeries) -> TimeSeries:
        input_fs = x.fs
        assert input_fs is not None, 'Missing info about sampling frequency!'
        fs_ratio = self.config['fs_ratio']
        if fs_ratio == 1:
            return x

        output_fs = fs_ratio * input_fs
        data_arr = self.process_numpy(x.data_arr, input_fs=input_fs, output_fs=output_fs)
        return Signal(
            data_arr=data_arr,
            time_map=x.time_map,
            meta=x.meta,
            fs=output_fs,
        )

    def transform_taken_length(self, length: int) -> int:
        return int(length / self.config['fs_ratio'])


class Standardizer(Transformer):
    takes = 'time_series'

    def process_numpy(self, x: np.ndarray) -> np.ndarray:
        mean = self.config.get('mean', 0)
        std = self.config.get('std', 1)

        new_x = (x - np.mean(x)) / np.std(x)
        return (new_x * std) + mean

    def _process(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.process_numpy(x.data_arr),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def transform_taken_length(self, length: int) -> int:
        return length


class Gain(Transformer):
    """
    gain_db: gain in decibels, can be negative
    """
    takes = 'time_series'

    @by_channel
    def process_numpy(self, x: np.ndarray, fs: float) -> np.ndarray:
        gain_db = get_parameter_uniform(self.config.get('gain_db'), default=[-10, 10])
        return PBGain(gain_db=gain_db)(input_array=x.astype('float32'), sample_rate=fs).astype(x.dtype)

    def _process(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.process_numpy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def transform_taken_length(self, length: int) -> int:
        return length


class Phaser(Transformer):
    """
    Does not take parameters.
    """
    takes = 'time_series'

    @by_channel
    def process_numpy(self, x: np.ndarray, fs: float) -> np.ndarray:
        return PBPhaser()(input_array=x.astype('float32'), sample_rate=fs).astype(x.dtype)

    def _process(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.process_numpy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def transform_taken_length(self, length: int) -> int:
        return length


class Chorus(Transformer):
    """
    centre_delay_ms: between 7 and 8 is reminds Chorus the most
    """
    takes = 'time_series'

    @by_channel
    def process_numpy(self, x: np.ndarray, fs: float) -> np.ndarray:
        centre_delay_ms = get_parameter_uniform(self.config.get('centre_delay_ms'), default=[7., 8.])
        return PBChorus(centre_delay_ms=centre_delay_ms)(input_array=x.astype('float32'), sample_rate=fs).astype(x.dtype)

    def _process(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.process_numpy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def transform_taken_length(self, length: int) -> int:
        return length


class Reverb(Transformer):
    """
    Parameters in range [0,1]:
        room_size
        damping
        wet_level
        dry_level
        freeze_mode
    """
    takes = 'time_series'

    @by_channel
    def process_numpy(self, x: np.ndarray, fs: float) -> np.ndarray:
        room_size = get_parameter_uniform(self.config.get('room_size'), default=[0., 1.])
        damping = get_parameter_uniform(self.config.get('damping'), default=[0., 1.])
        wet_level = get_parameter_uniform(self.config.get('wet_level'), default=[0., 1.])
        dry_level = get_parameter_uniform(self.config.get('dry_level'), default=[0., 1.])
        freeze_mode = get_parameter_uniform(self.config.get('freeze_mode'), default=[0., .1])
        return PBReverb(
            room_size=room_size,
            damping=damping,
            wet_level=wet_level,
            dry_level=dry_level,
            freeze_mode=freeze_mode,
        )(input_array=x.astype('float32'), sample_rate=fs).astype(x.dtype)

    def _process(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.process_numpy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def transform_taken_length(self, length: int) -> int:
        return length


class BandPassFilter(Transformer):
    """
    low_cut - None or frequency
    high_cut - None or frequency
    """
    takes = 'time_series'

    @by_channel
    def process_numpy(self, x: np.ndarray, fs: float) -> np.ndarray:
        low_cut = self.config.get('low_cut')
        high_cut = self.config.get('high_cut')

        return butter_bandpass_filter(x, low_cut, high_cut, fs)

    def _process(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.process_numpy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def transform_taken_length(self, length: int) -> int:
        return length


class STFT(Transformer):
    takes = 'time_series'

    @by_channel
    def process_numpy(self, x: np.ndarray, split_complex=True) -> np.ndarray:
        center = self.config.get('center', False)
        n_fft = self.config.get('n_fft', 256)
        hop_length = self.config.get('hop_length')
        Zxx = librosa.stft(x[0], center=center, n_fft=n_fft, hop_length=hop_length)
        if split_complex:
            return np.moveaxis(Zxx.view('(2,)float32'), -1, 0)

        return np.expand_dims(Zxx, 0)

    def _process(self, x: TimeSeries) -> TimeSeries:
        n_fft = self.config.get('n_fft', 256)
        if self.config.get('phase_as_meta', True):
            Zxx = self.process_numpy(x.data_arr, split_complex=False)
            data_arr = np.abs(Zxx)  # magnitude
            meta = x.meta.copy() | {'phase': np.angle(Zxx)}
        else:
            data_arr = self.process_numpy(x.data_arr, split_complex=True)
            meta = x.meta

        return Signal2D(
            data_arr=data_arr,
            time_map=x.time_map[..., int(n_fft/2): -int(n_fft/2)],
            meta=meta,
            fs=x.fs,
        )

    def transform_taken_length(self, length: int) -> int:
        return length


class ISTFT(Transformer):
    takes = 'time_series'

    def process_torch(self, x, phase=None):
        import torch
        device = x.device
        return torch.from_numpy(self.process_numpy(x.detach().cpu().numpy(), phase)).to(device)

    def to_complex(self, x: np.ndarray, phase=None) -> np.ndarray:
        if self.config.get('phase_as_meta', True):
            assert phase is not None, "Phase must be included in the input information"
            Zxx = x * np.exp(1j * phase)
        else:
            Zxx = x[0::2] + 1j * x[1::2]

        return Zxx

    def process_numpy(self, x: np.ndarray, phase=None) -> np.ndarray:
        return self._process_numpy(self.to_complex(x, phase))

    @by_channel
    def _process_numpy(self, x: np.ndarray) -> np.ndarray:
        center = self.config.get('center', False)
        hop_length = self.config.get('hop_length')
        return np.expand_dims(librosa.istft(x[0], hop_length=hop_length, center=center), 0)

    def _process(self, x: TimeSeries) -> TimeSeries:
        n_fft = self.config.get('n_fft', 256)
        meta = x.meta.copy()
        phase = meta.pop('phase', None)
        data_arr = x.data_arr

        s_data_arr = self.process_numpy(self.to_complex(data_arr, phase))
        new_time_map = np.zeros([*x.time_map.shape[:-1], x.time_map.shape[-1]+n_fft])
        new_time_map[..., int(n_fft/2): int(n_fft/2) + x.time_map.shape[-1]] = x.time_map
        return Signal(
            data_arr=s_data_arr,
            time_map=new_time_map,
            meta=meta,
            fs=x.fs,
        )

    def transform_taken_length(self, length: int) -> int:
        return length
