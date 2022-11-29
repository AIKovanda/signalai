from typing import Union, List

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
from taskchain.parameter import AutoParameterObject

from signalai.time_series import Signal, Signal2D
from signalai.tools.filters import butter_bandpass_filter
from signalai.time_series import TimeSeries, from_numpy
from tools.utils import by_channel

RETURN_CLASS = {
    2: Signal,
    3: Signal2D,
}


# class MultiSeries:
#
#     def sum_channels(self, channels: Optional[List[Union[int, List[int]]]] = None):
#         series = self._series_list(only_valid=True)
#         series_length = len(series[0])
#         assert all([len(ts) == series_length for ts in series]), 'Timeseries must have a same length to be joined.'
#         fs_set = {ts.fs for ts in series}
#         assert len(fs_set) == 1, f'Timeseries must have the same sampling frequency. {fs_set}'
#         if channels is None:
#             series_channels = series[0].channels_count
#             assert all([ts.channels_count == series_channels for ts in series]), \
#                 'Timeseries must have the same number of channels to be joined when channels are not defined.'
#
#         zero_series = series[0].take_channels(channels)
#         data_arrays = [zero_series.data_arr]
#         time_map = zero_series.time_map.copy()
#         metas = [zero_series.meta]
#
#         for ts in series[1:]:
#             s_series = ts.take_channels(channels)
#             data_arrays.append(s_series.data_arr)
#             time_map = np.logical_or(time_map, s_series.time_map)
#             metas.append(s_series.meta)
#
#         assert series[0].fs is not None, series
#         return type(series[0])(
#             data_arr=np.sum(data_arrays, axis=0),
#             time_map=time_map,
#             meta=join_dicts(*metas),
#             fs=series[0].fs,
#         )
#
#     def stack_series(self, only_valid=False, axis=0):
#         series = self._series_list(only_valid=only_valid)
#
#         series_shape = None
#         ts_type = None
#         time_map_shape = None
#
#         for ts in series:
#             if ts is not None:
#                 ts_type = type(ts)
#                 time_map_shape = ts.time_map.shape
#                 if ts.data_arr is not None:
#                     series_shape = ts.data_arr.shape
#                     break
#         else:
#             if time_map_shape is None:  # meaning there is no TimeSeries at all
#                 raise ValueError(f"At least one timeseries must not be empty while stacking timeseries.")
#
#         if series_shape is not None:
#             assert all([ts.data_arr.shape == series_shape for ts in series if ts is not None]), \
#                 'Timeseries must have the same shapes to be joined.'
#
#         data_arrays = []
#         time_maps = []
#         fss = []
#         metas = []
#
#         for ts in series:
#             if ts is not None:
#                 if series_shape is not None:
#                     data_arrays.append(ts.data_arr)
#                 fss.append(ts.fs)
#                 time_maps.append(ts.time_map)
#                 metas.append(ts.meta)
#             else:
#                 if series_shape is not None:
#                     data_arrays.append(np.zeros(series_shape))
#                 time_maps.append(np.zeros(time_map_shape, dtype=bool))
#
#         if series_shape is not None:
#             data_arr = np.concatenate(data_arrays, axis=axis)
#         else:
#             data_arr = None
#
#         assert len(set(fss)) == 1, fss  # todo: check if this works good
#         return ts_type(
#             data_arr=data_arr,
#             time_map=np.concatenate(time_maps, axis=0),
#             meta=join_dicts(*metas),
#             fs=fss[0],
#         )


class Transformer(AutoParameterObject):
    in_dim = None

    def __init__(self, **params):
        self.params = params
        self.evaluated_params = {}
        for key, val in params.items():
            if isinstance(val, str):
                val = eval(val)
            self.evaluated_params[key] = val

    def _get_parameter_uniform(self, param_name: str, default: Union[float, List[float]]) -> float:
        param_value = self.evaluated_params.get(param_name, default)
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) == 1:
                return param_value[0]

            if len(param_value) == 2:
                return np.random.uniform(*param_value)

            raise ValueError(f"Parameter '{param_name}' must have one or two values, not {len(param_value)}.")

        return param_value

    def transform_timeseries(self, x: TimeSeries) -> Union[TimeSeries, np.ndarray]:
        ...

    def original_signal_length(self, length: int, fs: int = None) -> int:
        raise NotImplementedError("Original length is not implemented for this transformation.")

    @property
    def keeps_signal_length(self) -> bool:
        raise NotImplementedError("Keeps length is not implemented for this transformation.")

    def _transform(self, ts: TimeSeries):
        if self.in_dim is not None and self.in_dim != ts.full_dimensions:
            raise ValueError(f"Transform '{type(self)}' takes input of dim {self.in_dim}, "
                             f"{type(ts)} has a dim of {ts.full_dimensions}.")

        transform_chance = self.evaluated_params.get('transform_chance', 1.)
        if np.random.rand() <= transform_chance:
            return self.transform_timeseries(ts)

        return ts

    def transform(self, x: Union[np.ndarray, TimeSeries]) -> Union[TimeSeries]:
        if isinstance(x, TimeSeries):
            ts = x
            return self._transform(ts)

        elif isinstance(x, np.ndarray):
            ts = from_numpy(x)
            return self._transform(ts)

        else:
            raise TypeError(
                f"Transformer got a type of '{type(x)}' which is not supported.")

    def __call__(self, x):
        return self.transform(x)


class Resampler(Transformer):
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, input_fs: int, output_fs: int) -> np.ndarray:
        return np.expand_dims(resample(
            x[0].astype('float32'), input_fs, output_fs, res_type='linear'
        ).astype(x.dtype), 0)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        input_fs = x.fs
        assert input_fs is not None, 'Missing info about sampling frequency!'
        output_fs = self.evaluated_params.get('output_fs')
        if input_fs == output_fs:
            return x

        data_arr = self.transform_npy(x.data_arr, input_fs=input_fs, output_fs=output_fs)
        return Signal(
            data_arr=data_arr,
            time_map=x.time_map,
            meta=x.meta,
            fs=output_fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return int(length * fs / self.evaluated_params.get('output_fs'))

    @property
    def keeps_signal_length(self) -> bool:
        return False


class Standardizer(Transformer):
    in_dim = 2

    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        mean = self.evaluated_params.get('mean', 0)
        std = self.evaluated_params.get('std', 1)

        new_x = (x - np.mean(x)) / np.std(x)
        return (new_x * std) + mean

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class Gain(Transformer):
    """
    gain_db: gain in decibels, can be negative
    """
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, fs: int) -> np.ndarray:
        gain_db = self._get_parameter_uniform('gain_db', default=[-10, 10])
        return PBGain(gain_db=gain_db)(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class Phaser(Transformer):
    """
    Does not take parameters.
    """
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, fs: int) -> np.ndarray:
        return PBPhaser()(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class Chorus(Transformer):
    """
    centre_delay_ms: between 7 and 8 is reminds Chorus the most
    """
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, fs: int) -> np.ndarray:
        centre_delay_ms = self._get_parameter_uniform('centre_delay_ms', default=[7., 8.])
        return PBChorus(centre_delay_ms=centre_delay_ms)(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class Reverb(Transformer):
    """
    Parameters in range [0,1]:
        room_size
        damping
        wet_level
        dry_level
        freeze_mode
    """
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, fs: int) -> np.ndarray:
        room_size = self._get_parameter_uniform('room_size', default=[0., 1.])
        damping = self._get_parameter_uniform('damping', default=[0., 1.])
        wet_level = self._get_parameter_uniform('wet_level', default=[0., 1.])
        dry_level = self._get_parameter_uniform('dry_level', default=[0., 1.])
        freeze_mode = self._get_parameter_uniform('freeze_mode', default=[0., .1])
        return PBReverb(
            room_size=room_size,
            damping=damping,
            wet_level=wet_level,
            dry_level=dry_level,
            freeze_mode=freeze_mode,
        )(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class BandPassFilter(Transformer):
    """
    low_cut - None or frequency
    high_cut - None or frequency
    """
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, fs: int) -> np.ndarray:
        low_cut = self.evaluated_params.get('low_cut')
        high_cut = self.evaluated_params.get('high_cut')

        return butter_bandpass_filter(x, low_cut, high_cut, fs)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class ChannelJoiner(Transformer):
    """
    channels: list containing list of integers
    """
    in_dim = 2

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        choose_channels = self.evaluated_params.get("choose_channels")
        if choose_channels is not None:
            channels = choose_channels[np.random.choice(len(choose_channels))]
        else:
            channels = self.evaluated_params.get("channels", [list(range(x.data_arr.shape[0]))])
        return x.take_channels(channels=channels)

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class Lambda(Transformer):
    """
    channels: list containing list of integers
    """

    def transform(self, x: TimeSeries) -> TimeSeries:
        return self.evaluated_params.get("lambda")(x)

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class TimeMapScale(Transformer):
    """
    channels: list containing list of integers
    """

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        target_length = self.evaluated_params.get("target_length")
        if target_length is None:
            target_length = len(x) * self.evaluated_params.get("scale")

        time_map = x.time_map.astype(int)
        # nearest
        new_time_map = time_map[:, np.round(np.linspace(0, time_map.shape[-1] - 1, int(target_length))).astype(int)]
        return TimeSeries(time_map=new_time_map, meta=x.meta)

    def original_signal_length(self, length: int, fs: int = None) -> int:  # todo: another approach
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class STFT(Transformer):
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, split_complex=False) -> np.ndarray:
        center = self.evaluated_params.get('center', False)
        n_fft = self.evaluated_params.get('n_fft', 256)
        hop_length = self.evaluated_params.get('hop_length')
        # _, _, Zxx = scipy_signal.stft(x[0], fs=fs)
        Zxx = librosa.stft(x[0], center=center, n_fft=n_fft, hop_length=hop_length)
        if split_complex:
            return np.moveaxis(Zxx.view('(2,)float32'), -1, 0)

        return np.expand_dims(Zxx, 0)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        if self.evaluated_params.get('phase_as_meta', True):
            Zxx = self.transform_npy(x.data_arr, split_complex=False)
            data_arr = np.abs(Zxx)  # magnitude
            meta = x.meta.copy() | {'phase': np.angle(Zxx)}
        else:
            data_arr = self.transform_npy(x.data_arr, split_complex=True)
            meta = x.meta

        return Signal2D(
            data_arr=data_arr,
            time_map=x.time_map,  # todo
            meta=meta,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class ISTFT(Transformer):
    in_dim = 3

    @by_channel
    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        # return scipy_signal.istft(x)[1]
        center = self.evaluated_params.get('center', False)
        hop_length = self.evaluated_params.get('hop_length')
        return np.expand_dims(librosa.istft(x[0], hop_length=hop_length, center=center), 0)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        meta = x.meta.copy()  # todo: without copy
        phase = meta.pop('phase', None)
        data_arr = x.data_arr

        if self.evaluated_params.get('phase_as_meta', True):
            assert phase is not None, "Phase must be included in the input information"
            Zxx = data_arr * np.exp(1j * phase)
        else:
            Zxx = data_arr[0::2] + 1j * data_arr[1::2]

        s_data_arr = self.transform_npy(Zxx)
        return Signal(
            data_arr=s_data_arr,
            time_map=x.time_map,  # todo
            meta=meta,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True
