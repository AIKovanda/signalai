from abc import abstractmethod
import abc
from typing import Union, List

from taskchain.parameter import AutoParameterObject
import numpy as np
from scipy import signal as scipy_signal

from signalai.timeseries import Signal, Signal2D, TimeSeries, from_numpy
from signalai.tools.filters import butter_bandpass_filter
from signalai.tools.utils import by_channel

from pedalboard import (
    Chorus as PBChorus,
    Gain as PBGain,  # Increase or decrease the volume of a signal by applying a gain value (in decibels).
    Reverb as PBReverb,  # Performs a simple reverb effect on a stream of audio data.
    Phaser as PBPhaser,  # A 6 stage phaser that modulates first order all-pass filters to create sweeping notches
                         # in the magnitude frequency response
)


RETURN_CLASS = {
    2: Signal,
    3: Signal2D,
}


class Transformer(AutoParameterObject, abc.ABC):
    in_dim = None

    def __init__(self, **params):
        self.params = params

    def _get_parameter_uniform(self, param_name: str, default: Union[float, int, List[Union[int, float]]]) -> float:
        param_value = self.params.get(param_name, default)
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) == 1:
                return param_value[0]

            if len(param_value) == 2:
                return np.random.uniform(*param_value)

            raise ValueError(f"Parameter '{param_name}' must have one or two values, not {len(param_value)}.")

        return param_value

    @abstractmethod
    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        pass

    def original_signal_length(self, length: int) -> int:
        raise NotImplementedError("Original length is not implemented for this transformation.")

    @property
    def keeps_signal_length(self) -> bool:
        raise NotImplementedError("Keeps length is not implemented for this transformation.")

    def transform(self, x: Union[np.ndarray, TimeSeries]) -> TimeSeries:
        if isinstance(x, TimeSeries):
            ts = x
        elif isinstance(x, np.ndarray):
            ts = from_numpy(x)
        else:
            raise TypeError(
                f"Transformer got a type of '{type(x)}' which is not supported.")

        if self.in_dim != ts.full_dimensions:
            raise ValueError(f"Transform '{type(self)}' takes input of dim {self.in_dim}, "
                             f"{type(ts)} has a dim of {ts.full_dimensions}.")

        transform_chance = self.params.get('transform_chance', 1.)
        if np.random.rand() < transform_chance:
            return self.transform_timeseries(ts)

        return ts

    def __call__(self, x):
        return self.transform(x)


class Standardizer(Transformer):
    in_dim = 2

    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        mean = self.params.get('mean', 0)
        std = self.params.get('std', 1)

        new_x = (x - np.mean(x)) / np.std(x)
        return (new_x * std) + mean

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
        )

    def original_signal_length(self, length: int) -> int:
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
    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        gain_db = self._get_parameter_uniform('gain_db', default=[-10, 10])
        fs = self.params.get('fs', 44100)
        return PBGain(gain_db=gain_db)(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
        )

    def original_signal_length(self, length: int) -> int:
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
    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        fs = self.params.get('fs', 44100)
        return PBPhaser()(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
        )

    def original_signal_length(self, length: int) -> int:
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
    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        centre_delay_ms = self._get_parameter_uniform('centre_delay_ms', default=[7., 8.])
        fs = self.params.get('fs', 44100)
        return PBChorus(centre_delay_ms=centre_delay_ms)(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
        )

    def original_signal_length(self, length: int) -> int:
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
    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        room_size = self._get_parameter_uniform('room_size', default=[0., 1.])
        damping = self._get_parameter_uniform('damping', default=[0., 1.])
        wet_level = self._get_parameter_uniform('wet_level', default=[0., 1.])
        dry_level = self._get_parameter_uniform('dry_level', default=[0., 1.])
        freeze_mode = self._get_parameter_uniform('freeze_mode', default=[0., .1])
        fs = self.params.get('fs', 44100)
        return PBReverb(
            room_size=room_size,
            damping=damping,
            wet_level=wet_level,
            dry_level=dry_level,
            freeze_mode=freeze_mode,
        )(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
        )

    def original_signal_length(self, length: int) -> int:
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
    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        fs = self.params.get('fs', 44100)
        low_cut = self.params.get('low_cut')
        high_cut = self.params.get('high_cut')

        return butter_bandpass_filter(x, low_cut, high_cut, fs)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
        )

    def original_signal_length(self, length: int) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class ChannelJoiner(Transformer):
    """
    channels: list of list of integers
    """
    in_dim = 2

    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        choose_channels = self.params.get("choose_channels")
        if choose_channels is not None:
            channels = choose_channels[np.random.choice(len(choose_channels))]
        else:
            channels = self.params.get("channels", [list(range(x.shape[0]))])
        return Signal(data_arr=x).take_channels(channels=channels).data_arr

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        data_arr = self.transform_npy(x.data_arr)
        return Signal(
            data_arr=data_arr,
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
        )

    def original_signal_length(self, length: int) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class STFT(Transformer):
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, split_complex=False) -> np.ndarray:
        fs = self.params.get('fs', 44100)
        _, _, Zxx = scipy_signal.stft(x[0], fs=fs)
        if split_complex:
            return np.moveaxis(Zxx.view('(2,)float32'), -1, 0)

        return np.expand_dims(Zxx, 0)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        if self.params.get('phase_as_meta', True):
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
            logger=x.logger,
        )

    def original_signal_length(self, length: int) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class ISTFT(Transformer):
    in_dim = 3

    @by_channel
    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        return scipy_signal.istft(x)[1]

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        meta = x.meta.copy()  # todo: without copy
        phase = meta.pop('phase', None)
        data_arr = x.data_arr

        if self.params.get('phase_as_meta', True):
            assert phase is not None, "Phase must be included in the input information"
            Zxx = data_arr * np.exp(1j * phase)
        else:
            Zxx = data_arr[0::2] + 1j * data_arr[1::2]

        s_data_arr = self.transform_npy(Zxx)
        return Signal(
            data_arr=s_data_arr,
            time_map=x.time_map,  # todo
            meta=meta,
            logger=x.logger,
        )

    def original_signal_length(self, length: int) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True
