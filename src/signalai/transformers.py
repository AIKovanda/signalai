from abc import abstractmethod
import abc
from typing import Union, List

import numpy as np
from taskchain.parameter import AutoParameterObject
import signalai
from scipy import signal as scipy_signal
from signalai.timeseries import Signal, Signal2D
from signalai.tools.filters import butter_bandpass_filter
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
    by_channel = None
    in_dim = None
    out_dim = None

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
    def _transform_numpy(self, x: np.ndarray) -> np.ndarray:
        pass

    def _transform_data_arr(self, data_arr: np.ndarray) -> np.ndarray:
        transform_chance = self.params.get('transform_chance', 1.)
        transform = [lambda x: x, self._transform_numpy][int(np.random.rand() < transform_chance)]

        if self.by_channel:
            new_data_arr = np.zeros_like(data_arr)
            for i in range(data_arr.shape[0]):
                new_data_arr[i] = transform(data_arr[i])
            return new_data_arr

        return transform(data_arr)

    @abstractmethod
    def original_length(self, length: int) -> int:
        pass

    @property
    @abstractmethod
    def keeps_length(self) -> bool:
        pass

    def transform(self, x):
        if isinstance(x, signalai.timeseries.TimeSeries):
            if self.in_dim != x.full_dimensions:
                raise ValueError(f"Transform '{type(self)}' takes input of dim {self.in_dim}, "
                                 f"{type(x)} has a dim of {x.full_dimensions}.")
            data_arr = self._transform_data_arr(x.data_arr)
            time_map = x.time_map if self.keeps_length else None
            return RETURN_CLASS[self.out_dim](
                data_arr=data_arr,
                meta=x.meta,
                time_map=time_map,
                logger=x.logger,
            )

        elif isinstance(x, np.ndarray):
            return self._transform_data_arr(x)

        else:
            raise TypeError(
                f"Transformer got a type of '{type(x)}' which is not supported.")

    def __call__(self, x):
        return self.transform(x)


class Standardizer(Transformer):
    by_channel = False
    in_dim = 2
    out_dim = 2

    def _transform_numpy(self, x: np.ndarray) -> np.ndarray:
        mean = self.params.get('mean', 0)
        std = self.params.get('std', 1)

        new_x = (x - np.mean(x)) / np.std(x)
        return (new_x * std) + mean

    def original_length(self, length: int) -> int:
        return length

    @property
    def keeps_length(self) -> bool:
        return True


class Gain(Transformer):
    """
    gain_db: gain in decibels, can be negative
    """
    by_channel = True
    in_dim = 2
    out_dim = 2

    def _transform_numpy(self, x: np.ndarray) -> np.ndarray:
        gain_db = self._get_parameter_uniform('gain_db', default=[-10, 10])
        fs = self.params.get('fs', 44100)
        return PBGain(gain_db=gain_db)(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def original_length(self, length: int) -> int:
        return length

    @property
    def keeps_length(self) -> bool:
        return True


class Phaser(Transformer):
    """
    Does not take parameters.
    """
    by_channel = True
    in_dim = 2
    out_dim = 2

    def _transform_numpy(self, x: np.ndarray) -> np.ndarray:
        fs = self.params.get('fs', 44100)
        return PBPhaser()(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def original_length(self, length: int) -> int:
        return length

    @property
    def keeps_length(self) -> bool:
        return True


class Chorus(Transformer):
    """
    centre_delay_ms: between 7 and 8 is reminds Chorus the most
    """
    by_channel = True
    in_dim = 2
    out_dim = 2

    def _transform_numpy(self, x: np.ndarray) -> np.ndarray:
        centre_delay_ms = self._get_parameter_uniform('centre_delay_ms', default=[7., 8.])
        fs = self.params.get('fs', 44100)
        return PBChorus(centre_delay_ms=centre_delay_ms)(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def original_length(self, length: int) -> int:
        return length

    @property
    def keeps_length(self) -> bool:
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
    by_channel = True
    in_dim = 2
    out_dim = 2

    def _transform_numpy(self, x: np.ndarray) -> np.ndarray:
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

    def original_length(self, length: int) -> int:
        return length

    @property
    def keeps_length(self) -> bool:
        return True


class BandPassFilter(Transformer):
    """
    low_cut - None or frequency
    high_cut - None or frequency
    """
    by_channel = True
    in_dim = 2
    out_dim = 2

    def _transform_numpy(self, x: np.ndarray) -> np.ndarray:
        fs = self.params.get('fs', 44100)
        low_cut = self.params.get('low_cut')
        high_cut = self.params.get('high_cut')

        return butter_bandpass_filter(x, low_cut, high_cut, fs)

    def original_length(self, length: int) -> int:
        return length

    @property
    def keeps_length(self) -> bool:
        return True


class ChannelJoiner(Transformer):
    """
    channels: list of list of integers
    """
    by_channel = False
    in_dim = 2
    out_dim = 2

    def _transform_numpy(self, x: np.ndarray) -> np.ndarray:
        choose_channels = self.params.get("choose_channels")
        if choose_channels is not None:
            channels = choose_channels[np.random.choice(len(choose_channels))]
        else:
            channels = self.params.get("channels", [list(range(x.shape[0]))])
        return Signal(data_arr=x).take_channels(channels=channels).data_arr

    def original_length(self, length: int) -> int:
        return length

    @property
    def keeps_length(self) -> bool:
        return True


class STFT(Transformer):
    by_channel = True
    in_dim = 2
    out_dim = 3

    def _transform_numpy(self, x: np.ndarray) -> np.ndarray:
        fs = self.params.get('fs', 44100)
        _, _, Zxx = scipy_signal.stft(x, fs=fs)
        return Zxx

    def original_length(self, length: int) -> int:
        return length

    @property
    def keeps_length(self) -> bool:
        return True
