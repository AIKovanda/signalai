from abc import abstractmethod

import numpy as np
from signalai.signal import Signal
from signalai.tools.filters import butter_bandpass_filter


class Transformer:
    by_channel = None

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        pass

    def transform_numpy(self, x):
        signal_arr = x.copy()
        if self.by_channel:
            for i in range(signal_arr.shape[0]):
                signal_arr[i] = self.transform(signal_arr[i])
            return signal_arr

        return self.transform(signal_arr)

    @abstractmethod
    def original_length(self, length: int) -> int:
        pass

    @property
    @abstractmethod
    def keeps_length(self) -> bool:
        pass

    def __call__(self, x):
        if isinstance(x, Signal):
            signal_arr = self.transform_numpy(x.signal)
            signal_map = x.signal_map if self.keeps_length else None
            return Signal(build_from=signal_arr, meta=x.meta, signal_map=signal_map, logger=x.logger)
        elif isinstance(x, np.ndarray):
            return self.transform_numpy(x)
        else:
            raise TypeError(f"Transformer got a type of '{type(x)}' which is not supported.")


class Standardizer(Transformer):
    by_channel = False

    def transform(self, x):
        mean = self.params.get('mean', 0)
        std = self.params.get('std', 1)

        new_x = (x - np.mean(x)) / np.std(x)
        return (new_x * std) + mean

    def original_length(self, length: int) -> int:
        return length

    @property
    def keeps_length(self) -> bool:
        return True


class BandPassFilter(Transformer):
    by_channel = True

    def transform(self, x):
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
    channels is a list of list of integers
    """
    by_channel = False

    def transform(self, x: np.ndarray) -> np.ndarray:
        channels = self.params.get("channels", [list(range(x.shape[0]))])

        if all([isinstance(i, int) for i in channels]):
            channels = [channels]

        new_signal = []
        for new_channel in channels:
            assert isinstance(new_channel, list), "channels is a list of list of integers"
            new_signal.append(np.sum(x[new_channel, :], axis=0))

        return np.array(new_signal)

    def original_length(self, length: int) -> int:
        return length

    @property
    def keeps_length(self) -> bool:
        return True
