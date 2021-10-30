from abc import abstractmethod

import numpy as np
from signalai.signal.signal import Signal


class Transformer:
    @abstractmethod
    def transform(self, x):
        pass

    def __call__(self, x):
        if isinstance(x, Signal):
            return Signal(self.transform(x.signal), info=x.info, signal_map=x.signal_map)

        return self.transform(x)


class Standardizer(Transformer):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def transform(self, x):
        mean_ = np.mean(x)
        std_ = np.std(x)
        y = (x - mean_) / std_
        return (y * self.std) + self.mean
