from abc import abstractmethod

import numpy as np
from signalai.signal_tools.signal import Signal


class Transformer:
    @abstractmethod
    def transform(self, x):
        pass

    def __call__(self, x):
        if isinstance(x, Signal):
            return self.transform(x.signal)

        return self.transform(x)


class Standardizer(Transformer):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def transform(self, x):
        mean_ = np.mean(x)
        std_ = np.std(x)
        x = (x - mean_) / std_
        return (x * self.std) + self.mean
