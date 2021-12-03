import abc
import pathlib
from pathlib import Path

import numpy as np

import signalai
from signalai.signal import Logger


class SignalModel(abc.ABC):
    def __init__(self, model, model_type, training_params, save_dir, logger=None, signal_generator=None):
        super().__init__()
        self.model = model
        self.signal_generator = signal_generator
        self.model_type = model_type
        self.training_params = training_params
        self.save_dir = Path(save_dir) if save_dir is not None else None

        self.evaluator = None
        self.criterion = None
        self.optimizer = None

        self.logger = logger or Logger(file=save_dir / "model.log")

    def train_on_generator(self):
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        if self.evaluator is None:
            eval_info = self.training_params.get("evaluator", None)
            if eval_info is not None:
                assert False
                # self.evaluator = get_instance(eval_info["class"], {
                #     "gen_gen": self.gen_generator})

        self._train_on_generator()

    @abc.abstractmethod
    def train_on_batch(self, x, y):
        pass

    @abc.abstractmethod
    def predict_batch(self, x) -> np.ndarray:
        pass
    
    def predict_numpy(self, x, split_by=None, residual_end=True):
        if split_by is None or split_by >= x.shape[-1]:
            return self.predict_batch(np.expand_dims(x, 0))[0]
        else:
            result = []
            for i in range(x.shape[-1] // split_by):
                result.append(self.predict_batch(np.expand_dims(x[..., i*split_by:(i+1)*split_by], 0))[0])
            if x.shape[-1] % split_by > 0 and residual_end:
                result.append(self.predict_batch(np.expand_dims(x[..., (x.shape[-1] % split_by):], 0))[0])
            return np.concatenate(result, axis=-1)

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass

    @abc.abstractmethod
    def _train_on_generator(self):
        pass

    @abc.abstractmethod
    def get_optimizer(self):
        pass

    @abc.abstractmethod
    def get_criterion(self):
        pass

    def __call__(self, x, split_by=None, residual_end=True):
        if isinstance(x, signalai.signal.Signal):
            return self.predict_numpy(x.signal, residual_end=residual_end)
        elif isinstance(x, np.ndarray):
            return self.predict_numpy(x, residual_end=residual_end)

        raise TypeError(f"X cannot be of type '{type(x)}'.")
