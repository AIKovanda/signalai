import abc
from pathlib import Path

import numpy as np
import pandas as pd
from taskchain.parameter import AutoParameterObject

from signalai.time_series import TimeSeries
from signalai.time_series_gen import Transformer
from signalai.torch_dataset import TorchDataset


class TimeSeriesModel(AutoParameterObject, abc.ABC):
    def __init__(self, model_definition, save_dir=".", device='cpu', taken_length: int = None, model_count=1,
                 training_params: dict = None, training_echo: dict = None, pre_transform: Transformer = None,
                 post_transform: Transformer = None):

        super().__init__()

        self.model_definition = model_definition
        self.device = device
        self.model = None
        self.init_model(model_definition)
        self.save_dir = Path(save_dir)

        self.taken_length = taken_length
        self.model_count = model_count

        self.training_params = training_params
        self.training_echo = training_echo

        self.pre_transform: Transformer | None = pre_transform
        self.post_transform: Transformer | None = post_transform

        self.criterion = None
        self.optimizer = None
        self.optimizer_info = None

    def ignore_persistence_args(self):
        return ['save_dir', 'device', 'training_echo', 'pre_transform', 'post_transform']

    def set_path(self, path):
        self.save_dir = Path(path)

    def set_pre_transform(self, pre_transform: Transformer):
        self.pre_transform = pre_transform

    def set_post_transform(self, post_transform: Transformer):
        self.post_transform = post_transform

    @abc.abstractmethod
    def init_model(self, model_definition):
        pass

    @abc.abstractmethod
    def train_on_batch(self, x: dict, y: dict):
        pass

    @abc.abstractmethod
    def train_on_generator(self, time_series_gen: TorchDataset, valid_time_series_gen: TorchDataset = None) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def predict_batch(self, x: tuple) -> tuple:
        pass

    @abc.abstractmethod
    def predict_numpy(self, *arr: tuple[np.ndarray]):
        pass

    def predict_ts(self, *ts: TimeSeries):
        assert self.pre_transform is not None
        return self.predict_numpy(*[np.expand_dims(self.pre_transform.process(i), 0) for i in ts])

    @abc.abstractmethod
    def eval_on_generator(self, time_series_gen: TorchDataset, evaluators: list, evaluation_params: dict,
                          use_tqdm=False) -> dict:
        pass

    @abc.abstractmethod
    def save(self, path, batch_id):
        pass

    @abc.abstractmethod
    def load(self, weights_path=None, epoch_id=None, batch_id=None):
        pass

    @abc.abstractmethod
    def set_optimizer(self, optimizer_info: dict):
        pass

    @abc.abstractmethod
    def set_criterion(self, criterion_info: dict):
        pass
