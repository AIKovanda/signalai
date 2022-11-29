import abc
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from taskchain.parameter import AutoParameterObject

from signalai.time_series import from_numpy, TimeSeries
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
    def train_on_generator(self, time_series_gen: TorchDataset) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def predict_batch(self, x: tuple) -> tuple:
        pass

    @abc.abstractmethod
    def predict_numpy(self, *arr: tuple[np.ndarray]):
        pass

    def predict_ts(self, *ts: TimeSeries):
        assert self.pre_transform is not None
        return self.predict_numpy(*[self.pre_transform.process(i) for i in ts])

    @abc.abstractmethod
    def eval_on_generator(self, time_series_gen: TorchDataset, evaluation_params: dict, post_transform: bool,
                          ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        pass

    @abc.abstractmethod
    def save(self, path, batch_id):
        pass

    @abc.abstractmethod
    def load(self, weights=None, epoch_id=None, batch_id=None):
        pass

    @abc.abstractmethod
    def set_optimizer(self, optimizer_info: dict):
        pass

    @abc.abstractmethod
    def set_criterion(self, criterion_info: dict):
        pass

    def _predict_one_timeseries(self, x: TimeSeries) -> TimeSeries:
        pass
        # assert isinstance(x, TimeSeries), 'Wrong type.'
        # assert x.data_arr is not None, 'Empty data_arr.'
        # new_data_arr = self.predict_numpy_batch(
        #     np.expand_dims(x.data_arr, 0)
        # )[0]
        # return type(x)(
        #     data_arr=new_data_arr,
        #     time_map=x.time_map,
        #     meta=x.meta,
        # )

    def predict_timeseries(self, ts: TimeSeries, target_length=None, residual_end=True):
        pass
        # transforms = self.pre_transform + self.transform.get('predict', [])
        #
        # if target_length is None or target_length >= len(ts):
        #     new_ts = apply_transforms(ts, transforms)
        #     new_data_arr = self.predict_numpy_batch(np.expand_dims(new_ts.data_arr, 0))[0]
        #     new_ts = from_numpy(data_arr=new_data_arr, meta=ts.meta, time_map=ts.time_map)
        #     return apply_transforms(new_ts, self.post_transform.get('predict', []))
        # else:
        #     length = original_length(target_length, transforms, fs=ts.fs)
        #     transformed_crops = []
        #     for i in range(len(ts) // length):
        #         new_ts = apply_transforms(ts.crop([i * length, (i + 1) * length]), transforms)
        #         transformed_crops.append(new_ts)
        #
        #     if len(ts) % length > 0 and residual_end:
        #         new_ts = apply_transforms(ts.crop([-(len(ts) % length), len(ts)]), transforms)
        #         transformed_crops.append(new_ts)
        #
        #     predicted_crops = map(self._predict_one_timeseries, transformed_crops)
        #
        #     result = [apply_transforms(_ts, self.post_transform.get('predict', [])) for _ts in predicted_crops]
        #     return MultiSeries(result).stack_series(axis=-1)

    def __call__(self, x, **kwargs):
        if isinstance(x, np.ndarray):
            ts = from_numpy(data_arr=x)
        elif isinstance(x, TimeSeries):
            ts = x
        else:
            raise TypeError(f"X cannot be of type '{type(x)}'.")

        return self.predict_timeseries(ts, **kwargs)
