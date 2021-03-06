import abc
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np
import torch

from signalai.config import DEVICE
from signalai.timeseries import (Logger, MultiSeries, Resampler, TimeSeries,
                                 from_numpy)
from signalai.tools.utils import apply_transforms, original_length


class SignalModel(abc.ABC):
    def __init__(self, model, save_dir, signal_model_type, target_signal_length: int, model_count=1,
                 processing_fs=None, output_type='label', logger=None, transform=None, post_transform=None):

        super().__init__()
        if post_transform is None:
            post_transform = {}
        if transform is None:
            transform = {}

        self.model = model
        self.save_dir = Path(save_dir) if save_dir is not None else None

        self.logger = logger or Logger(verbose=0, save=False)

        self.signal_model_type = signal_model_type
        if self.signal_model_type == 'torch_signal_model':
            import torch
            if DEVICE.startswith('cuda') and torch.cuda.is_available():
                self.model = self.model.to(DEVICE)

        self.processing_fs = processing_fs

        if self.processing_fs is not None:
            self.fs_transform = [Resampler(output_fs=self.processing_fs)]
        else:
            self.fs_transform = []

        self.target_signal_length = target_signal_length
        self.model_count = model_count
        self.output_type = output_type
        self.transform: dict = transform
        self.post_transform: dict = post_transform

        self.criterion = None
        self.optimizer = None
        self.optimizer_info = None

    def train_on_generator(self, series_processor, training_params: dict, models: list = None, **kwargs):
        self.logger.log("Starting training on generator.", priority=1)
        self.logger.log(f"Training params: {training_params}", priority=1)
        self._train_on_generator(series_processor, training_params, models, **kwargs)

    @abc.abstractmethod
    def _train_on_generator(self, series_processor, training_params: dict, models: list = None,
                            early_stopping_at=None, early_stopping_regression=None):
        pass

    @abc.abstractmethod
    def eval_on_generator(self, series_processor, evaluation_params: dict, post_transform: bool,
                          ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        pass

    @abc.abstractmethod
    def train_on_batch(self, x: np.ndarray, y: np.ndarray):
        pass

    @abc.abstractmethod
    def predict_batch(self, x):
        pass

    @abc.abstractmethod
    def predict_numpy_batch(self, x: np.ndarray) -> np.ndarray:
        pass

    def _predict_one_timeseries(self, x: TimeSeries) -> TimeSeries:
        new_data_arr = self.predict_numpy_batch(
            np.expand_dims(x.data_arr, 0)
        )[0]
        return type(x)(
            data_arr=new_data_arr,
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
        )

    def predict_timeseries(self, ts: TimeSeries, target_length=None, residual_end=True):
        transforms = self.fs_transform + self.transform.get('predict', [])

        if target_length is None or target_length >= len(ts):
            new_ts = apply_transforms(ts, transforms)
            new_data_arr = self.predict_numpy_batch(np.expand_dims(new_ts.data_arr, 0))[0]
            new_ts = from_numpy(data_arr=new_data_arr, meta=ts.meta, time_map=ts.time_map, logger=ts.logger)
            return apply_transforms(new_ts, self.post_transform.get('predict', []))
        else:
            length = original_length(target_length, transforms, fs=ts.fs)
            transformed_crops = []
            for i in range(len(ts) // length):
                new_ts = apply_transforms(ts.crop([i * length, (i + 1) * length]), transforms)
                transformed_crops.append(new_ts)

            if len(ts) % length > 0 and residual_end:
                new_ts = apply_transforms(ts.crop([-(len(ts) % length), len(ts)]), transforms)
                transformed_crops.append(new_ts)

            predicted_crops = map(self._predict_one_timeseries, transformed_crops)

            result = [apply_transforms(_ts, self.post_transform.get('predict', [])) for _ts in predicted_crops]
            return MultiSeries(result).stack_series(axis=-1)

    @abc.abstractmethod
    def save(self, path, batch_id):
        pass

    @abc.abstractmethod
    def load(self, path=None, batch=None):
        pass

    @abc.abstractmethod
    def set_optimizer(self, optimizer_info: dict):
        pass

    @abc.abstractmethod
    def set_criterion(self, criterion_info: dict):
        pass

    def __call__(self, x, target_length: Optional[int] = None, residual_end=False):
        if target_length is None:
            target_length = self.target_signal_length

        if isinstance(x, np.ndarray):
            ts = from_numpy(data_arr=x)

        elif isinstance(x, TimeSeries):
            ts = x

        else:
            raise TypeError(f"X cannot be of type '{type(x)}'.")

        return self.predict_timeseries(ts, target_length=target_length, residual_end=residual_end)
