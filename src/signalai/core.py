import abc
from pathlib import Path
from typing import Optional

import numpy as np
from signalai.config import DEVICE

from signalai.timeseries import Logger, from_numpy, TimeSeries, SeriesProcessor, Resampler, MultiSeries
from signalai.tools.utils import apply_transforms, original_length


class SignalModel(abc.ABC):
    def __init__(self, model, training_params, save_dir, signal_model_type, target_signal_length: int,
                 processing_fs=44100, output_type='label', logger=None, series_processor: SeriesProcessor = None,
                 evaluator=None, transform=None, post_transform=None):
        super().__init__()
        if post_transform is None:
            post_transform = {}
        if transform is None:
            transform = {}

        self.model = model
        self.training_params = training_params
        self.save_dir = Path(save_dir) if save_dir is not None else None

        self.logger = logger or Logger()
        self.series_processor = series_processor
        self.evaluator = evaluator

        self.signal_model_type = signal_model_type
        if self.signal_model_type == 'torch_signal_model':
            self.model = self.model.to(DEVICE)

        self.processing_fs = processing_fs or 44100  # todo: error

        if self.processing_fs is not None:
            self.fs_transform = [Resampler(output_fs=22050)]
        else:
            self.fs_transform = []

        self.target_signal_length = target_signal_length
        self.output_type = output_type

        self.transform: dict = transform
        self.post_transform: dict = post_transform

        self._criterion = None
        self._optimizer = None

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = self.get_optimizer()
        return self._optimizer

    @property
    def criterion(self):
        if self._criterion is None:
            self._criterion = self.get_criterion()
        return self._criterion

    def train_on_generator(self):
        self.logger.log("Starting training on generator.", priority=1)
        self.logger.log(f"Training params: {self.training_params}", priority=1)
        self._train_on_generator()

    @abc.abstractmethod
    def _train_on_generator(self):
        pass

    @abc.abstractmethod
    def train_on_batch(self, x: np.ndarray, y: np.ndarray):
        pass

    @abc.abstractmethod
    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        pass

    def _predict_one_timeseries(self, x: TimeSeries) -> TimeSeries:
        new_data_arr = self.predict_batch(
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
            new_data_arr = self.predict_batch(np.expand_dims(new_ts.data_arr, 0))[0]
            new_ts = from_numpy(data_arr=new_data_arr, meta=ts.meta, time_map=ts.time_map, logger=ts.logger)
            return apply_transforms(new_ts, self.post_transform.get('predict', []))
        else:
            length = original_length(target_length, transforms)
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
    def get_optimizer(self):
        pass

    @abc.abstractmethod
    def get_criterion(self):
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
