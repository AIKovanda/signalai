import abc
import json
import re
from typing import Union, Optional, List, Iterable, Type, Tuple

import librosa
from librosa import resample
from librosa.display import specshow
from pydub.utils import mediainfo
from taskchain.parameter import AutoParameterObject
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import signal as scipy_signal
from signalai.config import LOGS_DIR, LOADING_THREADS
from signalai.tools.utils import join_dicts, time_now, original_length, apply_transforms, by_channel
from tqdm import trange, tqdm
import os
from pathlib import Path
import numpy as np
import pydub
from pydub import AudioSegment
from signalai.config import DTYPE_BYTES


class TimeSeries:
    """
    Stores either 1D signal or a 2D time-frequency transformation of a signal.
    First axis represents channel axis, the last one time axis.
    Operators:
    a + b - summing a and b
    a | b - joining channels of a and b
    a & b = concatenation of a and b
    """
    full_dimensions = None

    def __init__(self, data_arr: np.ndarray = None, meta=None, time_map=None, logger=None, fs: int = None):
        if meta is None:
            meta = {}

        self.logger = logger if logger is not None else Logger()

        if data_arr is not None and not isinstance(data_arr, np.ndarray):
            self.logger.log(f"Unknown signal type {type(data_arr)}.", priority=5)
            raise TypeError(f"Unknown signal type {type(data_arr)}.")

        if self.full_dimensions is not None:
            if len(data_arr.shape) == self.full_dimensions - 1:  # channel axis missing
                data_arr = np.expand_dims(data_arr, axis=0)

            if len(data_arr.shape) != self.full_dimensions:
                raise ValueError(f"Type {type(self)} must have {self.full_dimensions} dimensions, not "
                                 f"{len(data_arr.shape)}.")

        self._data_arr = data_arr

        if time_map is None:
            if self._data_arr is None:
                raise ValueError("Time map must be set when data_arr is None.")
            self._time_map = np.ones((self._data_arr.shape[0], self._data_arr.shape[-1]), dtype=bool)
        else:
            self._time_map = time_map.astype(bool)

        if len(self._time_map.shape) == 1:
            self._time_map = np.expand_dims(self._time_map, axis=0)

        if len(self._time_map.shape) > 2:
            raise ValueError(f"Data map must have one or two axes, not {len(self._time_map.shape)}.")

        self.meta = meta.copy()
        self.fs = fs

    def crop(self, interval=None):
        if interval is None:
            data_arr = self._data_arr
            time_map = self._time_map
        else:
            data_arr = self._data_arr[..., interval[0]:interval[1]]
            time_map = self._time_map[..., interval[0]:interval[1]]

        return type(self)(
            data_arr=data_arr,
            time_map=time_map,
            meta=self.meta,
            logger=self.logger,
            fs=self.fs,
        )

    @property
    def data_arr(self):
        return self._data_arr

    @property
    def time_map(self):
        return self._time_map

    def __len__(self):
        if self._data_arr is not None:
            return self._data_arr.shape[-1]

        return self._time_map.shape[-1]

    def astype_(self, dtype):
        self._data_arr = self._data_arr.astype(dtype)

    @property
    def channels_count(self):
        if self._data_arr is not None:
            return self._data_arr.shape[0]
        return self._time_map.shape[0]

    def take_channels(self, channels: Optional[List[Union[List[int], int]]] = None):
        if channels is None:
            return self

        data_arrays = []
        time_maps = []
        for channel_gen in channels:
            if isinstance(channel_gen, int):
                data_arrays.append(self._data_arr[[channel_gen], ...])
                time_maps.append(self._time_map[[channel_gen], ...])
            elif isinstance(channel_gen, list):
                data_arrays.append(np.sum(self._data_arr[channel_gen, ...], axis=0))
                time_maps.append(np.all(self._time_map[channel_gen, ...], axis=0))
            else:
                raise TypeError(f"Channel cannot be generated using type '{type(channel_gen)}'.")

        return type(self)(
            data_arr=np.vstack(data_arrays),
            time_map=np.vstack(time_maps),
            meta=self.meta,
            logger=self.logger,
            fs=self.fs,
        )

    def margin_interval(self, interval_length: int = None, start_id=0):
        if interval_length is None:
            interval_length = len(self)

        if interval_length == len(self) and (start_id == 0 or start_id is None):
            return self

        new_data_arr = np.zeros((*self._data_arr.shape[:-1], interval_length), dtype=self._data_arr.dtype)
        new_time_map = np.zeros((self._data_arr.shape[0], interval_length), dtype=bool)

        if 0 < len(self) + start_id and start_id < interval_length:
            new_data_arr[..., max(0, start_id):min(interval_length, start_id + len(self))] = \
                self._data_arr[..., max(0, -start_id):min(len(self), interval_length - start_id)]

            new_time_map[..., max(0, start_id):min(interval_length, start_id + len(self))] = \
                self._time_map[..., max(0, -start_id):min(len(self), interval_length - start_id)]

        return type(self)(
            data_arr=new_data_arr,
            meta=self.meta,
            time_map=new_time_map,
            logger=self.logger,
            fs=self.fs,
        )

    def __add__(self, other):
        if isinstance(other, type(self)):
            assert len(self) == len(other), "Adding signals with different lengths is forbidden (for a good reason)"
            new_data_arr = self._data_arr + other._data_arr
            new_info = join_dicts(self.meta, other.meta)
            new_time_map = self._time_map | other._time_map

        else:
            new_data_arr = self._data_arr + other
            new_info = self.meta
            new_time_map = self._time_map.copy()

        return type(self)(
            data_arr=new_data_arr,
            meta=new_info,
            time_map=new_time_map,
            logger=self.logger,
            fs=self.fs,
        )

    def __or__(self, other):
        if isinstance(other, type(self)):
            other_ts = other
            new_info = join_dicts(self.meta, other.meta)
        else:
            other_ts = type(self)(other)
            new_info = self.meta

        assert len(self) == len(other_ts), "Joining signals with different lengths is forbidden (for a good reason)"
        new_data_arr = np.concatenate([self._data_arr, other_ts._data_arr], axis=0)
        new_time_map = np.concatenate([self._time_map, other_ts._time_map], axis=0)

        return type(self)(
            data_arr=new_data_arr,
            meta=new_info,
            time_map=new_time_map,
            logger=self.logger,
            fs=self.fs,
        )

    def __mul__(self, other):
        return type(self)(
            data_arr=self._data_arr * other,
            meta=self.meta,
            time_map=self._time_map,
            logger=self.logger,
            fs=self.fs,
        )

    def __truediv__(self, other):
        return type(self)(
            data_arr=self._data_arr / other,
            meta=self.meta,
            time_map=self._time_map,
            logger=self.logger,
            fs=self.fs,
        )

    def __repr__(self):
        return str(pd.DataFrame.from_dict(
            self.meta | {'length': len(self), 'channels': self.channels_count, 'fs': self.fs},
            orient='index', columns=['value'],
        ))

    def update_meta(self, dict_):
        self.meta = self.meta.copy()
        self.meta.update(dict_)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.channels_count:
            sub_arr = self._data_arr[self.n: self.n+1]
            self.n += 1
            return type(self)(
                data_arr=sub_arr,
                meta=self.meta,
                time_map=self.time_map,
                logger=self.logger,
                fs=self.fs,
            )
        else:
            raise StopIteration


class Signal(TimeSeries):
    full_dimensions = 2

    def show(self, channels=None, figsize=(16, 3), save_as=None, show=True, split=False, title=None):
        if channels is None:
            channels = range(self.channels_count)
        elif isinstance(channels, int):
            channels = [channels]

        if split:
            if self.fs is None:
                with plt.style.context('seaborn-darkgrid'):
                    fig, axes = plt.subplots(self.channels_count, 1, figsize=figsize, squeeze=False)
                    for channel_id in channels:
                        y = self._data_arr[channel_id]
                        sns.lineplot(x=range(len(y)), y=y, ax=axes[channel_id, 0])
            else:
                fig, axes = plt.subplots(self.channels_count, 1, figsize=figsize, squeeze=False)
                for channel_id in channels:
                    f, t, Sxx = scipy_signal.spectrogram(self._data_arr[channel_id], self.fs)
                    axes[channel_id, 0].pcolormesh(t, f, Sxx, shading='gouraud')
        else:
            if self.fs is None:
                with plt.style.context('seaborn-darkgrid'):
                    fig = plt.figure(figsize=figsize)
                    for channel_id in channels:
                        if self.fs is None:
                            y = self._data_arr[channel_id]
                            sns.lineplot(x=range(len(y)), y=y)
            else:
                fig = plt.figure(figsize=figsize)
                f, t, Sxx = scipy_signal.spectrogram(self._data_arr[0], self.fs)
                plt.pcolormesh(t, f, Sxx, shading='gouraud')

        fig.patch.set_facecolor('white')
        if title is not None:
            fig.suptitle(title)

        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()

    def show_all(self, title=None):
        self.show(figsize=(18, 1.5 * self.channels_count), split=True, title=title)
        self.margin_interval(interval_length=150).show(
            figsize=(18, 1.5 * self.channels_count), split=True,
            title=f"{title} - first 150 samples")
        self.play()

    def spectrogram(self, figsize=(16, 9), save_as=None, show=True):
        plt.figure(figsize=figsize)
        f, t, Sxx = scipy_signal.spectrogram(self._data_arr[0], self.fs)
        Sxx = np.sqrt(Sxx)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()

    def spectrogram3(self, figsize=(16, 9), save_as=None, show=True):
        plt.figure(figsize=figsize)
        Sxx = np.abs(librosa.stft(self._data_arr[0, ::2], center=False, n_fft=2048, hop_length=1024))
        Sxx = Sxx / np.max(Sxx)
        print(Sxx.shape, np.max(Sxx), np.min(Sxx))
        plt.pcolormesh(Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()

    def spectrogram2(self):
        S = np.abs(librosa.stft(self._data_arr[0]))
        fig, ax = plt.subplots()

        img = specshow(librosa.amplitude_to_db(S, ref=np.max),
                       y_axis='log', x_axis='time', sr=self.fs, ax=ax)
        ax.set_title('Power spectrogram')
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    def play(self, channel_id=0, fs: int = None, volume=32):
        import sounddevice as sd
        fs = fs or self.fs
        sd.play(volume * self._data_arr[channel_id].astype('float32'), fs)
        sd.wait()

    def to_mp3(self, file, normalized=True):  # todo channels
        channels = 2 if (self._data_arr.ndim == 2 and self._data_arr.shape[0] == 2) else 1
        if normalized:  # normalized array - each item should be a float in [-1, 1)
            y = np.int16(self._data_arr.T * 2 ** 15)
        else:
            y = np.int16(self._data_arr.T)

        song = pydub.AudioSegment(y.tobytes(), frame_rate=self.fs, sample_width=2, channels=channels)
        song.export(file, format="mp3", bitrate="320k")


class Signal2D(TimeSeries):
    full_dimensions = 3

    def show(self):
        pass


class MultiSeries:
    def __init__(self,
                 series: Union[Iterable[TimeSeries], dict[str, TimeSeries]],
                 class_order: Optional[Iterable] = None,  # only usable if series is a dict
                 ):
        assert series is not None and len(series) > 0, "There is no input timeseries, MultiSeries does not make sense."
        if isinstance(series, dict):
            assert all([isinstance(val, TimeSeries) for val in series.values()]), [type(x) for x in series.values()]
        else:
            assert all([isinstance(val, TimeSeries) for val in series])

        self.series = series
        self.class_order = class_order

    def _series_list(self, only_valid=True) -> List[TimeSeries]:
        if isinstance(self.series, dict):
            if self.class_order is None:
                series = list(self.series.values())
            else:
                series = [self.series.get(i, None) for i in self.class_order]
        else:
            series = self.series

        if only_valid:
            series = [i for i in series if i is not None]

        return series

    def sum_channels(self, channels: Optional[List[Union[int, List[int]]]] = None):
        series = self._series_list(only_valid=True)
        series_length = len(series[0])
        assert all([len(ts) == series_length for ts in series]), 'Timeseries must have a same length to be joined.'
        assert len({ts.fs for ts in series}) == 1, 'Timeseries must have the same sampling frequency.'
        if channels is None:
            series_channels = series[0].channels_count
            assert all([ts.channels_count == series_channels for ts in series]), \
                'Timeseries must have the same number of channels to be joined when channels are not defined.'

        zero_series = series[0].take_channels(channels)
        data_arrays = [zero_series.data_arr]
        time_map = zero_series.time_map.copy()
        metas = [zero_series.meta]

        for ts in series[1:]:
            s_series = ts.take_channels(channels)
            data_arrays.append(s_series.data_arr)
            time_map = np.logical_or(time_map, s_series.time_map)
            metas.append(s_series.meta)

        assert series[0].fs is not None, series
        return type(series[0])(
            data_arr=np.sum(data_arrays, axis=0),
            time_map=time_map,
            meta=join_dicts(*metas),
            logger=series[0].logger,
            fs=series[0].fs,
        )

    def stack_series(self, only_valid=False, axis=0):
        series = self._series_list(only_valid=only_valid)

        series_shape = None
        ts_type = None
        logger = None
        time_map_shape = None

        for ts in series:
            if ts is not None:
                logger = ts.logger
                ts_type = type(ts)
                time_map_shape = ts.time_map.shape
                if ts.data_arr is not None:
                    series_shape = ts.data_arr.shape
                    break
        else:
            if time_map_shape is None:  # meaning there is no TimeSeries at all
                raise ValueError(f"At least one timeseries must not be empty while stacking timeseries.")

        if series_shape is not None:
            assert all([ts.data_arr.shape == series_shape for ts in series if ts is not None]), \
                'Timeseries must have the same shapes to be joined.'

        data_arrays = []
        time_maps = []
        fss = []
        metas = []

        for ts in series:
            if ts is not None:
                if series_shape is not None:
                    data_arrays.append(ts.data_arr)
                fss.append(ts.fs)
                time_maps.append(ts.time_map)
                metas.append(ts.meta)
            else:
                if series_shape is not None:
                    data_arrays.append(np.zeros(series_shape))
                time_maps.append(np.zeros(time_map_shape, dtype=bool))

        if series_shape is not None:
            data_arr = np.concatenate(data_arrays, axis=axis)
        else:
            data_arr = None

        assert len(set(fss)) == 1, fss  # todo: check if this works good
        return ts_type(
            data_arr=data_arr,
            time_map=np.concatenate(time_maps, axis=0),
            meta=join_dicts(*metas),
            logger=logger,
            fs=fss[0],
        )

    def apply_(self, func):
        if isinstance(self.series, dict):
            for key in self.series.keys():
                self.series[key] = func(self.series[key])
        else:
            self.series = list(map(func, self.series))

    def apply(self, func):
        if isinstance(self.series, dict):
            series = {}
            for key in self.series.keys():
                series[key] = func(self.series[key])
        else:
            series = [func(ts) for ts in self.series]

        return MultiSeries(series, class_order=self.class_order)


class SeriesClass:
    def __init__(self, series: List[TimeSeries] = None, series_build=None,
                 class_name: str = None, superclass_name: str = None, logger=None):
        self.series: List[TimeSeries] = series or []  # list
        self.series_build = series_build  # list
        self.class_name = class_name
        self.superclass_name = superclass_name
        self.logger = logger if logger is not None else Logger()
        self.fs = None

    def stat(self, to_json=False):
        info_dict = {
            'series': len(self.series),
            'class_name': self.class_name,
            'superclass_name': self.superclass_name,
            'series_length': [len(i) for i in self.series],
            'series_shape': [s.data_arr.shape for s in self.series],
        }
        if to_json:
            return json.dumps(info_dict, indent=2)

        return info_dict

    def load_to_ram(self) -> None:
        if not self.series:
            if LOADING_THREADS == 1:
                self.series = list(tqdm(map(build_series, self.series_build), total=len(self.series_build)))
            else:
                import multiprocessing as mp
                pool = mp.Pool(processes=LOADING_THREADS)
                self.series = list(tqdm(pool.imap(build_series, self.series_build), total=len(self.series_build)))
                pool.close()
                pool.terminate()
                pool.join()  # solving memory leaks
                self.logger.log(f"Signals loaded to RAM.", priority=1)

        fss = [ts.fs for ts in self.series]
        assert len(set(fss)) == 1, 'Signals in SeriesClass must have the same sample frequency!'
        self.fs = fss[0]  # todo: without loading to ram

    def get_index_map(self) -> List:
        return [len(i) for i in self.series]

    def get_individual_series(self, individual_id, start_id, length):
        if self.series is not None:
            return self.series[individual_id].crop(interval=(start_id, start_id + length))
        if self.series_build is None:
            self.logger.log(f"Signal cannot be built, there is no information how to do so.", raise_=ValueError)

        return build_series(self.series_build[individual_id], interval=(start_id, start_id + length))

    @property
    def total_length(self):
        return np.sum([len(i) for i in self.series])

    def __len__(self):
        return len(self.series)


class SeriesDataset(AutoParameterObject, abc.ABC):
    """
    split_range: tuple of two floats in range of [0,1]
    """

    def __init__(self, split_range: tuple = (0., 1.), **params):
        self.logger = None
        self.split_range = split_range
        self.params = params

    def set_logger(self, logger):
        self.logger = logger

    def set_split_range(self, split_range=None):
        if split_range is None:
            self.split_range = [0., 1.]
        else:
            self.split_range = split_range

    @abc.abstractmethod
    def get_class_objects(self) -> list[SeriesClass]:
        pass


class SeriesDatasetsKeeper:
    def __init__(self, datasets_config, logger, split_range=None):
        self.datasets_config = datasets_config
        self.split_range = split_range
        self.logger = logger

        self.classes_dict: dict[str, SeriesClass] = {}  # class_name: class_obj
        self.superclasses_dict = {}  # superclass_name: [class_name, ...]

        for dataset in datasets_config:
            dataset.set_logger(self.logger)
            dataset.set_split_range(self.split_range)
            for class_obj in dataset.get_class_objects():
                class_name = class_obj.class_name
                superclass_name = class_obj.superclass_name
                if class_name in self.classes_dict:
                    self.logger.log(f"Class '{class_name}' is defined multiple times. This is not allowed.",
                                    raise_=ValueError)

                self.classes_dict[class_name] = class_obj
                if superclass_name in self.superclasses_dict:
                    self.superclasses_dict[superclass_name].append(class_name)
                else:
                    self.superclasses_dict[superclass_name] = [class_name]

        self.total_lengths = {class_name: class_obj.total_length for class_name, class_obj in self.classes_dict.items()}
        self.fs = None

    def stat(self, to_json=False):
        info_dict = {
            'classes': np.sum([len(val) for key, val in self.superclasses_dict.items()]),
            'superclasses': self.superclasses_dict,
            'classes_stats': {key: val.stat() for key, val in self.classes_dict.items()},
            'total_lengths': self.total_lengths,
        }
        if to_json:
            return json.dumps(info_dict, indent=2)

        return info_dict

    def _check_valid_class(self, class_name):
        if class_name not in self.classes_dict:
            self.logger.log(f"Class '{class_name}' cannot be found, see the config.", raise_=ValueError)

    def load_to_ram(self, classes_name=None) -> None:
        if classes_name is None:
            classes_name = list(self.classes_dict.keys())
        for class_name in classes_name:
            self._check_valid_class(class_name)
            self.classes_dict[class_name].load_to_ram()

        fss = []
        for class_name, class_obj in self.classes_dict.items():
            fss.append(class_obj.fs)

        assert len(set(fss)) == 1, 'Datasets must have the same sample frequency!'
        assert fss[0] is not None
        self.fs = fss[0]

    def relevant_classes(self, superclasses: Iterable) -> list:
        return [j for i in superclasses for j in self.superclasses_dict[i]]

    def get_class(self, class_name):
        self._check_valid_class(class_name)
        return self.classes_dict[class_name]


class EndOfDataset(Exception):
    pass


class SeriesTaker:
    """
    strategy:   'random' - taking one random interval from all the relevant signals
                'sequence' - taking one interval is the logical order
                'only_once' - taking one interval is the logical order - error when reaching the end of the last signal
                'start' - taking starting interval of a random signal
                dict - more complex taking definition
    zero_padding: True - no error, makes zero padding if needed, False - error when length is higher than available
    """

    def __init__(self, keeper, class_name, strategy, logger=None, stride=0, zero_padding=True):
        self.keeper = keeper
        self.class_name = class_name
        self.strategy = strategy
        self.logger: Logger = logger or Logger()
        self.stride = stride
        self.zero_padding = zero_padding  # todo

        self.taken_class = self.keeper.get_class(self.class_name)
        self.index_map = self.taken_class.get_index_map()

        self.id_next = [0, 0]
        self.individual_now = 0

    def next(self, length) -> TimeSeries:
        index_map_clean = [max(0, j - length) for i, j in enumerate(self.index_map)]
        if self.strategy == 'random' or self.strategy == 'start':
            p = np.array(index_map_clean) / np.sum(index_map_clean)
            individual_id = np.random.choice(len(p), p=p)
            start_id = np.random.choice(index_map_clean[individual_id]) if self.strategy == 'random' else 0
            self.logger.log(f"Taking individual_id='{individual_id}', start_id='{start_id}' and "
                            f"length='{length}'.", priority=0)
            return self.taken_class.get_individual_series(
                individual_id=individual_id,
                start_id=start_id,
                length=length,
            )

        elif self.strategy == 'sequence':
            if self.id_next[1] <= index_map_clean[self.id_next[0]]:
                self._next_individual(length, index_map_clean)

            ts = self.taken_class.get_individual_series(
                individual_id=self.id_next[0],
                start_id=self.id_next[1],
                length=length,
            )
            self._next_individual(length, index_map_clean)
            return ts

        elif self.strategy == 'only_once':
            ts = self.taken_class.get_individual_series(
                individual_id=self.id_next[0],
                start_id=self.id_next[1],
                length=length,
            )
            self._next_individual(length, index_map_clean, no_beginning=True)
            return ts
        else:
            self.logger.log(f"Strategy '{self.strategy}' is not recognized.", raise_=ValueError)

    def _next_individual(self, length, index_map_clean, no_beginning=False):
        if self.id_next[1] + length <= index_map_clean[self.id_next[0]]:
            self.id_next[1] += length
        else:
            self.id_next[1] = 0
            while True:
                if self.id_next[0] == len(index_map_clean):
                    if no_beginning:
                        self.logger.log(f"Class '{self.class_name}' reached its maximum.", raise_=EndOfDataset)

                    self.id_next[0] = 0
                    self.logger.log(f"Class '{self.class_name}' reached its maximum, continuing from the beginning.", 3)
                else:
                    self.id_next[0] += 1
                if index_map_clean[self.id_next[0]] > 0:
                    break


class SeriesTrack(AutoParameterObject, abc.ABC):
    def __init__(self, transform: Optional[dict] = None, **params):
        if transform is None:
            transform = {}

        self.keeper = None
        self.superclasses = None
        self.relevant_classes = None
        self.logger = None

        self.params = params
        self.transform = transform
        self.takers = {}
        self.fs = None

    def set_logger(self, logger):
        self.logger: Logger = logger

    def set_keeper(self, keeper):
        self.keeper = keeper
        self.fs = keeper.fs
        self.relevant_classes: List[str] = list(keeper.relevant_classes(superclasses=self.params['superclasses']))
        for relevant_class in self.relevant_classes:
            self.takers[relevant_class] = SeriesTaker(
                keeper=keeper,
                class_name=relevant_class,
                strategy=self.params['strategy'],
                stride=self.params.get('stride', 0),
                logger=self.logger,
            )

    def _choose_class(self) -> str:
        if len(self.relevant_classes) == 0:
            raise ValueError(f"There is no class defined for track.")

        if len(self.relevant_classes) == 1:
            return self.relevant_classes[0]

        if self.params['equal_classes']:
            p = np.ones(len(self.relevant_classes))
        else:
            p = np.array([self.keeper.total_lengths[i] for i in self.relevant_classes])

        p = p / np.sum(p)
        return np.random.choice(self.relevant_classes, p=p)

    def next(self, length: int) -> Union[TimeSeries, MultiSeries]:
        pass


class SeriesProcessor:
    def __init__(self, processor_config, keeper, logger=None):
        self.processor_config = processor_config

        self.keeper = keeper
        self.logger: Logger = logger or Logger()

        self.fs_transform = []

        self.tracks = {}
        self.transform = self.processor_config.get('transform', [])

        self.X_re = self.processor_config['X']
        self.Y_re = self.processor_config['Y']

        self.x_original_length = {}
        self.y_original_length = {}

        for track_name, track_obj in self.processor_config['tracks'].items():
            self.tracks[track_name] = track_obj
            self.tracks[track_name].set_logger(self.logger)
            self.tracks[track_name].set_keeper(self.keeper)
            self.logger.log(f"Track '{track_name}' successfully initialized.", priority=2)

            self.X_re = re.sub(fr"(^|[\W])({track_name})($|[\W])", fr"\1track_buffer_x['\2']\3", self.X_re)
            self.Y_re = re.sub(fr"(^|[\W])({track_name})($|[\W])", fr"\1track_buffer_y['\2']\3", self.Y_re)

    def set_processing_fs(self, processing_fs: int):
        if processing_fs is not None:
            self.fs_transform = [Resampler(output_fs=processing_fs)]

    def next_one(self, target_length):
        track_buffer_x = {}
        track_buffer_y = {}

        for track_name, track_obj in self.tracks.items():
            all_track_transforms = self.fs_transform + track_obj.transform['base'] + track_obj.transform['X']
            track_buffer = track_obj.next(
                length=original_length(target_length, all_track_transforms + self.processor_config['transform']['X'],
                                       fs=track_obj.fs),
            )
            transformed_buffer = apply_transforms(
                track_buffer,
                self.fs_transform + track_obj.transform['base'])

            track_buffer_x[track_name] = apply_transforms(
                transformed_buffer, track_obj.transform['X'])
            track_buffer_y[track_name] = apply_transforms(
                transformed_buffer, track_obj.transform['Y'])

        x = apply_transforms(
            eval(self.X_re),
            self.processor_config['transform']['X'])
        y = apply_transforms(
            eval(self.Y_re),
            self.processor_config['transform']['Y'])
        return x, y

    def next_batch(self, target_length, batch_size=1) -> Tuple[np.ndarray, np.ndarray]:
        x_batch, y_batch = [], []
        for _ in range(batch_size):
            x, y = self.next_one(target_length)
            if isinstance(x, TimeSeries):
                x_batch.append(x.data_arr)
            else:
                x_batch.append(x)
            if isinstance(y, TimeSeries):
                y_batch.append(y.data_arr)
            else:
                y_batch.append(y)

        return np.array(x_batch), np.array(y_batch)

    def benchmark(self, target_length, batch_size=1, num=1000):
        for _ in trange(num):
            _ = self.next_batch(target_length, batch_size)

    def load_to_ram(self):
        self.keeper.load_to_ram()


class Logger:
    def __init__(self, file=None, name=None, verbose=0, save=True):
        if file is None:
            self.file = LOGS_DIR / f"{time_now(millisecond=False)} - {name}.log"
        else:
            self.file = file

        self.verbose = verbose
        self.save = save

    def log(self, message, priority=0, raise_: Optional[Type[Exception]] = None):
        if raise_ is not None:
            priority = 5

        space = "\t" * (5 - priority)
        message = f"{time_now(millisecond=True)} - {priority} {space}- {message}\n"
        if self.save:
            with open(self.file, "a") as f:
                f.write(message)

        if raise_ is not None:
            raise raise_(message)

        if self.verbose:
            print(message)


def pydub2numpy(audio: pydub.AudioSegment) -> (np.ndarray, int):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate


def audio_file2numpy(file) -> tuple[np.ndarray, int]:
    file = Path(file)
    suffix = file.suffix
    file = str(file.absolute())
    if suffix == '.wav':
        audio = AudioSegment.from_wav(file)
    elif suffix == '.mp3':
        audio = AudioSegment.from_mp3(file)
    elif suffix == '.aac':
        audio = AudioSegment.from_file(file, "aac")
    else:
        raise TypeError(f"Suffix '{suffix}' is not supported yet!")

    return pydub2numpy(audio)[0].T, int(mediainfo(file)['sample_rate'])


def read_audio(filename, file_sample_interval=None, interval=None, dtype=None):
    real_start, interval_length = _get_start_length(file_sample_interval, interval)
    data_arr, fs = audio_file2numpy(filename)
    if dtype is not None:
        data_arr = data_arr.astype(dtype)
    if not real_start:
        return Signal(data_arr=data_arr, fs=fs)
    if not file_sample_interval:
        return Signal(data_arr=data_arr[:, real_start:], fs=fs)

    return Signal(data_arr=data_arr[:, real_start: real_start + interval_length], fs=fs)


def read_bin(filename, file_sample_interval=None, interval=None, source_dtype='float32', dtype=None, meta=None,
             fs: int = None):
    real_start, interval_length = _get_start_length(file_sample_interval, interval)
    with open(filename, "rb") as f:
        start_byte = int(DTYPE_BYTES[source_dtype] * real_start)
        assert start_byte % DTYPE_BYTES[source_dtype] == 0, "Bytes are not loading properly."
        f.seek(start_byte, 0)
        data_arr = np.expand_dims(np.fromfile(f, dtype=source_dtype, count=interval_length or -1), axis=0)
        if dtype is not None:
            data_arr = data_arr.astype(dtype)
        return Signal(data_arr=data_arr, meta=meta, fs=fs)


def read_npy(filename, file_sample_interval=None, interval=None, dtype=None, fs: int = None):
    real_start, interval_length = _get_start_length(file_sample_interval, interval)
    full_data_arr = np.load(filename)
    if len(full_data_arr.shape) == 1:
        full_data_arr = np.expand_dims(full_data_arr, axis=0)
    data_arr = full_data_arr[:, real_start: real_start + interval_length]
    if dtype is not None:
        data_arr = data_arr.astype(dtype)

    if len(full_data_arr.shape) == 2:
        return Signal(data_arr=data_arr, fs=fs)
    elif len(full_data_arr.shape) == 3:
        return Signal2D(data_arr=data_arr, fs=fs)
    else:
        raise ValueError(f"Loaded array '{filename}' has {len(full_data_arr.shape)} channels, maximum is 3.")


def from_numpy(data_arr: np.ndarray, meta=None, time_map=None, logger=None, fs: int = None) -> TimeSeries:
    assert isinstance(data_arr, np.ndarray), f"data_arr must be a numpy.ndarray, not '{type(data_arr)}'"
    if len(data_arr.shape) in [1, 2]:
        return Signal(data_arr=data_arr, meta=meta, time_map=time_map, logger=logger, fs=fs)
    elif len(data_arr.shape) == 3:
        return Signal2D(data_arr=data_arr, fs=fs)
    else:
        raise ValueError(f"data_arr has unsupported dimension {len(data_arr.shape)}.")


def build_series(build_dict: dict, interval: Optional[Iterable[int]] = None) -> TimeSeries:
    if not build_dict:
        raise ValueError(f"There is no information of how to build a signal.")

    transform = build_dict.pop('transform', [])
    loaded_channels = []
    assert len(build_dict['files']) > 0, f"There is no file to be loaded."
    for file_dict in build_dict['files']:
        suffix = str(file_dict['filename'])[-4:]
        if suffix in [".aac", ".wav", ".mp3"]:
            loaded_channels.append(read_audio(interval=interval, **file_dict))
        if suffix in [".bin", ".dat"]:
            loaded_channels.append(read_bin(interval=interval, **file_dict))
        if suffix == ".npy":
            loaded_channels.append(read_npy(interval=interval, **file_dict))

    new_series = apply_transforms(MultiSeries(loaded_channels).stack_series(), transforms=transform)
    if build_dict.get('target_dtype'):
        new_series.astype_(build_dict['target_dtype'])
        new_series.update_meta(build_dict['meta'])

    return new_series


def signal_len(build_dict: dict) -> int:
    file_dict = build_dict['files'][0]
    if file_dict.get("file_sample_interval"):
        return int(file_dict["file_sample_interval"][1] - file_dict["file_sample_interval"][0])

    if str(file_dict['filename'])[-4:] in ['.bin', '.dat']:
        return int(os.path.getsize(file_dict['filename']) // DTYPE_BYTES[file_dict.get("dtype", "float32")])

    raise NotImplementedError  # todo: more options


def _get_start_length(file_sample_interval, interval) -> Tuple[int, int]:
    """
    Takes file_sample_interval and inner interval and returns tuple of real_start and interval_length.
    """
    real_start = 0
    interval_length = None
    if file_sample_interval is not None:
        real_start += file_sample_interval[0]
        interval_length = file_sample_interval[1] - file_sample_interval[0]
    if interval is not None:
        real_start += interval[0]
        if interval_length is not None:
            interval_length = min(interval_length, interval[1] - interval[0])
        else:
            interval_length = interval[1] - interval[0]

    if interval_length is not None:
        interval_length = int(interval_length)
    return int(real_start), interval_length


# todo: build_generator somehow


class Transformer(AutoParameterObject):
    in_dim = None

    def __init__(self, **params):
        self.params = params
        self.evaluated_params = {}
        for key, val in params.items():
            if isinstance(val, str):
                val = eval(val)
            self.evaluated_params[key] = val

    def _get_parameter_uniform(self, param_name: str, default: Union[float, List[float]]) -> float:
        param_value = self.evaluated_params.get(param_name, default)
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) == 1:
                return param_value[0]

            if len(param_value) == 2:
                return np.random.uniform(*param_value)

            raise ValueError(f"Parameter '{param_name}' must have one or two values, not {len(param_value)}.")

        return param_value

    def transform_timeseries(self, x: TimeSeries) -> Union[TimeSeries, np.ndarray]:
        pass

    def original_signal_length(self, length: int, fs: int = None) -> int:
        raise NotImplementedError("Original length is not implemented for this transformation.")

    @property
    def keeps_signal_length(self) -> bool:
        raise NotImplementedError("Keeps length is not implemented for this transformation.")

    def _transform(self, ts: TimeSeries):
        if self.in_dim is not None and self.in_dim != ts.full_dimensions:
            raise ValueError(f"Transform '{type(self)}' takes input of dim {self.in_dim}, "
                             f"{type(ts)} has a dim of {ts.full_dimensions}.")

        transform_chance = self.evaluated_params.get('transform_chance', 1.)
        if np.random.rand() < transform_chance:
            return self.transform_timeseries(ts)

        return ts

    def transform(self, x: Union[np.ndarray, TimeSeries, MultiSeries]) -> Union[TimeSeries, MultiSeries]:
        if isinstance(x, TimeSeries):
            ts = x
            return self._transform(ts)

        if isinstance(x, MultiSeries):
            return x.apply(self._transform)

        elif isinstance(x, np.ndarray):
            ts = from_numpy(x)
            return self._transform(ts)

        else:
            raise TypeError(
                f"Transformer got a type of '{type(x)}' which is not supported.")

    def __call__(self, x):
        return self.transform(x)


class Resampler(Transformer):
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, input_fs: int, output_fs: int) -> np.ndarray:
        return np.expand_dims(resample(
            x[0].astype('float32'), input_fs, output_fs, res_type='linear'
        ).astype(x.dtype), 0)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        input_fs = x.fs
        assert input_fs is not None, 'Missing info about sampling frequency!'
        output_fs = self.evaluated_params.get('output_fs')
        if input_fs == output_fs:
            return x

        data_arr = self.transform_npy(x.data_arr, input_fs=input_fs, output_fs=output_fs)
        return Signal(
            data_arr=data_arr,
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
            fs=output_fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return int(length * fs / self.evaluated_params.get('output_fs'))

    @property
    def keeps_signal_length(self) -> bool:
        return False


# todo: only load datasets that is needed
