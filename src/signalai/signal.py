import abc
import re
from typing import Union, Optional, List, Iterable

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sounddevice as sd
from scipy import signal as scipy_signal
from signalai.config import LOGS_DIR, LOADING_PROCESSES
from signalai.tools.utils import join_dicts, time_now, set_union, timefunc
from tqdm import trange, tqdm
import os
from pathlib import Path
import numpy as np
import pydub
from pydub import AudioSegment
from signalai.config import DTYPE_BYTES
import multiprocessing as mp


class Signal:
    def __init__(self, signal_arr: np.ndarray, meta=None, signal_map=None, logger=None):
        if meta is None:
            meta = {}

        self.logger = logger if logger is not None else Logger()

        if not isinstance(signal_arr, np.ndarray):
            self.logger.log(f"Unknown signal type {type(signal_arr)}.", priority=5)
            raise TypeError(f"Unknown signal type {type(signal_arr)}.")

        self._signal_arr = signal_arr
        if signal_map is None:
            self._signal_map = np.ones_like(self._signal_arr).astype(bool)
        else:
            self._signal_map = signal_map

        self.meta = meta.copy()

    def crop(self, interval=None):
        if interval is None:
            return Signal(self.signal, meta=self.meta, signal_map=self.signal_map, logger=self.logger)
        new_signal_map = self._signal_map[:, interval[0]:interval[1]]
        signal_arr = self._signal_arr[:, interval[0]:interval[1]]
        return Signal(signal_arr=signal_arr, meta=self.meta, signal_map=new_signal_map, logger=self.logger)

    @property
    def signal(self):
        return self._signal_arr

    @property
    def signal_map(self):
        if self._signal_map is None:
            return None
        return self._signal_map

    def __len__(self):
        return self._signal_arr.shape[1]

    def take_channels(self, channels: Optional[List[Union[List[int], int]]] = None):
        if channels is None:
            return self

        signal_arrays = []
        signal_maps = []
        for channel_gen in channels:
            if isinstance(channel_gen, int):
                signal_arrays.append(self._signal_arr[[channel_gen], :])
                signal_maps.append(self._signal_map[[channel_gen], :])
            elif isinstance(channel_gen, list):
                signal_arrays.append(np.sum(self._signal_arr[channel_gen, :], axis=0))
                signal_maps.append(np.all(self._signal_map[channel_gen, :], axis=0))
            else:
                raise TypeError(f"Channel cannot by generated using type '{type(channel_gen)}'.")

        return Signal(
            signal_arr=np.concatenate(signal_arrays),
            signal_map=np.concatenate(signal_maps),
            meta=self.meta,
            logger=self.logger,
        )

    def show(self, channels=None, figsize=(16, 3), save_as=None, show=True, split=False, title=None,
             spectrogram_freq=None):
        if channels is None:
            channels = range(self.channels_count)
        elif isinstance(channels, int):
            channels = [channels]

        if split:
            if spectrogram_freq is None:
                with plt.style.context('seaborn-darkgrid'):
                    fig, axes = plt.subplots(self.channels_count, 1, figsize=figsize, squeeze=False)
                    for channel_id in channels:
                        y = self._signal_arr[channel_id]
                        sns.lineplot(x=range(len(y)), y=y, ax=axes[channel_id, 0])
            else:
                fig, axes = plt.subplots(self.channels_count, 1, figsize=figsize, squeeze=False)
                for channel_id in channels:
                    f, t, Sxx = scipy_signal.spectrogram(self._signal_arr[channel_id], spectrogram_freq)
                    axes[channel_id, 0].pcolormesh(t, f, Sxx, shading='gouraud')
        else:
            if spectrogram_freq is None:
                with plt.style.context('seaborn-darkgrid'):
                    fig = plt.figure(figsize=figsize)
                    for channel_id in channels:
                        if spectrogram_freq is None:
                            y = self._signal_arr[channel_id]
                            sns.lineplot(x=range(len(y)), y=y)
            else:
                fig = plt.figure(figsize=figsize)
                f, t, Sxx = scipy_signal.spectrogram(self._signal_arr[0], spectrogram_freq)
                plt.pcolormesh(t, f, Sxx, shading='gouraud')

        fig.patch.set_facecolor('white')
        if title is not None:
            fig.suptitle(title)

        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()

    def show_all(self, spectrogram_freq, title=None):
        self.show(figsize=(18, 1.5 * self.channels_count), split=True, title=title)
        self.show(figsize=(18, 1.5 * self.channels_count), split=True, title=title, spectrogram_freq=spectrogram_freq)

        self.margin_interval(interval_length=150).show(
            figsize=(18, 1.5 * self.channels_count), split=True,
            title=f"{title} - first 150 samples")
        self.play()

    def spectrogram(self, fs=44100, figsize=(16, 9), save_as=None, show=True):
        plt.figure(figsize=figsize)
        f, t, Sxx = scipy_signal.spectrogram(self.signal[0], fs)
        Sxx = np.sqrt(Sxx)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()

    @property
    def channels_count(self):
        return self.signal.shape[0]

    def play(self, channel_id=0, fs=44100, volume=32):
        sd.play(volume * self._signal_arr[channel_id].astype('float32'), fs)
        sd.wait()

    def margin_interval(self, interval_length=None, start_id=0):
        if interval_length is None:
            interval_length = len(self)

        if interval_length == len(self) and (start_id == 0 or start_id is None):
            return self

        signal = self._signal_arr

        new_signal = np.zeros((self._signal_arr.shape[0], interval_length), dtype=self._signal_arr.dtype)
        sig_len = signal.shape[1]

        new_signal[:, max(0, start_id):min(interval_length, start_id + sig_len)] = \
            signal[:, max(0, -start_id):min(sig_len, interval_length - start_id)]

        signal_map = self._signal_map
        new_signal_map = np.zeros((self._signal_arr.shape[0], interval_length), dtype=bool)
        new_signal_map[:, max(0, start_id):min(interval_length, start_id + sig_len)] = \
            signal_map[:, max(0, -start_id):min(sig_len, interval_length - start_id)]

        return Signal(signal_arr=new_signal, meta=self.meta, signal_map=new_signal_map, logger=self.logger)

    def __add__(self, other):
        if not isinstance(other, Signal):
            return Signal(signal_arr=self._signal_arr + other, meta=self.meta, signal_map=self._signal_map, logger=self.logger)

        assert len(self) == len(other), "Adding signals with different lengths is forbidden (for a good reason)"
        new_signal = self._signal_arr + other._signal_arr
        new_info = join_dicts(self.meta, other.meta)

        return Signal(signal_arr=new_signal, meta=new_info, signal_map=(self._signal_map | other._signal_map), logger=self.logger)

    def __or__(self, other):
        if isinstance(other, Signal):
            other_signal = other
            new_info = join_dicts(self.meta, other.meta)
            new_signal_map = (self._signal_map | other._signal_map)
        else:
            other_signal = Signal(other)
            new_info = self.meta
            new_signal_map = self._signal_map

        assert len(self) == len(other), "Adding signals with different lengths is forbidden (for a good reason)"
        new_signal = np.concatenate([self.signal, other_signal.signal], axis=0)

        return Signal(signal_arr=new_signal, meta=new_info, signal_map=new_signal_map, logger=self.logger)

    def __mul__(self, other):
        return Signal(signal_arr=self._signal_arr * other, meta=self.meta, signal_map=self._signal_map, logger=self.logger)

    def __truediv__(self, other):
        return Signal(signal_arr=self._signal_arr / other, meta=self.meta, signal_map=self._signal_map, logger=self.logger)

    def __repr__(self):
        return str(pd.DataFrame.from_dict(self.meta, orient='index'))

    def update_meta(self, dict_):
        self.meta.update(dict_)

    def to_mp3(self, file, sf=44100, normalized=True):  # todo channels
        channels = 2 if (self.signal.ndim == 2 and self.signal.shape[0] == 2) else 1
        if normalized:  # normalized array - each item should be a float in [-1, 1)
            y = np.int16(self.signal.T * 2 ** 15)
        else:
            y = np.int16(self.signal.T)
        song = pydub.AudioSegment(y.tobytes(), frame_rate=sf, sample_width=2, channels=channels)
        song.export(file, format="mp3", bitrate="320k")


class MultiSignal:
    def __init__(self, signals: Union[list, dict, tuple], class_order: Optional[Iterable] = None):
        assert signals is not None and len(signals) > 0, "There is no input signal, MultiSignal does not make sense."
        self.signals = signals  # can be None
        self.class_order = class_order

    def _signal_list(self, only_valid=True) -> List[Signal]:
        if isinstance(self.signals, dict):
            if self.class_order is None:
                signals = list(self.signals.values())
            else:
                signals = [self.signals.get(i, None) for i in self.class_order]
        else:
            signals = self.signals

        if only_valid:
            signals = [i for i in signals if i is not None]

        return signals

    def sum_channels(self, channels: Optional[List[Union[int, List[int]]]] = None):
        signals = self._signal_list(only_valid=True)
        signal_length = len(signals[0])
        assert all([len(s) == signal_length for s in signals]), 'Signals must have a same length to be joined.'
        if channels is None:
            signal_channels = signals[0].channels_count
            assert all([s.channels_count == signal_channels for s in signals]), \
                'Signals must have the same number of channels to be joined when channels are not defined.'

        zero_signal = signals[0].take_channels(channels)
        signal_arr = zero_signal.signal.copy()
        signal_map = zero_signal.signal_map.copy()
        signal_meta = zero_signal.meta.copy()

        for s in signals[1:]:
            s_signal = s.take_channels(channels)
            signal_arr += s_signal.signal
            signal_map = np.logical_and(signal_map, s_signal.signal_map)
            signal_meta = join_dicts(signal_meta, s_signal.meta)

        return Signal(signal_arr=signal_arr, signal_map=signal_map, meta=signal_meta)

    def stack_signals(self, only_valid=False):
        signals = self._signal_list(only_valid=only_valid)
        for signal in signals:
            if signal is not None:
                signal_shape = signal.signal.shape
                break
        else:
            raise ValueError(f"At least one signal must not be empty while stacking signals.")

        assert all([s.signal.shape == signal_shape for s in signals if s is not None]), \
            'Signals must have the same shapes to be joined.'
        signal_arrays = []
        signal_maps = []
        signal_metas = []

        for signal in signals:
            if signal is not None:
                signal_arrays.append(signal.signal)
                signal_maps.append(signal.signal_map)
                signal_metas.append(signal.meta)
            else:
                signal_arrays.append(np.zeros(signal_shape))
                signal_maps.append(np.zeros(signal_shape, dtype=bool))

        return Signal(
            signal_arr=np.concatenate(signal_arrays, axis=0),
            signal_map=np.concatenate(signal_maps, axis=0),
            meta=join_dicts(*signal_metas),
        )


class SignalClass:
    def __init__(self, signals=None, signals_build=None, class_name=None, superclass_name=None, logger=None):
        self.signals = signals or []  # list
        self.signals_build = signals_build  # list
        self.class_name = class_name
        self.superclass_name = superclass_name
        self.logger = logger if logger is not None else Logger()

    def load_to_ram(self) -> None:
        if not self.signals:
            pool = mp.Pool(processes=LOADING_PROCESSES)
            self.signals = list(tqdm(pool.imap(build_signal, self.signals_build), total=len(self.signals_build)))
            self.logger.log(f"Signals loaded to RAM.", priority=1)

    def get_index_map(self) -> list:
        return [len(i) for i in self.signals]

    def get_signal(self, individual_id, start_id, length):
        if self.signals is not None:
            return self.signals[individual_id].crop(interval=(start_id, start_id + length))
        assert self.signals_build is not None, f"Signal cannot be built, there is no information how to do so."
        return build_signal(self.signals_build[individual_id], interval=(start_id, start_id + length))

    @property
    def total_length(self):
        return np.sum([len(i) for i in self.signals])

    def __len__(self):
        return len(self.signals)


class SignalDataset(abc.ABC):
    """
    split_range: tuple of two floats in range of [0,1]
    """

    def __init__(self, split_range: tuple = (0., 1.), **params):
        self.logger = None
        self.split_range = split_range
        self.params = params

    def set_logger(self, logger):
        self.logger = logger

    def set_split_range(self, split_range):
        self.split_range = split_range

    @abc.abstractmethod
    def get_class_objects(self):
        pass


class SignalDatasetsKeeper:
    def __init__(self, datasets_config, split_range, logger):
        self.datasets_config = datasets_config
        self.split_range = split_range
        self.logger = logger

        self.classes_dict = {}  # class_name: class_obj
        self.superclasses_dict = {}  # superclass_name: [class_name, ...]

        for dataset in datasets_config:
            dataset.set_logger(self.logger)
            dataset.set_split_range(self.split_range)
            for class_obj in dataset.get_class_objects():
                class_name = class_obj.class_name
                superclass_name = class_obj.superclass_name
                if class_name in self.classes_dict:
                    self.logger.log(f"Class '{class_name}' is defined multiple times. This is not allowed.", 5)
                    raise ValueError(f"Class '{class_name}' is defined multiple times. This is not allowed.")  # todo: raise
                self.classes_dict[class_name] = class_obj
                if superclass_name in self.superclasses_dict:
                    self.superclasses_dict[superclass_name].append(class_name)
                else:
                    self.superclasses_dict[superclass_name] = [class_name]

        self.total_lengths = {class_name: class_obj.total_length for class_name, class_obj in self.classes_dict.items()}

    def _check_valid_class(self, class_name):
        if class_name not in self.classes_dict:
            self.logger.log(f"Class '{class_name}' cannot be found, see the config.", priority=5)
            raise ValueError(f"Class '{class_name}' cannot be found, see the config.")

    def load_to_ram(self, classes_name=None) -> None:
        if classes_name is None:
            classes_name = list(self.classes_dict.keys())
        for class_name in classes_name:
            self._check_valid_class(class_name)
            self.classes_dict[class_name].load_to_ram()

    def relevant_classes(self, superclasses=None):
        if superclasses is not None:
            return set_union(*[set(self.superclasses_dict[i]) for i in superclasses])

    def get_class(self, class_name):
        self._check_valid_class(class_name)
        return self.classes_dict[class_name]


class EndOfDataset(Exception):
    pass


class SignalTaker:
    """
    strategy:   'random' - taking one random interval from all the relevant signals
                'sequence' - taking one interval is the logical order
                'only_once' - taking one interval is the logical order - error when reaching the end of the last signal
                'start' - taking starting interval of a random signal
                dict - more complex taking definition
    zero_padding: True - no error, makes zero padding if needed, False - error when length is higher than available
    """

    def __init__(self, keeper, class_name, strategy, logger, stride=0, zero_padding=True):
        self.keeper = keeper
        self.class_name = class_name
        self.strategy = strategy
        self.logger = logger
        self.stride = stride
        self.zero_padding = zero_padding  # todo

        self.taken_class = self.keeper.get_class(self.class_name)
        self.index_map = self.taken_class.get_index_map()

        self.id_next = [0, 0]
        self.individual_now = 0

    def next(self, length) -> Signal:
        index_map_clean = [max(0, j - length) for i, j in enumerate(self.index_map)]
        if self.strategy == 'random' or self.strategy == 'start':
            p = np.array(index_map_clean) / np.sum(index_map_clean)
            individual_id = np.random.choice(len(p), p=p)
            start_id = np.random.choice(index_map_clean[individual_id]) if self.strategy == 'random' else 0
            self.logger.log(f"Taking '{individual_id=}', '{start_id=}' and '{length=}'.", priority=0)
            return self.taken_class.get_signal(
                individual_id=individual_id,
                start_id=start_id,
                length=length
            )

        elif self.strategy == 'sequence':
            if self.id_next[1] <= index_map_clean[self.id_next[0]]:
                self._next_individual(length, index_map_clean)

            signal = self.taken_class.get_signal(individual_id=self.id_next[0], start_id=self.id_next[1], length=length)
            self._next_individual(length, index_map_clean)
            return signal

        elif self.strategy == 'only_once':
            signal = self.taken_class.get_signal(individual_id=self.id_next[0], start_id=self.id_next[1], length=length)
            self._next_individual(length, index_map_clean, no_beginning=True)
            return signal
        else:
            self.logger.log(f"Strategy '{self.strategy}' is not recognized.", priority=5)
            raise ValueError(f"Strategy '{self.strategy}' is not recognized.")

    def _next_individual(self, length, index_map_clean, no_beginning=False):
        if self.id_next[1] + length <= index_map_clean[self.id_next[0]]:
            self.id_next[1] += length
        else:
            self.id_next[1] = 0
            while True:
                if self.id_next[0] == len(index_map_clean):
                    if no_beginning:
                        raise EndOfDataset(f"Class '{self.class_name}' reached its maximum.")

                    self.id_next[0] = 0
                    self.logger.log(f"Class '{self.class_name}' reached its maximum, continuing from the beginning.", 3)
                else:
                    self.id_next[0] += 1
                if index_map_clean[self.id_next[0]] > 0:
                    break


class SignalTrack:
    def __init__(self, name, keeper, superclasses, logger,
                 equal_classes=False, strategy: Union[str, dict] = 'random', stride=0, transforms=None):
        if transforms is None:
            transforms = {}

        self.name = name
        self.keeper = keeper
        self.superclasses = superclasses
        self.logger = logger

        self.equal_classes = equal_classes
        self.strategy = strategy
        self.stride = stride
        self.transforms = transforms

        self.takers = {}
        self.relevant_classes = list(keeper.relevant_classes(superclasses=superclasses))
        for relevant_class in self.relevant_classes:
            if isinstance(self.strategy, dict):
                taker_strategy = self.strategy.get('inner_strategy', 'start')
            else:
                taker_strategy = self.strategy
            self.takers[relevant_class] = SignalTaker(
                keeper=keeper, class_name=relevant_class, strategy=taker_strategy, stride=self.stride, logger=self.logger
            )
            self.logger.log(f"Taker of '{relevant_class}' successfully initialized.", priority=2)

    def _choose_class(self):
        if len(self.relevant_classes) == 0:
            self.logger.log(f"There is no class defined for track '{self.name}'.", priority=5)
            raise ValueError(f"There is no class defined for this track'{self.name}'.")

        if len(self.relevant_classes) == 1:
            return self.relevant_classes[0]

        if self.equal_classes:
            p = np.ones(len(self.relevant_classes))
        else:
            p = np.array([self.keeper.total_lengths[i] for i in self.relevant_classes])

        p = p / np.sum(p)
        return np.random.choice(list(self.relevant_classes), p=p)

    def next(self, length: int) -> Union[Signal, MultiSignal]:
        if isinstance(self.strategy, str):
            chosen_class = self._choose_class()
            return self.takers[chosen_class].next(length=length)
        if isinstance(self.strategy, dict):
            return self._next_compose(length)
        raise TypeError(f"Strategy type of '{type(self.strategy)}' is not supported.")

    def _next_compose(self, length: int) -> MultiSignal:
        count_min, count_max = self.strategy.get('tones_count_range', (0, 10))
        tones_count = np.random.choice(range(count_min, count_max + 1))
        chosen_classes = [np.random.choice(self.relevant_classes) for _ in range(tones_count)]

        possible_starting_index = np.arange(*self.strategy.get('start_arange', [1]))
        possible_tone_length = np.arange(*self.strategy.get('tone_length_arange', [length]))

        signals = {}
        for class_name in self.relevant_classes:
            class_count = chosen_classes.count(class_name)
            if class_count == 0:
                continue

            final_intervals = []

            for i in range(class_count):
                starting_index = np.random.choice(possible_starting_index)
                tone_length = np.random.choice(possible_tone_length)
                ending_index = starting_index + tone_length
                for j, interval in enumerate(final_intervals):
                    if interval[0] < starting_index < interval[1]:
                        break
                    if interval[0] < ending_index < interval[1]:
                        break
                else:
                    final_intervals.append([starting_index, ending_index])

            assert len(final_intervals) > 0, "Something is wrong with chosen tone intervals."
            signals[class_name] = MultiSignal(
                signals=[
                    self.takers[class_name].next(interval[1] - interval[0]).margin_interval(
                        interval_length=length,
                        start_id=interval[0],
                    ) for interval in final_intervals]
            ).sum_channels()

        return MultiSignal(signals=signals, class_order=self.relevant_classes)


class SignalProcessor:
    def __init__(self, processor_config, keeper, logger):
        self.processor_config = processor_config
        self.to_ram = self.processor_config.get('to_ram', False)
        self.op_type = self.processor_config.get('op_type', 'float32')

        self.logger = logger
        self.keeper = keeper

        self.tracks = {}
        self.transforms = self.processor_config.get('transforms', [])

        self.X_re = self.processor_config['X']
        self.Y_re = self.processor_config['Y']

        self.x_original_length = {}
        self.y_original_length = {}

        for track_name, track_config in self.processor_config['tracks'].items():
            self.tracks[track_name] = SignalTrack(
                name=track_name,
                keeper=self.keeper,
                logger=self.logger,
                **track_config)
            self.logger.log(f"Track '{track_name}' successfully initialized.", priority=2)

            self.X_re = re.sub(fr"(^|[\W])({track_name})($|[\W])", fr"\1track_buffer_x['\2']\3", self.X_re)
            self.Y_re = re.sub(fr"(^|[\W])({track_name})($|[\W])", fr"\1track_buffer_y['\2']\3", self.Y_re)

            x_transform = (track_config['transforms']['base'] + track_config['transforms']['X'] +
                           self.processor_config['transforms']['base'] + self.processor_config['transforms']['X'])
            y_transform = (track_config['transforms']['base'] + track_config['transforms']['Y'] +
                           self.processor_config['transforms']['base'] + self.processor_config['transforms']['Y'])

            self.x_original_length[track_name] = original_length(self.processor_config['target_length'], x_transform)
            self.y_original_length[track_name] = original_length(self.processor_config['target_length'], y_transform)
            if self.x_original_length[track_name] != self.x_original_length[track_name]:
                message = (f"Transform makes X and Y original lengths different.\n"
                           f"X: {self.x_original_length[track_name]}, Y: {self.x_original_length[track_name]},"
                           f"at track '{track_name}'.")
                self.logger.log(message, priority=5)
                raise ValueError(message)

    def next_one(self):
        track_buffer_x = {}
        track_buffer_y = {}
        for track_name, track_obj in self.tracks.items():
            track_buffer = track_obj.next(length=self.x_original_length[track_name])
            track_buffer_x[track_name] = apply_transforms(
                track_buffer,
                track_obj.transforms['base'] + track_obj.transforms['X'])
            track_buffer_y[track_name] = apply_transforms(
                track_buffer,
                track_obj.transforms['base'] + track_obj.transforms['Y'])

        x = apply_transforms(
            eval(self.X_re),
            self.processor_config['transforms']['base'] + self.processor_config['transforms']['X'])
        y = apply_transforms(
            eval(self.Y_re),
            self.processor_config['transforms']['base'] + self.processor_config['transforms']['Y'])

        return x, y

    def next_batch(self, batch_size=1):
        x_batch, y_batch = [], []
        for _ in range(batch_size):
            x, y = self.next_one()
            if isinstance(x, Signal):
                x_batch.append(x.signal)
            else:
                x_batch.append(x)
            if isinstance(y, Signal):
                y_batch.append(y.signal)
            else:
                y_batch.append(y)

        return np.array(x_batch), np.array(y_batch)

    def benchmark(self, batch_size=1, num=1000):
        for _ in trange(num):
            _ = self.next_batch(batch_size)

    def load_to_ram(self):
        self.keeper.load_to_ram()


class Logger:
    def __init__(self, file=None, name=None, verbose=0):
        if file is None:
            self.file = LOGS_DIR / f"{time_now(millisecond=False)} - {name}.log"
        else:
            self.file = file

        self.verbose = verbose

    def log(self, message, priority=0):
        space = "\t" * (5 - priority)
        message = f"{time_now(millisecond=True)} - {priority} {space}- {message}\n"
        with open(self.file, "a") as f:
            f.write(message)

        if self.verbose:
            print(message)


def apply_transforms(s, transforms=()):
    if len(transforms) == 0:
        return s
    for transform in transforms:
        s = transform(s)
    return s


def original_length(target_length, transforms=()):
    if len(transforms) == 0:
        return target_length
    for transform in transforms[::-1]:
        target_length = transform.original_length(target_length)
    assert target_length is not None and target_length > 0, "Output of chosen transformations does not make sense."
    return target_length


def pydub2numpy(audio: pydub.AudioSegment) -> (np.ndarray, int):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate


def audio_file2numpy(file):
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

    return pydub2numpy(audio)[0].T


def read_audio(filename, file_sample_interval=None, interval=None, dtype=None):
    real_start, interval_length = get_interval_values(file_sample_interval, interval)
    signal_arr = audio_file2numpy(filename)
    if dtype is not None:
        signal_arr = signal_arr.astype(dtype)
    if not real_start:
        return Signal(signal_arr=signal_arr)
    if not file_sample_interval:
        return Signal(signal_arr=signal_arr[:, real_start:])
    return Signal(signal_arr=signal_arr[:, real_start: real_start + interval_length])


def read_bin(filename, file_sample_interval=None, interval=None, source_dtype='float32', dtype=None):
    real_start, interval_length = get_interval_values(file_sample_interval, interval)
    with open(filename, "rb") as f:
        start_byte = int(DTYPE_BYTES[source_dtype] * real_start)
        assert start_byte % DTYPE_BYTES[source_dtype] == 0, "Bytes are not loading properly."
        f.seek(start_byte, 0)
        signal_arr = np.expand_dims(np.fromfile(f, dtype=source_dtype, count=interval_length or -1), axis=0)
        if dtype is not None:
            signal_arr = signal_arr.astype(dtype)
        return Signal(signal_arr=signal_arr)


def read_npy(filename, file_sample_interval=None, interval=None, dtype=None):
    real_start, interval_length = get_interval_values(file_sample_interval, interval)
    signal = np.load(filename)
    if len(signal.shape) == 1:
        signal = np.expand_dims(signal, axis=0)
    signal_arr = signal[:, real_start: real_start + interval_length]
    if dtype is not None:
        signal_arr = signal_arr.astype(dtype)
    return Signal(signal_arr=signal_arr)


def build_signal(build_dict, interval=None):
    if not build_dict:
        raise ValueError(f"There is no information of how to build a signal.")

    transforms = build_dict.pop('transforms', [])
    loaded_channels = []
    assert len(build_dict['files']) > 0, f"There is no file to be loaded."
    for file_dict in build_dict['files']:
        suffix = str(file_dict['filename'])[-4:]
        if suffix in [".aac", ".wav", ".mp3"]:
            loaded_channels.append(read_audio(interval=interval, **file_dict).signal)
        if suffix in [".bin", ".dat"]:
            loaded_channels.append(read_bin(interval=interval, **file_dict).signal)
        if suffix == ".npy":
            loaded_channels.append(read_npy(interval=interval, **file_dict).signal)

    new_signal = apply_transforms(np.concatenate(loaded_channels, axis=0), transforms=transforms)
    if build_dict.get('target_dtype'):
        new_signal = new_signal.astype(build_dict['target_dtype'])

    return Signal(signal_arr=new_signal, meta=build_dict['meta'])


def signal_len(build_dict):
    file_dict = build_dict['files'][0]
    if file_dict.get("file_sample_interval"):
        return int(file_dict["file_sample_interval"][1] - file_dict["file_sample_interval"][0])

    if str(file_dict['filename'])[-4:] in ['.bin', '.dat']:
        return int(os.path.getsize(file_dict['filename']) // DTYPE_BYTES[file_dict.get("dtype", "float32")])

    raise NotImplementedError  # todo: more options


def get_interval_values(file_sample_interval, interval):
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
