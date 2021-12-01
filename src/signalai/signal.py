import abc
import os
import re
import numpy as np
import pandas as pd
import pydub
import seaborn as sns
from matplotlib import pyplot as plt
import simpleaudio as sa
from scipy import signal as scipy_signal
from signalai.config import LOGS_DIR, DTYPE_BYTES
from signalai.tools.signal_loading import from_audio, from_bin, from_npy
from signalai.tools.utils import join_dicts, time_now, set_union
from tqdm import trange, tqdm


class Signal:
    def __init__(self, build_from, meta=None, signal_map=None, logger=None):
        if meta is None:
            meta = {}

        self.logger = logger if logger is not None else Logger()
        self.signal_arr = None
        self.signal_map = signal_map
        self.build_dict = None
        self.meta = meta.copy()

        if isinstance(build_from, np.ndarray):
            self.signal_arr = build_from.copy()
            if signal_map is not None:
                if signal_map.shape != self.signal_arr.shape:
                    message = (f"Signal map has a shape of {signal_map.shape} while signal array has "
                               f"a shape of {self.signal_arr.shape}. These shapes must be the same.")
                    self.logger.log(message, priority=5)
                    raise ValueError(message)

        elif isinstance(build_from, Signal):
            if build_from.signal_arr is None:
                raise NotImplementedError(f"Object of class 'Signal' cannot be created from unloaded signal object.")  # todo
            self.signal_arr = build_from.signal_arr.copy()
            self.signal_map = signal_map or build_from.signal_map
            self.meta = meta.update(build_from.meta)

        elif isinstance(build_from, dict):
            self.build_dict = build_from.copy()

        else:
            self.logger.log(f"Unknown signal type {type(build_from)}.", priority=5)
            raise TypeError(f"Unknown signal type {type(build_from)}.")

    @property
    def signal(self):
        if self.signal_arr is not None:
            return self.signal_arr
        return self.get_signal().signal

    def transform(self):
        pass  # todo save transformed etc.

    def get_signal(self, interval=None):
        if self.signal_arr is not None:
            if interval is None:
                return Signal(self.signal_arr, meta=self.meta, signal_map=self.signal_map)
            
            new_signal_map = None if self.signal_map is None else self.signal_map[:, interval[0]:interval[1]]
            return Signal(build_from=self.signal_arr[:, interval[0]:interval[1]], 
                          meta=self.meta, signal_map=new_signal_map)

        if self.build_dict:
            transforms = self.build_dict.pop('transforms', [])
            loaded_channels = []
            for file_dict in self.build_dict['files']:
                suffix = str(file_dict['filename'])[-4:]
                if suffix in [".aac", ".wav", ".mp3"]:
                    loaded_channels.append(from_audio(interval=interval, **file_dict))
                if suffix in [".bin", ".dat"]:
                    loaded_channels.append(from_bin(interval=interval, **file_dict))
                if suffix == ".npy":
                    loaded_channels.append(from_npy(interval=interval, **file_dict))

            new_signal = apply_transforms(np.concatenate(loaded_channels, axis=0), transforms=transforms)
            if self.build_dict.get('target_dtype'):
                new_signal = new_signal.astype(self.build_dict['target_dtype'])

            return Signal(build_from=new_signal, meta=self.meta)

        raise ValueError(f"There is no information of how to build a signal.")

    def __len__(self):
        if self.signal_arr is not None:
            return self.signal_arr.shape[1]

        file_dict = self.build_dict['files'][0]
        if file_dict.get("file_sample_interval"):
            return int(file_dict["file_sample_interval"][1] - file_dict["file_sample_interval"][0])

        if str(file_dict['filename'])[-4:] in ['.bin', '.dat']:
            return int(os.path.getsize(file_dict['filename']) // DTYPE_BYTES[file_dict.get("dtype", "float32")])

        return self.get_signal().signal.shape[1]

    def load_to_ram(self):
        if self.signal_arr is None:
            self.signal_arr = self.signal
            self.logger.log(f"Signal loaded to RAM.", priority=1)

    def show(self, channels=None, figsize=(16, 3), save_as=None, show=True, split=False, title=None,
             spectrogram_freq=None):
        if channels is None:
            channels = range(self.signal.shape[0])
        elif isinstance(channels, int):
            channels = [channels]

        if split:
            if spectrogram_freq is None:
                with plt.style.context('seaborn-darkgrid'):
                    fig, axes = plt.subplots(self.channels, 1, figsize=figsize, squeeze=False)
                    for channel_id in channels:
                        y = self.signal[channel_id]
                        sns.lineplot(x=range(len(y)), y=y, ax=axes[channel_id, 0])
            else:
                fig, axes = plt.subplots(self.channels, 1, figsize=figsize, squeeze=False)
                for channel_id in channels:
                    f, t, Sxx = scipy_signal.spectrogram(self.signal[channel_id], spectrogram_freq)
                    axes[channel_id, 0].pcolormesh(t, f, Sxx, shading='gouraud')
        else:
            if spectrogram_freq is None:
                with plt.style.context('seaborn-darkgrid'):
                    fig = plt.figure(figsize=figsize)
                    for channel_id in channels:
                        if spectrogram_freq is None:
                            y = self.signal[channel_id]
                            sns.lineplot(x=range(len(y)), y=y)
            else:
                fig = plt.figure(figsize=figsize)
                f, t, Sxx = scipy_signal.spectrogram(self.signal[0], spectrogram_freq)
                plt.pcolormesh(t, f, Sxx, shading='gouraud')

        fig.patch.set_facecolor('white')
        if title is not None:
            fig.suptitle(title)

        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()

    def show_all(self, spectrogram_freq, title=None):
        self.show(figsize=(18, 1.5 * self.channels), split=True, title=title)
        self.show(figsize=(18, 1.5 * self.channels), split=True, title=title, spectrogram_freq=spectrogram_freq)

        self.margin_interval(interval_length=150).show(
            figsize=(18, 1.5 * self.channels), split=True,
            title=f"{title} - first 150 samples")
        self.play()

    def spectrogram(self, fs, figsize=(16, 9), save_as=None, show=True):
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
    def channels(self):
        return self.signal.shape[0]

    @property
    def dataset(self):
        assert "dataset" in self.meta, """This signal does not have a category, 
        this can happen e.g. by summing two different category signals."""
        return self.meta["dataset"]

    def play(self, channel_id=0, fs=44100):
        # Ensure that highest value is in 16-bit range
        audio = self.signal[channel_id] * (2 ** 15 - 1) / np.max(np.abs(self.signal[channel_id]))
        audio = audio.astype(np.int16)
        play_obj = sa.play_buffer(audio, 1, 2, fs)  # Start playback
        play_obj.wait_done()  # Wait for playback to finish before exiting

    def margin_interval(self, interval_length=None, start_id=0, crop=None):
        if interval_length is None:
            interval_length = len(self)

        if interval_length == len(self) and (start_id == 0 or start_id is None) and crop is None:
            return self

        if crop is None:
            signal = self.signal
            signal_map = self.signal_map
        else:
            assert crop[0] < crop[1], f"Wrong crop interval {crop}"
            signal = self.signal[:, max(crop[0], 0):min(crop[1], len(self))]
            signal_map = self.signal_map[:, max(crop[0], 0):min(crop[1], len(self))]

        new_signal = np.zeros((self.signal.shape[0], interval_length), dtype=self.signal.dtype)
        new_signal_map = np.zeros((self.signal.shape[0], interval_length), dtype=bool)
        sig_len = signal.shape[1]

        new_signal[:, max(0, start_id):min(interval_length, start_id + sig_len)] = \
            signal[:, max(0, -start_id):min(sig_len, interval_length - start_id)]

        new_signal_map[:, max(0, start_id):min(interval_length, start_id + sig_len)] = \
            signal_map[:, max(0, -start_id):min(sig_len, interval_length - start_id)]

        return Signal(build_from=new_signal, meta=self.meta, signal_map=new_signal_map)

    def __add__(self, other):
        if not isinstance(other, Signal):
            return Signal(build_from=self.signal + other, meta=self.meta, signal_map=self.signal_map)

        assert len(self) == len(other), "Adding signals with different lengths is forbidden (for a good reason)"
        signal = self.signal + other.signal
        new_info = join_dicts(self.meta, other.meta)

        return Signal(build_from=signal, meta=new_info, signal_map=(self.signal_map | other.signal_map))

    def __or__(self, other):
        if isinstance(other, Signal):
            other_signal = other
            new_info = join_dicts(self.meta, other.meta)
            new_signal_map = (self.signal_map | other.signal_map)
        else:
            other_signal = Signal(other)
            new_info = self.meta
            new_signal_map = self.signal_map

        assert len(self) == len(other), "Adding signals with different lengths is forbidden (for a good reason)"
        new_signal = np.concatenate([self.signal, other_signal.signal], axis=0)

        return Signal(build_from=new_signal, meta=new_info, signal_map=new_signal_map)

    def __mul__(self, other):
        return Signal(build_from=self.signal * other, meta=self.meta, signal_map=self.signal_map)

    def __truediv__(self, other):
        return Signal(build_from=self.signal / other, meta=self.meta, signal_map=self.signal_map)

    def __repr__(self):
        return str(pd.DataFrame.from_dict(self.meta, orient='index'))

    def to_mp3(self, file, sf=44100, normalized=True):  # todo
        channels = 2 if (self.signal.ndim == 2 and self.signal.shape[0] == 2) else 1
        if normalized:  # normalized array - each item should be a float in [-1, 1)
            y = np.int16(self.signal.T * 2 ** 15)
        else:
            y = np.int16(self.signal.T)
        song = pydub.AudioSegment(y.tobytes(), frame_rate=sf, sample_width=2, channels=channels)
        song.export(file, format="mp3", bitrate="320k")


from multiprocessing import Process


class SignalClass:
    def __init__(self, signals, class_name=None):
        self.signals = signals
        self.class_name = class_name

    def load_to_ram(self) -> None:
        def load_one(signals):
            for signal in signals:
                signal.load_to_ram()

        n = 8
        multi_signals = [self.signals[int(i * len(self.signals) / n): int((i + 1) * len(self.signals) / n)] for i in range(n)]
        p = [Process(target=load_one, args=(signals,)) for signals in multi_signals]
        for p_ in tqdm(p):
            p_.start()

        for p_ in tqdm(p):
            p_.join()


    def get_index_map(self) -> list:
        return [len(i) for i in self.signals]

    def get_signal(self, individual_id, start_id, length):
        return self.signals[individual_id].get_signal(interval=(start_id, start_id+length))

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
        self.total_lengths = {class_name: class_obj.total_length for class_name, class_obj in self.classes_dict.items()}

        for dataset in datasets_config:
            dataset.set_logger(self.logger)
            dataset.set_split_range(self.split_range)
            for class_name, superclass_name, class_obj in dataset.get_class_objects():
                if class_name in self.classes_dict:
                    self.logger.log(f"Class '{class_name}' is defined multiple times. This is not allowed.", 5)
                    raise ValueError(f"Class '{class_name}' is defined multiple times. This is not allowed.")
                self.classes_dict[class_name] = class_obj
                if superclass_name in self.superclasses_dict:
                    self.superclasses_dict[superclass_name].append(class_name)
                else:
                    self.superclasses_dict[superclass_name] = [class_name]

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
    strategy: "random" or "sequence" or "only_once"
    zero_padding: True - no error, makes zero padding if needed, False - error when length is higher than available
    """

    def __init__(self, keeper, class_name, strategy, logger, stride=0, zero_padding=True):
        self.keeper = keeper
        self.class_name = class_name
        self.strategy = strategy
        self.logger = logger
        self.stride = stride
        self.zero_padding = zero_padding

        self.taken_class = self.keeper.get_class(self.class_name)
        self.index_map = self.taken_class.get_index_map()

        self.id_next = [0, 0]
        self.individual_now = 0

    def next(self, length) -> Signal:
        index_map_clean = [max(0, j - length) for i, j in enumerate(self.index_map)]
        if self.strategy == 'random':
            p = np.array(index_map_clean) / np.sum(index_map_clean)
            individual_id = np.random.choice(len(p), p=p)
            start_id = np.random.choice(index_map_clean[individual_id])
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
                 equal_classes=False, strategy='random', stride=0, transforms=None):
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
            self.takers[relevant_class] = SignalTaker(
                keeper=keeper, class_name=relevant_class, strategy=self.strategy, stride=self.stride, logger=self.logger
            )
            self.logger.log(f"Taker of '{relevant_class}' successfully initialized.", priority=2)

    def choose_class(self):
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

    def next(self, length) -> Signal:
        chosen_class = self.choose_class()
        return self.takers[chosen_class].next(length=length)


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

    def benchmark(self, num=1000):
        for _ in trange(num):
            _ = self.next_one()

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
