import json
import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import simpleaudio as sa
from signalai.config import DEVICE
from tqdm import tqdm, trange


def join_dicts(*args):
    if all([i == args[0] for i in args]):
        return args[0]
    else:
        new_info = {}
        for key, value in args[0].items():
            if all([key in i for i in args]):
                if all([value == i[key] for i in args]):
                    new_info[key] = value
        return new_info


class SignalManagerGenerator:
    def __init__(self, df, manager_config, default_tracks_config, fake_datasets=None, log=0):
        self.df = df
        self.manager_config = manager_config
        self.default_tracks_config = default_tracks_config
        self.fake_datasets = fake_datasets
        self.log = log
        self.signal_loader = None

    def get_generator(self, split, batch_size=1, log=None, x_name="X", y_name="Y"):
        self.signal_loader = SignalLoader(self.df, log=0)
        if log is None:
            log = self.log
        return SignalManager(
            self.df,
            manager_config=self.manager_config, signal_loader=self.signal_loader,
            default_tracks_config=self.default_tracks_config, batch_size=batch_size,
            fake_datasets=self.fake_datasets, log=log, split=split, x_name=x_name, y_name=y_name)


class SignalManager:
    def __init__(self, df, manager_config, signal_loader, default_tracks_config, batch_size=1, split=None, fake_datasets=None, log=0, x_name="X", y_name="Y"):
        if fake_datasets is None:
            fake_datasets = {}

        self.df = df
        self.signal_loader = signal_loader
        self.split = split
        self.all_available_datasets = self.df.dataset.drop_duplicates().to_list()

        self.manager_config = manager_config
        self.fake_datasets = fake_datasets
        self.default_tracks_config = default_tracks_config
        self.batch_size = batch_size
        self.log = log

        self.max_signal_length = None
        self.transformers = {}
        self.tracks = {}
        self.present_tracks = []

        self.X_re = self.manager_config[x_name]
        self.Y_re = self.manager_config[y_name]

        assert "type" in manager_config, "manager_config must have specified type"
        self.type_ = manager_config["type"]

        assert "tracks" in self.manager_config, f"Tracks info missing in manager_config"
        self.tracks_info = self.manager_config["tracks"]

        if self.type_ == 'simple_manager':
            self.init_simple_manager()
            self.init_transformers()
        elif self.type_ == 'midi':
            raise NotImplementedError
        else:
            raise ValueError(f"{self.type_} is an unknown manager type, choose either 'simple_manager' or 'midi'")

    def init_simple_manager(self):
        for track_name, track_info in self.tracks_info.items():
            track_datasets = []
            for dataset_regex in track_info["datasets"]:
                for available_dataset in self.all_available_datasets:
                    if re.match(dataset_regex, available_dataset):
                        track_datasets.append(available_dataset)

            next_after_samples = self.get_info(track_name, "next_after_samples")

            if self.log > 0:
                print(f"track {track_name} initialized with datasets {json.dumps(track_datasets)}")
                print(track_name, next_after_samples)

            max_signal_length = self.get_info(track_name, "max_signal_length")
            self.tracks[track_name] = {
                "datasets": self.init_datasets(track_datasets, max_signal_length, next_after_samples),
                "max_signal_length": max_signal_length,
                "equal_category": self.get_info(track_name, "equal_category"),
                "next_after_samples": next_after_samples,
                "length": self.get_info(track_name, "length")
            }
            if re.search(fr"(^|[\W])({track_name})($|[\W])", self.X_re) or re.search(fr"(^|[\W])({track_name})($|[\W])", self.Y_re):
                self.present_tracks.append(track_name)
            self.X_re = re.sub(fr"(^|[\W])({track_name})($|[\W])", fr"\1signal_dict['\2']\3", self.X_re)
            self.Y_re = re.sub(fr"(^|[\W])({track_name})($|[\W])", fr"\1signal_dict['\2']\3", self.Y_re)

    def get_info(self, track_name, info_name):
        assert info_name in self.manager_config or info_name in self.default_tracks_config, f"{info_name} is missing in manager_config"
        return self.tracks_info[track_name].get(info_name, self.default_tracks_config.get(info_name))

    def init_datasets(self, datasets, max_signal_length, next_after_samples):
        initialized_datasets = {}
        for dataset_name, sub_df in self.df.groupby("dataset"):
            if dataset_name in datasets:
                initialized_datasets[dataset_name] = SignalDataset(
                    df=sub_df,
                    signal_loader=self.signal_loader,
                    max_signal_length=max_signal_length,
                    split=self.split,
                    next_after_samples=next_after_samples,
                    log=self.log)

        for dataset in self.fake_datasets:  # todo
            pass

        return initialized_datasets

    def init_transformers(self):
        for transformer_name, transformer_info in self.manager_config.get("transformers", {}).items():
            transformer_class = transformer_info["class"]
            transformer_from = ".".join(transformer_class.split(".")[:-1])
            transformer_class_name = transformer_class.split(".")[-1]
            exec(f"from {transformer_from} import {transformer_class_name}")

            params = transformer_info.get("params", {})
            transformer = eval(f"{transformer_class_name}(**params)")
            self.transformers[transformer_name] = transformer
            self.X_re = re.sub(fr"(^|[\W])({transformer_name})($|[\W])", fr"\1self.transformers['\2']\3", self.X_re)
            self.Y_re = re.sub(fr"(^|[\W])({transformer_name})($|[\W])", fr"\1self.transformers['\2']\3", self.Y_re)

    def next_simple_manager(self):
        signal_dict = {}
        start_id = 0
        for track_name, track_info in self.tracks.items():
            if track_name not in self.present_tracks:
                continue
            if track_info["equal_category"]:
                p = np.ones(len(track_info["datasets"]))
            else:
                p = np.array([i.total_interval_length for i in track_info["datasets"]])

            p = p / np.sum(p)
            chosen_dataset = np.random.choice(list(track_info["datasets"].values()), p=p)
            signal, start_id = next(chosen_dataset)
            signal.margin_interval(track_info["length"], start_id=None, crop=None)  # to fit the track_info["length"]
            signal_dict[track_name] = signal

        X = eval(self.X_re)
        Y = eval(self.Y_re)
        return X, Y , start_id

    def next_batch(self):
        X_b, Y_b, ids = [], [], []
        for _ in range(self.batch_size):
            X, Y, start_id = self.next_simple_manager()
            X_b.append(X)
            Y_b.append(Y)
            ids.append(start_id)

        return X_b, Y_b  #, ids

    def benchmark_data_generator(self, num=1000, device=None):
        import torch
        if device is None:
            device = DEVICE
        for _ in trange(num):
            x, y = self.__next__()
            _ = torch.from_numpy(np.array(x)).to(device)
            _ = torch.from_numpy(np.array(y)).to(device)

    def __next__(self):
        if self.type_ == 'simple_manager':
            return self.next_batch()
        elif self.type_ == 'midi':
            raise NotImplementedError
        else:
            raise ValueError(f"{self.type_} is an unknown manager type, choose either 'simple_manager' or 'midi'")


class SignalLoader:
    def __init__(self, df, log=0):
        self.df = df
        self.log = log
        self.loaded_signals = {}
        for chosen_filename_id in tqdm(self.df.query("to_ram").filename_id.drop_duplicates().to_list(),
                                       desc=f"Loading datasets {self.df.query('to_ram').dataset.drop_duplicates().to_list()} to RAM"):
            self.loaded_signals[chosen_filename_id] = self.load_from_disc(filename_id=chosen_filename_id)

    def load_from_disc(self, filename_id, start_relative=0, max_interval_length=None):
        chosen_sub_df = self.df.query(f"filename_id=='{filename_id}'").sort_values(by="channel_id")
        chosen_sub_df_info = join_dicts(*[chosen_sub_df.iloc[i].to_dict() for i in range(len(chosen_sub_df))])
        interval_start = int(chosen_sub_df_info["interval_start"])
        all_interval_length = int(chosen_sub_df_info["interval_length"])
        if max_interval_length is None:
            max_interval_length = all_interval_length
        interval_length = min(max_interval_length, all_interval_length)

        loaded_signal = []
        for row in chosen_sub_df.itertuples():
            real_start = interval_start + start_relative + int(row.adjustment)
            if self.log > 0:
                print(f"Sample taken from {real_start} to {real_start + interval_length}, channel {row.channel_id}")

            with open(row.filename, "rb") as f:
                f.seek(int(row.dtype_bytes) * real_start, 0)
                loaded_signal.append(np.fromfile(f, dtype=row.source_dtype, count=interval_length))

        stacked_signal = np.vstack(loaded_signal)
        if chosen_sub_df_info["standardize"]:
            stacked_signal = (stacked_signal - np.mean(stacked_signal)) / np.std(stacked_signal)

        return Signal(stacked_signal.astype(chosen_sub_df_info["op_dtype"]), info=chosen_sub_df_info)

    def load(self, filename_id, start_relative=0, max_interval_length=None):
        if filename_id in self.loaded_signals:
            return self.loaded_signals[filename_id].margin_interval(max_interval_length, start_id=-start_relative)

        return self.load_from_disc(filename_id, start_relative, max_interval_length)


class SignalIndexer:
    def __init__(self, df, max_signal_length, next_after_samples=False, log=0):
        self.df = df.drop_duplicates(subset="filename_id").reset_index(drop=True)
        self.max_signal_length = max_signal_length
        self.next_after_samples = next_after_samples
        self.p = self.df.interval_length.to_numpy()
        self.total_interval_length = np.sum(self.p)
        self.p = self.p / np.sum(self.p)
        self.indexing_end = {
            int(i.Index): max(0, int(i.interval_length)-self.max_signal_length) for i in self.df.itertuples()
        }
        self.id_now = 0
        self.file_now = 0
        self.log = log

    def __next__(self):
        if self.next_after_samples:
            id_ = self.id_now
            filename_id = self.df.loc[self.file_now, "filename_id"]
            self.id_now += self.next_after_samples
            if self.id_now > self.indexing_end[self.file_now]:
                self.id_now = 0
                if self.log > 0:
                    print("new cycle")

                self.file_now += 1
                if self.file_now >= len(self.indexing_end):
                    self.file_now = 0
            return filename_id, id_

        else:
            random_df_index = np.random.choice(self.df.index.to_list(), p=self.p)
            chosen_filename_id = self.df.at[random_df_index, 'filename_id']
            chosen_relative_start_id = np.random.choice(self.indexing_end[random_df_index]+1)
            return chosen_filename_id, chosen_relative_start_id


class SignalDataset:
    def __init__(self, df, signal_loader, max_signal_length, split=None, next_after_samples=False, log=0):
        self.df = df
        self.signal_loader = signal_loader
        self.max_signal_length = max_signal_length
        self.next_after_samples = next_after_samples
        self.split = split
        if self.split is not None:
            self.df = self.df.query(f"split=='{self.split}'")
        self.log = log
        self.signal_indexer = SignalIndexer(self.df, self.max_signal_length, self.next_after_samples, log=self.log)
        self.total_interval_length = self.signal_indexer.total_interval_length

    def __next__(self):
        chosen_filename_id, relative_start = next(self.signal_indexer)
        return self.signal_loader.load(
            chosen_filename_id,
            start_relative=relative_start,
            max_interval_length=self.max_signal_length
        ), relative_start


class Signal:
    def __init__(self, signal: np.ndarray, info=None, signal_map=None):
        if info is None:
            info = {}
        self.signal = signal.copy()
        self.signal_map = np.ones_like(self.signal, dtype=bool) if signal_map is None else signal_map
        self.info = info

    def show(self, channels=None, figsize=(16, 9), save_as=None, show=True):
        if channels is None:
            channels = list(range(self.signal.shape[0]))

        if isinstance(channels, int):
            channels = [channels]

        plt.figure(figsize=figsize)
        for channel_id in channels:
            y = self.signal[channel_id].copy()
            y -= np.mean(y)
            y = y / np.std(y)
            sns.lineplot(x=range(len(y)), y=y)
        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()

    @property
    def joined_signal(self):
        return Signal(signal=self._joined_signal, info=self.info, signal_map=self.signal_map[:1, :])

    @property
    def _joined_signal(self):
        return np.expand_dims(np.sum(self.signal, axis=0), 0)

    @property
    def dataset(self):
        assert "dataset" in self.info, """This signal does not have a category, 
        this can happen e.g. by summing two different category signals."""
        return self.info["dataset"]

    @property
    def category(self):
        return self.dataset

    @property
    def category_id(self):
        return self.info["dataset_id"]

    @property
    def dummy(self):
        dummy_out = np.zeros(self.info["dataset_total"])
        dummy_out[self.info["dataset_id"]] = 1
        return dummy_out

    def play(self, fs=44100, channel_id=0):
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

        new_signal[:, max(0, start_id):min(interval_length, start_id+sig_len)] = \
            signal[:, max(0, -start_id):min(sig_len, interval_length-start_id)]

        new_signal_map[:, max(0, start_id):min(interval_length, start_id + sig_len)] = \
            signal_map[:, max(0, -start_id):min(sig_len, interval_length-start_id)]

        return Signal(signal=new_signal, info=self.info, signal_map=new_signal_map)

    def __add__(self, other):
        if not isinstance(other, Signal):
            return Signal(signal=self.signal+other, info=self.info, signal_map=self.signal_map)

        assert len(self) == len(other), "Adding signals with different lengths is forbidden (for a good reason)"
        signal = self.signal + other.signal
        new_info = join_dicts(self.info, other.info)

        return Signal(signal=signal, info=new_info, signal_map=(self.signal_map | other.signal_map))

    def __mul__(self, other):
        return Signal(signal=self.signal*other, info=self.info, signal_map=self.signal_map)

    def __truediv__(self, other):
        return Signal(signal=self.signal/other, info=self.info, signal_map=self.signal_map)

    def __len__(self):
        return self.signal.shape[1]

    def __repr__(self):
        return str(pd.DataFrame.from_dict(self.info, orient='index'))
