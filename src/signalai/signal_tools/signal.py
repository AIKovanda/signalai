import importlib
import json
import re
from abc import abstractmethod

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import simpleaudio as sa


class SignalManager:
    def __init__(self, df, manager_config, default_tracks_config, fake_datasets=None, log=0):
        if fake_datasets is None:
            fake_datasets = {}

        self.df = df
        self.all_available_datasets = self.df.dataset.drop_duplicates().to_list()

        self.manager_config = manager_config
        self.fake_datasets = fake_datasets
        self.default_tracks_config = default_tracks_config
        self.log = log

        self.max_signal_length = None
        self.datasets = {}
        self.transformers = {}
        self.tracks = {}

        self.X_re = self.manager_config["X"]
        self.Y_re = self.manager_config["Y"]

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
            track_dict = {}
            for dataset_regex in track_info["datasets"]:
                for available_dataset in self.all_available_datasets:
                    if re.match(dataset_regex, available_dataset):
                        track_datasets.append(available_dataset)

            if self.log > 0:
                print(f"track {track_name} initialized with datasets {json.dumps(track_datasets)}")

            next_after_samples = self.get_info(track_name, "next_after_samples")
            max_signal_length = self.get_info(track_name, "max_signal_length")
            self.tracks[track_name] = {
                "datasets": self.init_datasets(track_datasets, max_signal_length, next_after_samples),
                "max_signal_length": max_signal_length,
                "equal_category": self.get_info(track_name, "equal_category"),
                "next_after_samples": next_after_samples,
                "length": self.get_info(track_name, "length")
            }

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
                    max_signal_length=max_signal_length,
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

        for track_name, track_info in self.tracks.items():
            if track_info["equal_category"]:
                p = np.ones(len(track_info["datasets"]))
            else:
                p = np.array([i.total_interval_length for i in track_info["datasets"]])

            p = p / np.sum(p)
            chosen_dataset = np.random.choice(list(track_info["datasets"].values()), p=p)
            signal = next(chosen_dataset).margin_interval(track_info["length"], start_id=None, crop=None)
            signal_dict[track_name] = signal

        X = eval(self.X_re)
        Y = eval(self.Y_re)
        return X, Y

    def __next__(self):
        if self.type_ == 'simple_manager':
            return self.next_simple_manager()
        elif self.type_ == 'midi':
            raise NotImplementedError
        else:
            raise ValueError(f"{self.type_} is an unknown manager type, choose either 'simple_manager' or 'midi'")


class SignalDataset:
    def __init__(self, df, max_signal_length, next_after_samples=False, log=0):
        self.df = df
        self.max_signal_length = max_signal_length
        self.next_after_samples = next_after_samples
        self.start_relative = 0  # only useful if next_after_samples != False
        self.log = log
        self.total_interval_length = None

    def __next__(self):
        p = self.df.interval_length.to_numpy()
        p = p / np.sum(p)
        self.total_interval_length = np.sum(p)
        chosen_filename_id = self.df.at[np.random.choice(self.df.index.to_list(), p=p), 'filename_id']
        chosen_sub_df = self.df.query(f"filename_id=='{chosen_filename_id}'").sort_values(
            by="channel_id")

        chosen_sub_df_info = chosen_sub_df.iloc[0].to_dict()
        interval_start = int(chosen_sub_df_info["interval_start"])
        interval_length = int(chosen_sub_df_info["interval_length"])
        if interval_length <= self.max_signal_length:
            start_relative = 0
            taken_interval_length = interval_length
        else:
            taken_interval_length = self.max_signal_length
            if not self.next_after_samples:
                start_relative = np.random.choice(interval_length - self.max_signal_length)
            else:
                if self.start_relative + taken_interval_length > interval_length:
                    self.start_relative = 0
                start_relative = self.start_relative
                self.start_relative += self.next_after_samples
        
        loaded_signal = []
        for row in chosen_sub_df.itertuples():
            real_start = interval_start + start_relative + int(row.adjustment)

            if self.log > 0:
                print(f"Sample taken from {real_start} to {real_start+taken_interval_length}, channel {row.channel_id}")

            with open(row.filename, "rb") as f:
                f.seek(int(row.dtype_bytes)*real_start, 0)
                loaded_signal.append(np.fromfile(f, dtype=row.source_dtype, count=taken_interval_length))
        
        return Signal(np.vstack(loaded_signal), info=chosen_sub_df_info)
        

class Signal:
    def __init__(self, signal: np.ndarray, info=None, signal_map=None):
        if info is None:
            info = {}
        self.signal = signal
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

    def play(self, fs=44100, channel_id=0):
        # Ensure that highest value is in 16-bit range
        audio = self.signal[channel_id] * (2 ** 15 - 1) / np.max(np.abs(self.signal[channel_id]))
        audio = audio.astype(np.int16)
        play_obj = sa.play_buffer(audio, 1, 2, fs)  # Start playback
        play_obj.wait_done()  # Wait for playback to finish before exiting

    def margin_interval(self, interval_length, start_id=None, crop=None):
        if interval_length == len(self) and (start_id == 0 or start_id is None) and crop is None:
            return self

        if crop is None:
            signal = self.signal
            signal_map = self.signal_map
        else:
            assert crop[0] < crop[1], f"Wrong crop interval {crop}"
            signal = self.signal[:, max(crop[0], 0):min(crop[1], len(self))]
            signal_map = self.signal_map[:, max(crop[0], 0):min(crop[1], len(self))]

        new_signal = np.zeros((self.signal.shape[0], interval_length))
        new_signal_map = np.zeros((self.signal.shape[0], interval_length), dtype=bool)
        sig_len = signal.shape[1]

        if start_id is None:
            start_id = np.random.randint(0, interval_length - sig_len)

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
        if self.info == other.info:
            new_info = self.info
        else:
            new_info = {}
            for key, value in self.info.items():
                if key in other.info:
                    if value == other.info[key]:
                        new_info[key] = value

        return Signal(signal=signal, info=new_info, signal_map=(self.signal_map | other.signal_map))

    def __mul__(self, other):
        return Signal(signal=self.signal*other, info=self.info, signal_map=self.signal_map)

    def __truediv__(self, other):
        return Signal(signal=self.signal/other, info=self.info, signal_map=self.signal_map)

    def __len__(self):
        return self.signal.shape[1]
