import json
import re

import numpy as np
from signalai.config import DEVICE
from tqdm import trange

from signalai.signal.signal_dataset import SignalDataset
from signalai.signal.signal_loader import SignalLoader


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