from pathlib import Path

import numpy as np
import pandas as pd

from signalai.config import DATA_DIR
from signalai.datasets.file_loader import FileLoader, GlobFileLoader
from signalai.time_series import read_audio, Signal, stack_time_series, sum_time_series, TimeSeries
from signalai.time_series_gen import TimeSeriesGen, TimeSeriesHolder
from signalai.tools.utils import get_config, join_dicts


class EventsGen(TimeSeriesGen):
    def __init__(self, **config):
        super().__init__(**config)
        self.signal_info = {}

    def __len__(self):
        pass

    def is_infinite(self) -> bool:
        return True

    def set_signal_info(self):
        dim_set = set()
        meta_list = []
        fs_set = set()
        for input_ts_gen in self.input_ts_gen_kwargs.values():
            s = input_ts_gen.get_random_item()
            dim_set.add(s.data_arr.shape[0])
            meta_list.append(s.meta)
            fs_set.add(s.fs)
        assert len(fs_set) == 1, 'Items have different sample frequency.'
        assert len(dim_set) == 1, 'Items have different channel number.'
        self.signal_info['fs'] = fs_set.pop()
        self.signal_info['dim'] = dim_set.pop()
        self.signal_info['meta'] = join_dicts(*meta_list)

    def _build(self):
        assert 'structure' in self.config

        for class_name, event_structure in get_config(self.config['structure']).items():
            assert 'base_dir' in event_structure
            assert 'all_file_structure' in event_structure
            self.input_ts_gen_kwargs[class_name] = FileLoader(
                base_dir=event_structure['base_dir'].format(DATA_DIR=DATA_DIR), all_file_structure=event_structure['all_file_structure'],
            )
            self.input_ts_gen_kwargs[class_name].build()

        self.set_signal_info()

    def _getitem(self, item: int) -> TimeSeries:
        assert self.taken_length is not None
        events_number = np.random.choice(range(*self.config.get('event_count_range', (0, 10))))
        event_names = list(self.input_ts_gen_kwargs.keys())
        chosen_event_names = [np.random.choice(event_names) for _ in range(events_number)]

        allowed_starting_indices = np.arange(*self.config.get('start_arange', [1]))
        allowed_event_length = np.arange(*self.config.get('event_length_arange', [self.taken_length]))

        non_zero_signals = {}

        for chosen_event_name in event_names:
            chosen_event_intervals = []

            for i in range(chosen_event_names.count(chosen_event_name)):
                starting_index = np.random.choice(allowed_starting_indices)
                event_length = np.random.choice(allowed_event_length)
                ending_index = min(starting_index + event_length, self.taken_length)  # not to be over the edge
                for interval in chosen_event_intervals:
                    if interval[0] <= starting_index <= interval[1]:
                        break
                    if interval[0] <= ending_index <= interval[1]:
                        break
                    if starting_index <= interval[0] and ending_index >= interval[1]:
                        break
                else:
                    chosen_event_intervals.append([starting_index, ending_index])

            if len(chosen_event_intervals) != 0:
                non_zero_signals[chosen_event_name] = sum_time_series(
                    [self.input_ts_gen_kwargs[chosen_event_name].get_random_item().crop(
                        (0, ending_index - starting_index)).margin_interval(
                        interval_length=self.taken_length,
                        start_id=starting_index,
                    ) for (starting_index, ending_index) in chosen_event_intervals])

        zero_signal = Signal(
            data_arr=np.zeros((self.signal_info['dim'], self.taken_length)),
            time_map=np.zeros((self.signal_info['dim'], self.taken_length)),
            meta=self.signal_info['meta'],
            fs=self.signal_info['fs'],
        )
        return stack_time_series([non_zero_signals.get(key, zero_signal) for key in event_names])


class GlobEventsGen(EventsGen):
    def _build(self):
        assert 'structure' in self.config

        for class_name, event_structure in get_config(self.config['structure']).items():
            assert 'base_dir' in event_structure
            assert 'file_structure' in event_structure, 'file_structure is not set (at least to empty dict)'
            self.input_ts_gen_kwargs[class_name] = GlobFileLoader(
                base_dir=event_structure['base_dir'].format(DATA_DIR=DATA_DIR), file_structure=event_structure['file_structure'],
            )
            self.input_ts_gen_kwargs[class_name].build()

        self.set_signal_info()


class CSVEventsGen(EventsGen):
    def _build(self):
        assert 'file_path' in self.config
        assert 'csv_file' in self.config

        df = pd.read_csv(self.config['csv_file'])
        unique_events = df['event'].drop_duplicates().to_list()

        files = list(Path('/'.join(self.config["file_path"].split('/')[:-1])).glob(
            self.config["file_path"].split('/')[-1]))
        assert len(files) > 0, f'There is no file at {self.config["file_path"]}!'

        # loading all files
        all_signals = {}
        for file in files:
            full_signal = read_audio(file, dtype=self.config.get("target_dtype"))
            assert full_signal.fs is not None
            full_signal.update_meta(self.config.get("meta", {}))
            all_signals[str(file)] = full_signal

        for unique_event_type in unique_events:
            sub_df = df.query(f"event=='{unique_event_type}'")
            event_signals = []
            for row in sub_df.itertuples():
                for origin_filename, full_signal in all_signals.items():
                    s = full_signal.crop(interval=(row.sample_start, row.sample_end))
                    # s.trim_(threshold=1e-7)  # found around 1000 empty samples in the beginning
                    s.update_meta({'force': row.force, 'origin': origin_filename, 'class_name': unique_event_type})
                    event_signals.append(s)

            assert len(event_signals) > 0, f'Event type {unique_event_type!r} cannot load its signals.'
            self.input_ts_gen_kwargs[unique_event_type] = TimeSeriesHolder(
                timeseries=event_signals,
            )
            self.input_ts_gen_kwargs[unique_event_type].build()

        self.set_signal_info()
