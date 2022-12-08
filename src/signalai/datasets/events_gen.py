from pathlib import Path

import numpy as np
import pandas as pd

from signalai.time_series_gen import TimeSeriesGen, TimeSeriesHolder
from signalai.time_series import TimeSeries, sum_time_series, stack_time_series, Signal, read_audio


class EventsGen(TimeSeriesGen):
    def __len__(self):
        pass

    def _getitem(self, item: int) -> TimeSeries:
        pass

    def is_infinite(self) -> bool:
        return True

    def _build(self):
        assert 'filename' in self.config
        assert 'classes_file' in self.config
        self.build_class_dicts()

    def build_class_dicts(self):
        df = pd.read_csv(self.config['classes_file'])
        unique_events = df['event'].drop_duplicates().to_list()

        files = list(Path('/'.join(self.config["filename"].split('/')[:-1])).glob(
            self.config["filename"].split('/')[-1]))
        assert len(files) > 0, 'There is no file!'

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
    
    def next(self) -> TimeSeries:
        """
        Event compose, randomly taking events into a MultiSeries.
        """
        events_number = np.random.choice(range(*self.config.get('events_count_range', (0, 10))))
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

        non_zero_signal = list(non_zero_signals.values())[0]
        zero_signal = Signal(data_arr=np.zeros_like(non_zero_signal.data_arr), time_map=np.zeros_like(non_zero_signal.time_map), meta=non_zero_signal.meta)

        return stack_time_series([non_zero_signals.get(key, zero_signal) for key in event_names])
