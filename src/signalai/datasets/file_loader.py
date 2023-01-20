import pathlib
from functools import partial

import numpy as np
from tqdm import tqdm

from signalai.time_series import read_audio, read_bin, read_npy, stack_time_series, TimeSeries
from signalai.time_series_gen import TimeSeriesHolder
from signalai.tools.utils import get_config


def build_one_series(file_dict: dict, base_dir, target_dtype: str = None, zero_time_map=False) -> TimeSeries:
    if not file_dict:
        raise ValueError(f"There is no information of how to build a signal.")

    func_map = {
        ".aac": read_audio,
        ".wav": read_audio,
        ".mp3": read_audio,
        ".npy": read_npy,
        ".bin": read_bin,
        ".dat": read_bin,
    }

    loaded_channels = []
    assert len(file_dict.get('channels', [])) > 0, f"There is no file to be loaded."
    for channel_filename in file_dict['channels']:
        suffix = str(channel_filename)[-4:]
        kwargs = {
            "filename": pathlib.Path(base_dir) / channel_filename,
            "interval": file_dict.get("interval"),
            "dtype": file_dict.get("target_dtype"),
            "fs": file_dict.get("fs"),
            "meta": file_dict.get("meta"),
        }
        if suffix in [".bin", ".dat"]:
            kwargs['source_dtype'] = file_dict['source_dtype']

        loaded_channels.append(func_map[suffix](**kwargs))

    new_series = stack_time_series(loaded_channels)
    if zero_time_map:
        new_series.time_map = new_series.time_map * 0
    if target_dtype:
        new_series.data_arr = new_series.data_arr.astype(target_dtype)
    assert isinstance(new_series, TimeSeries)
    return new_series


class FileLoader(TimeSeriesHolder):

    def load_individuals(self, all_file_structure: list[dict]) -> None:
        build_info = get_config(all_file_structure)  # may be a dict, json_path or yaml_path
        partial_build_series = partial(build_one_series, base_dir=self.config['base_dir'],
                                       target_dtype=self.config.get('target_dtype'),
                                       zero_time_map=self.config.get('zero_time_map', False))
        if build_info[0].get('priority') is not None:
            self.priorities = [sig_info['priority'] for sig_info in build_info]
        if self.config.get('num_workers', 1) == 1:
            self.timeseries = list(tqdm(map(partial_build_series, build_info), total=len(build_info)))
        else:
            import multiprocessing as mp
            pool = mp.Pool(processes=self.config.get('num_workers', 1))
            self.timeseries = list(tqdm(pool.imap(partial_build_series, build_info), total=len(build_info)))
            pool.close()
            pool.terminate()
            pool.join()  # solving memory leaks
        self._build_index_list()

    def _build(self):
        assert "base_dir" in self.config
        assert "all_file_structure" in self.config
        self.load_individuals(self.config['all_file_structure'])

    def _set_taken_length(self, length):
        assert length is not None
        build_info = get_config(self.config.get('file_structure', [])) + get_config(self.config.get('all_file_structure', []))
        if build_info[0].get('relative_priority') is not None:
            max_interval_count = int(np.max([np.ceil(sig_info['relative_priority'] * len(s) / length) for sig_info, s in zip(build_info, self.timeseries)]))
            self.priorities = [max_interval_count] * len(self.timeseries)

        self._build_index_list()


class GlobFileLoader(FileLoader):

    def _build(self):
        assert "base_dir" in self.config
        assert "file_structure" in self.config

        all_file_structure = [{
            'channels': [i.name],
            **self.config['file_structure'],
        } for i in sorted(pathlib.Path(self.config['base_dir']).glob(self.config.get('glob', '*')))]

        self.load_individuals(all_file_structure)
