import json
import pathlib
from functools import partial

import yaml
from tqdm import tqdm

from signalai.time_series import read_audio, read_bin, read_npy, TimeSeries, stack_time_series
from signalai.time_series_gen import TimeSeriesHolder


def build_series(file_dict: dict, base_dir: pathlib.PosixPath, target_dtype: str = None) -> TimeSeries:
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
            "filename": base_dir / channel_filename,
            "interval": file_dict.get("crop"),
            "dtype": file_dict.get("target_dtype"),
            "fs": file_dict.get("fs"),
            "meta": file_dict.get("meta"),
        }
        if suffix in [".bin", ".dat"]:
            kwargs['source_dtype'] = file_dict['source_dtype']

        loaded_channels.append(func_map[suffix](**kwargs))

    new_series = stack_time_series(loaded_channels)
    if target_dtype:
        new_series.data_arr = new_series.data_arr.astype(target_dtype)
    assert isinstance(new_series, TimeSeries)
    return new_series


class FileLoader(TimeSeriesHolder):

    def _get_build_info(self) -> list[dict]:
        build_info = self.config['file_structure']
        if isinstance(build_info, str):
            with open(build_info, 'r') as f:
                if build_info.endswith('.json'):
                    build_info = json.load(f)
                elif build_info.endswith('.yaml'):
                    build_info = yaml.load(f, yaml.FullLoader)

        return build_info

    def _build(self):
        assert "base_dir" in self.config
        assert "file_structure" in self.config

        build_info = self._get_build_info()
        partial_build_series = partial(build_series, base_dir=self.config['base_dir'], target_dtype=self.config.get('target_dtype'))
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
