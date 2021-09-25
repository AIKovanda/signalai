import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from signalai.signal_tools.signal import SignalManager
from taskorganizer.pipeline import PipelineTask


class DatasetLoader(PipelineTask):
    @staticmethod
    def load_signal(datasets, default_signal_info):
        categories, filenames, filenames_id, channel_ids, op_dtypes, to_rams, adjustments = [], [], [], [], [], [], []
        interval_starts, interval_ends, values_, frequencies, big_endian, source_dtypes, signed = [], [], [], [], [], [], []
        functional_units, interval_lengths, dtype_bytes = [], [], []
        for dataset_name, dataset_info in datasets.items():
            print("processing dataset:", dataset_name)
            dataset_files = []
            dataset_files_id = []
            file_channel_id = []
            num_channels = len(dataset_info["channels"])
            for channel_id, channel in enumerate(dataset_info["channels"]):
                for path_type in ["path", "re_path"]:
                    if path_type in channel:
                        # dataset_dir, dataset_regex = re.search(r"^(.+)/([^/]+)$", channel[path_type]).groups()
                        if path_type == "path":
                            matched_files = list(Path(dataset_info["folder"]).rglob(channel[path_type]))
                            assert len(matched_files) == 1, \
                                "Must match only one file when using 'path', consider using 're_path', dataset '{dataset_name}'"
                            dataset_files += matched_files
                            dataset_files_id += [dataset_name]
                            file_channel_id += [channel_id]
                        else:
                            for i in Path(dataset_info["folder"]).rglob("*"):
                                match = re.search(f"{channel[path_type]}$", str(i))
                                if match:
                                    dataset_files.append(i)
                                    if num_channels > 1:
                                        start_id, end_id = match.span("channel")
                                        dataset_files_id += [
                                            str(i)[len(dataset_info["folder"]) + 1:start_id] + str(i)[end_id:]]
                                    else:
                                        dataset_files_id += [dataset_name]

                                    file_channel_id += [channel_id]
            
            def get_info(info_name, default=None):
                return dataset_info.get(info_name, default_signal_info.get(info_name, default))
            
            for file_id, dataset_file in enumerate(dataset_files):
                channel_id = file_channel_id[file_id]
                source_dtype = get_info("source_dtype", "float32")
                dtype_byte = np.dtype(source_dtype).itemsize
                file_size = os.path.getsize(dataset_file)
                assert file_size % dtype_byte == 0, f"""file {dataset_file} may be corrupted, 
                                the byte length {file_size} is not compatible 
                                with data type {source_dtype} which has a size of {dtype_byte}"""
                file_size = file_size // dtype_byte
                if "adjustments" in dataset_info:
                    assert len(dataset_info["channels"]) == len(dataset_info["adjustments"]), \
                        "adjustments and channels must have same size, dataset '{dataset_name}'"
                    adjustment = dataset_info["adjustments"][channel_id]
                else:
                    adjustment = 0

                interval_byte = dataset_info.get("relevant_intervals", [[0, file_size]])
                for interval_id, interval in enumerate(interval_byte):
                    assert interval[0] < interval[1], f"interval {interval} makes no sense, dataset '{dataset_name}'"
                    for interval_end in interval:
                        assert 0 <= interval_end + adjustment <= file_size, \
                            f"adjusted interval goes out of the file, dataset '{dataset_name}'"

                    dataset_file = dataset_file.absolute()
                    categories.append(dataset_name)
                    filenames.append(str(dataset_file))
                    filenames_id.append(f"{dataset_files_id[file_id]}-{interval_id}")
                    channel_ids.append(channel_id)
                    values_.append(file_size)

                    adjustments.append(adjustment)

                    interval_starts.append(interval[0])
                    interval_ends.append(interval[1])
                    interval_lengths.append(interval[1] - interval[0])

                    frequencies.append(get_info("frequency", 44100))
                    big_endian.append(get_info("big_endian", True))
                    signed.append(get_info("signed", True))
                    source_dtypes.append(source_dtype)
                    dtype_bytes.append(dtype_byte)
                    functional_units.append(get_info("functional_unit", "anything"))
                    op_dtypes.append(get_info("op_dtype", source_dtype))
                    to_rams.append(get_info("to_ram", True))

        return pd.DataFrame({
            "dataset": categories,
            "filename": filenames,
            "filename_id": filenames_id,
            "channel_id": channel_ids,
            "interval_start": interval_starts,
            "interval_end": interval_ends,
            "interval_length": interval_lengths,
            "values": values_,
            "frequency": frequencies,
            "big_endian": big_endian,
            "source_dtype": source_dtypes,
            "dtype_bytes": dtype_bytes,
            "signed": signed,
            "op_dtype": op_dtypes,
            "to_ram": to_rams,
            "adjustment": adjustments,
            "functional_unit": functional_units
        }).sort_values(by=["filename_id", "channel_id"]).reset_index(drop=True)  # for better readability

    def run(self, datasets, default_signal_info):
        return self.load_signal(datasets, default_signal_info)


class DataGenerator(PipelineTask):
    def run(self, dataset_loader: pd.DataFrame, manager_config, default_tracks_config, fake_datasets):
        return SignalManager(dataset_loader, manager_config, default_tracks_config, fake_datasets, log=1)
