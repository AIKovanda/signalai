import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from signalai.signal_tools.signal import SignalManagerGenerator, SignalLoader
from taskorganizer.pipeline import PipelineTask


class DatasetLoader(PipelineTask):
    @staticmethod
    def load_signal(datasets, default_signal_info):
        series_list = []
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
                                            dataset_name + "-" + str(i)[len(dataset_info["folder"]) + 1:start_id] + str(i)[end_id:]]
                                    else:
                                        dataset_files_id += [dataset_name + "-" + str(i)[len(dataset_info["folder"]) + 1:]]

                                    file_channel_id += [channel_id]
            
            def get_info(info_name, default=None):
                return dataset_info.get(info_name, default_signal_info.get(info_name, default))

            unique_dataset_files_id = sorted(set(dataset_files_id))
            splittable = get_info("splittable", True)
            split_ratios = get_info("split", [.5, .25, .25])

            if splittable:
                split_map = None
            else:
                np.random.seed(42)
                np.random.shuffle(unique_dataset_files_id)
                split_map = ["train"] * int(split_ratios[0] * len(unique_dataset_files_id))
                split_map += ["valid"] * int(split_ratios[1] * len(unique_dataset_files_id))
                split_map += ["test"] * (len(unique_dataset_files_id) - len(split_map))

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
                    if split_map is not None:
                        total_split_info = [{
                            "split": split_map[unique_dataset_files_id.index(dataset_files_id[file_id])],
                            "interval_start": interval[0],
                            "interval_end": interval[1],
                            "interval_length": interval[1] - interval[0]
                        }]
                    else:
                        interval_length = interval[1] - interval[0]
                        total_split_info = [{
                            "split": "train",
                            "interval_start": interval[0],
                            "interval_end": interval[0] + int(split_ratios[0] * interval_length),
                            "interval_length": int(split_ratios[0] * interval_length)
                        }, {
                            "split": "valid",
                            "interval_start": interval[0] + int(split_ratios[0] * interval_length),
                            "interval_end": interval[0] + int(split_ratios[0] * interval_length) + int(split_ratios[1] * interval_length),
                            "interval_length": int(split_ratios[1] * interval_length)
                        }, {
                            "split": "test",
                            "interval_start": interval[0] + int(split_ratios[0] * interval_length) + int(split_ratios[1] * interval_length),
                            "interval_end": interval[1],
                            "interval_length": interval[1] - (interval[0] + int(split_ratios[0] * interval_length) + int(split_ratios[1] * interval_length))
                        }]

                    dataset_file = dataset_file.absolute()
                    for split_info in total_split_info:
                        series_list.append(pd.Series({
                            "dataset": dataset_name,
                            "dataset_id": get_info("dataset_id"),
                            "filename": str(dataset_file),
                            "filename_id": f"{dataset_files_id[file_id]}-{interval_id}-{split_info['split']}",
                            "channel_id": channel_id,
                            "split": split_info["split"],
                            "interval_start": split_info["interval_start"],
                            "interval_end": split_info["interval_end"],
                            "interval_length": split_info["interval_length"],
                            "values": file_size,
                            "frequency": get_info("frequency", 44100),
                            "big_endian": get_info("big_endian", True),
                            "source_dtype": source_dtype,
                            "dtype_bytes": dtype_byte,
                            "signed": get_info("signed", True),
                            "op_dtype": get_info("op_dtype", source_dtype),
                            "to_ram": get_info("to_ram", True),
                            "standardize": get_info("standardize", False),
                            "adjustment": adjustment
                        }))

        df = pd.DataFrame(series_list).sort_values(by=["filename_id", "channel_id"]).reset_index(drop=True)  # for better readability

        dataset_list = df.dataset.drop_duplicates().sort_values().to_list()
        df["dataset_total"] = len(dataset_list)
        return df

    def run(self, datasets, default_signal_info):
        return self.load_signal(datasets, default_signal_info)


class DataGenerator(PipelineTask):
    def run(self, dataset_loader: pd.DataFrame, manager_config, default_tracks_config, fake_datasets):
        return SignalManagerGenerator(
            dataset_loader,
            manager_config,
            default_tracks_config,
            fake_datasets,
            log=1)
