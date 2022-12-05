import re
from pathlib import Path

import numpy as np

from time_series_old import SeriesClass, SeriesDataset
from signalai.tools.utils import set_intersection


class FileLoader(SeriesDataset):  # todo: remake as MultiSignal
    def structurize_files(self) -> dict:
        num_channels = len(self.params["channels"])
        base_dir = Path(self.params["base_dir"])
        class_structure = self.params["class_structure"]
        structured = {}
        for class_name in class_structure.keys():
            per_channel = []
            for i in range(num_channels):
                per_channel.append({})

            structured[class_name] = per_channel

        # structured = {'class_name': [{}, {}, {'file_id': path}]}

        for channel_id, channel_dict in enumerate(self.params["channels"]):
            for path_type, path_value in channel_dict.items():
                if path_type == "re_path":
                    for found_file in base_dir.rglob("*"):
                        file_id = str(found_file)
                        match = re.search(fr"{path_value}$", file_id)
                        if match:
                            match_dict = match.groupdict()
                            if 'channel' in match_dict:
                                start_id, end_id = match.span("channel")
                                file_id = file_id[:start_id] + file_id[end_id:]

                            if 'class' in match_dict:
                                class_name = match_dict['class']
                            else:
                                if len(class_structure) != 1:
                                    raise ValueError(f'There must be only one class defined if it is not a part of the regex.')
                                class_name = list(class_structure.keys())[0]

                            structured[class_name][channel_id][file_id] = found_file
                else:
                    raise ValueError(f"Not recognized path type '{path_type}' in dataset config.")

        assert len(structured) > 0, "FileLoader did not find any files."
        return structured

    def get_class_objects(self) -> list[SeriesClass]:
        generated_result = []
        structured = self.structurize_files()
        for class_name, channel_structure_list in structured.items():
            superclass_name = self.params["class_structure"][class_name]
            valid_file_ids = sorted(set_intersection(*list(set(channel_dict.keys())
                                                     for channel_dict in channel_structure_list)))
            np.random.seed(20150113)
            np.random.shuffle(valid_file_ids)
            if self.params.get("split_by_files", False):
                valid_file_ids = valid_file_ids[int(len(valid_file_ids)*self.split_range[0]):
                                                int(len(valid_file_ids)*self.split_range[1])]
            signals_build = []
            relevant_sample_intervals = self.params.get("relevant_sample_intervals") or [None]
            for relevant_sample_interval in relevant_sample_intervals:
                if not self.params.get("split_by_files", False) and relevant_sample_interval is not None:
                    relevant_sample_length = relevant_sample_interval[1] - relevant_sample_interval[0]
                    relevant_sample_interval = [int(relevant_sample_interval[0] + x * relevant_sample_length)
                                                for x in self.split_range]

                for valid_file_id in valid_file_ids:
                    filenames = [channel_dict[valid_file_id] for channel_dict in channel_structure_list]
                    build_dict = {
                        "files": [{
                            "filename": filename,
                            "file_sample_interval": relevant_sample_interval,
                            'fs': self.params['fs'],
                            **self.params.get("loading_params", {})
                        } for filename in filenames],
                        "transform": self.params.get("transform", []),
                        "target_dtype": self.params.get("target_dtype"),
                        "meta": self.params.get("meta", {}),
                    }
                    self.logger.log(f"Building interval '{relevant_sample_interval}' from files "
                                    f"'{filenames}'. Split range is '{self.split_range}.'", priority=2)
                    signals_build.append(build_dict)

            generated_result.append(SeriesClass(
                series_build=signals_build, class_name=class_name, superclass_name=superclass_name, logger=self.logger,
                purpose=self.purpose,
            ))

        return generated_result


# class ToneGenerator(SeriesDataset):  # todo: repair
#     def __init__(self, fs, max_signal_length, freq, noise_ratio=0., noise_range=(), name=""):
#         self.fs = fs
#         self.max_signal_length = max_signal_length
#         self.freq = freq
#         self.noise_ratio = noise_ratio
#         self.noise_range = noise_range
#         self.name = name
#         self.total_interval_length = max_signal_length
#
#     def get_class_objects(self):
#         pass
#
#     def __next__(self):
#         if isinstance(self.freq, list):
#             assert len(self.freq) == 2 and self.freq[0] < self.freq[1]
#             freq = self.freq[0] + np.random.rand() * (self.freq[1] - self.freq[0])
#         else:
#             freq = self.freq
#
#         start_phase = np.random.rand() * 2 * np.pi
#         base_signal = np.sin(start_phase + 2.0 * np.pi * freq * np.arange(self.max_signal_length) / self.fs)
#         base_signal = np.expand_dims(base_signal, 0)
#         if self.noise_ratio > 0.:
#             base_noise = (np.random.rand(1, self.max_signal_length)-0.5) * 2 * self.noise_ratio
#             if len(self.noise_range) == 2:
#                 base_noise = BandPassFilter(
#                     fs=self.fs,
#                     low_cut=self.noise_range[0],
#                     high_cut=self.noise_range[1]
#                 )(base_noise)
#             base_signal += base_noise
#         return Signal(signal=base_signal), self.name
