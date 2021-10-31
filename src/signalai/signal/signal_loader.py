import numpy as np
from tqdm import tqdm

from signalai.signal.signal import Signal
from signalai.tools.utils import join_dicts


class SignalLoader:
    def __init__(self, df, log=0):
        self.df = df
        self.log = log
        self.loaded_signals = {}
        if len(self.df) > 0:
            self.load_to_ram()

    def load_to_ram(self):
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