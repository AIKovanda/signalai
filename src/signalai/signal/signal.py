import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import simpleaudio as sa

from signalai.signal.tools import join_dicts


class Signal:
    def __init__(self, signal: np.ndarray, info=None, signal_map=None):
        if info is None:
            info = {}
        self.signal = signal.copy()
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

    @property
    def category_id(self):
        return self.info["dataset_id"]

    @property
    def dummy(self):
        dummy_out = np.zeros(self.info["dataset_total"])
        dummy_out[self.info["dataset_id"]] = 1
        return dummy_out

    def play(self, fs=44100, channel_id=0):
        # Ensure that highest value is in 16-bit range
        audio = self.signal[channel_id] * (2 ** 15 - 1) / np.max(np.abs(self.signal[channel_id]))
        audio = audio.astype(np.int16)
        play_obj = sa.play_buffer(audio, 1, 2, fs)  # Start playback
        play_obj.wait_done()  # Wait for playback to finish before exiting

    def margin_interval(self, interval_length=None, start_id=0, crop=None):
        if interval_length is None:
            interval_length = len(self)

        if interval_length == len(self) and (start_id == 0 or start_id is None) and crop is None:
            return self

        if crop is None:
            signal = self.signal
            signal_map = self.signal_map
        else:
            assert crop[0] < crop[1], f"Wrong crop interval {crop}"
            signal = self.signal[:, max(crop[0], 0):min(crop[1], len(self))]
            signal_map = self.signal_map[:, max(crop[0], 0):min(crop[1], len(self))]

        new_signal = np.zeros((self.signal.shape[0], interval_length), dtype=self.signal.dtype)
        new_signal_map = np.zeros((self.signal.shape[0], interval_length), dtype=bool)
        sig_len = signal.shape[1]

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
        new_info = join_dicts(self.info, other.info)

        return Signal(signal=signal, info=new_info, signal_map=(self.signal_map | other.signal_map))

    def __mul__(self, other):
        return Signal(signal=self.signal*other, info=self.info, signal_map=self.signal_map)

    def __truediv__(self, other):
        return Signal(signal=self.signal/other, info=self.info, signal_map=self.signal_map)

    def __len__(self):
        return self.signal.shape[1]

    def __repr__(self):
        return str(pd.DataFrame.from_dict(self.info, orient='index'))
