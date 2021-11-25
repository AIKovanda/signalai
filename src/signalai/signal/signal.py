import numpy as np
import pandas as pd
import pydub
import seaborn as sns
from matplotlib import pyplot as plt
import simpleaudio as sa
from scipy import signal as scipy_signal
from signalai.tools.utils import join_dicts


class Signal:
    def __init__(self, signal, info=None, signal_map=None):
        if info is None:
            info = {}
        if isinstance(signal, np.ndarray):
            self.signal = signal.copy()
            self.signal_map = np.ones_like(self.signal, dtype=bool) if signal_map is None else signal_map
            self.info = info
        elif isinstance(signal, Signal):
            self.signal = signal.signal.copy()
            self.signal_map = signal_map or signal.signal_map
            self.info = info or signal.info
        else:
            raise TypeError(f"Unknown signal type {type(signal)}.")

    def show(self, channels=None, figsize=(16, 3), save_as=None, show=True, split=False, title=None,
             spectrogram_freq=None):
        if channels is None:
            channels = range(self.signal.shape[0])
        elif isinstance(channels, int):
            channels = [channels]

        if split:
            if spectrogram_freq is None:
                with plt.style.context('seaborn-darkgrid'):
                    fig, axes = plt.subplots(self.channels, 1, figsize=figsize, squeeze=False)
                    for channel_id in channels:
                        y = self.signal[channel_id]
                        sns.lineplot(x=range(len(y)), y=y, ax=axes[channel_id, 0])
            else:
                fig, axes = plt.subplots(self.channels, 1, figsize=figsize, squeeze=False)
                for channel_id in channels:
                    f, t, Sxx = scipy_signal.spectrogram(self.signal[channel_id], spectrogram_freq)
                    axes[channel_id, 0].pcolormesh(t, f, Sxx, shading='gouraud')
        else:
            if spectrogram_freq is None:
                with plt.style.context('seaborn-darkgrid'):
                    fig = plt.figure(figsize=figsize)
                    for channel_id in channels:
                        if spectrogram_freq is None:
                            y = self.signal[channel_id]
                            sns.lineplot(x=range(len(y)), y=y)
            else:
                fig = plt.figure(figsize=figsize)
                f, t, Sxx = scipy_signal.spectrogram(self.signal[0], spectrogram_freq)
                plt.pcolormesh(t, f, Sxx, shading='gouraud')

        fig.patch.set_facecolor('white')
        if title is not None:
            fig.suptitle(title)

        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()

    def show_all(self, spectrogram_freq, title=None):
        self.show(figsize=(18, 1.5 * self.channels), split=True, title=title)
        self.show(figsize=(18, 1.5 * self.channels), split=True, title=title, spectrogram_freq=spectrogram_freq)

        self.margin_interval(interval_length=150).show(
            figsize=(18, 1.5 * self.channels), split=True,
            title=f"{title} - first 150 samples")
        self.play()

    def spectrogram(self, fs, figsize=(16, 9), save_as=None, show=True):
        plt.figure(figsize=figsize)
        f, t, Sxx = scipy_signal.spectrogram(self.signal[0], fs)
        Sxx = np.sqrt(Sxx)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        if save_as is not None:
            plt.savefig(save_as)
        if show:
            plt.show()

    @property
    def channels(self):
        return self.signal.shape[0]

    def join_channels(self, channels):
        assert isinstance(channels, list) or isinstance(channels, tuple), \
            f"Wrong channels type, list or tuple is needed, not {type(channels)}"
        if all([isinstance(i, int) for i in channels]):
            channels = [channels]

        new_signal = []
        for new_channel in channels:
            assert all([isinstance(i, int) for i in new_channel]), \
                f"Wrong channel type, list of ints needed, not list of {type(new_channel[0])}"
            new_signal.append(np.sum(self.signal[new_channel, :], axis=0))

        return Signal(signal=np.array(new_signal))

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

    def play(self, channel_id=0, fs=44100):
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

    def __or__(self, other):
        if isinstance(other, Signal):
            other_signal = other
            new_info = join_dicts(self.info, other.info)
            new_signal_map = (self.signal_map | other.signal_map)
        else:
            other_signal = Signal(other)
            new_info = self.info
            new_signal_map = self.signal_map

        assert len(self) == len(other), "Adding signals with different lengths is forbidden (for a good reason)"
        new_signal = np.concatenate([self.signal, other_signal.signal], axis=0)

        return Signal(signal=new_signal, info=new_info, signal_map=new_signal_map)

    def __mul__(self, other):
        return Signal(signal=self.signal*other, info=self.info, signal_map=self.signal_map)

    def __truediv__(self, other):
        return Signal(signal=self.signal/other, info=self.info, signal_map=self.signal_map)

    def __len__(self):
        return self.signal.shape[1]

    def __repr__(self):
        return str(pd.DataFrame.from_dict(self.info, orient='index'))

    def to_mp3(self, file, sf=44100, normalized=True):
        channels = 2 if (self.signal.ndim == 2 and self.signal.shape[0] == 2) else 1
        if normalized:  # normalized array - each item should be a float in [-1, 1)
            y = np.int16(self.signal.T * 2 ** 15)
        else:
            y = np.int16(self.signal.T)
        song = pydub.AudioSegment(y.tobytes(), frame_rate=sf, sample_width=2, channels=channels)
        song.export(file, format="mp3", bitrate="320k")
