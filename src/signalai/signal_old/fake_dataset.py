import numpy as np
from signalai.signal.signal import Signal
from signalai.signal.transformers import BandPassFilter


class ToneGenerator:
    def __init__(self, fs, max_signal_length, freq, noise_ratio=0., noise_range=(), name=""):
        self.fs = fs
        self.max_signal_length = max_signal_length
        self.freq = freq
        self.noise_ratio = noise_ratio
        self.noise_range = noise_range
        self.name = name
        self.total_interval_length = max_signal_length

    def __next__(self):
        if isinstance(self.freq, list):
            assert len(self.freq) == 2 and self.freq[0] < self.freq[1]
            freq = self.freq[0] + np.random.rand() * (self.freq[1] - self.freq[0])
        else:
            freq = self.freq

        start_phase = np.random.rand() * 2 * np.pi
        base_signal = np.sin(start_phase + 2.0 * np.pi * freq * np.arange(self.max_signal_length) / self.fs)
        base_signal = np.expand_dims(base_signal, 0)
        if self.noise_ratio > 0.:
            base_noise = (np.random.rand(1, self.max_signal_length)-0.5) * 2 * self.noise_ratio
            if len(self.noise_range) == 2:
                base_noise = BandPassFilter(
                    fs=self.fs,
                    low_cut=self.noise_range[0],
                    high_cut=self.noise_range[1]
                )(base_noise)
            base_signal += base_noise
        return Signal(signal=base_signal), self.name


if __name__ == '__main__':
    print("TEST")
    fs = 44100
    gen = ToneGenerator(fs, 16384*4, [5000, 6000], noise_ratio=5., noise_range=(100, 5000))
    signal, _ = next(gen)/2
    print(signal.signal)
    signal.spectrogram(fs=fs)
    signal = BandPassFilter(fs=fs, low_cut=1, high_cut=18000.)(signal)
    print(signal.signal)
    signal.spectrogram(fs=fs)
    signal.play()
    signal.show()
