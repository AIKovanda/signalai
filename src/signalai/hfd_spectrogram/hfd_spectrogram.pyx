import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"

from scipy import signal
from math import sin, cos, pi
from libc.stdlib cimport malloc, free


class KernelFunction:
    def __init__(self, source_name, windows, param, dtype):
        self.source_name = source_name
        self.source_path = experiment_config["files"][self.source_name][0]
        self.windows = windows  # [window_length, window_number, shift]
        self.window_length = int(windows[0])
        self.window_number = int(windows[1])
        self.stride = int(windows[2])
        self.parameters = param
        self.dtype = dtype
        print(f"Loading {self.source_path}")
        if self.source_path.split(".")[-1] == "npy":
            self.bin = False
            self.opened_file = np.load(self.source_path)
        else:
            self.bin = True
            self.opened_file = open(self.source_path, 'rb')
        self.transform_length = self.window_length + (self.window_number - 1) * self.stride
        self.seek_index = 0  # for numpy loader

    def get_data(self):  # data loading
        # loaded_raw=np.array([int.from_bytes(self.opened_file.read(SOURCE_CODING[0]), byteorder=SOURCE_CODING[1], signed=SOURCE_CODING[2]) for i in range(self.transformed_length)])
        if self.bin:
            loaded_raw = np.fromfile(self.opened_file, dtype=experiment_config["source_type"],
                                 count=self.transform_length)
        else:
            loaded_raw = self.opened_file[self.seek_index: self.seek_index + self.transform_length]
        # self.opened_file.seek(self.stride, 1)
        # window_values=np.array([(1-(2*i/N-1)**2) for i in range(self.transformed_length)]) # Welch window
        window_values = np.array([1 for i in range(self.transform_length)])  # rectangular window
        return loaded_raw * window_values

    def set_reader_byte_position(self, position):
        if self.bin:
            self.opened_file.seek(position * experiment_config["source_bytes"], 0)
        else:
            self.seek_index = position


class HDSpectrogram(KernelFunction):
    def transform(self):
        cdef int x,k,freq
        cdef double w

        cdef double[:] frequency = self.parameters[0]
        cdef int freq_len = len(self.parameters[0])
        cdef double[:] loaded_data = signal.detrend(self.get_data())

        cdef int y_wide = self.parameters[0].shape[0]
        cdef int x_wide = self.window_number
        cdef int shift = self.stride

        cdef int window_length =self.window_length
        cdef complex *exp_core = <complex *> malloc(window_length * sizeof(complex))
        cdef double *transformed_data = <double *> malloc(freq_len * x_wide * sizeof(double))
        cdef complex temp
        for freq in range(y_wide):  # iteration through frequencies
            for k in range(window_length):
                exp_core[k] = cos(frequency[freq] * pi * k) + 1j * sin(frequency[freq] * pi * k)
            for x in range(x_wide):  # stride
                temp = 0 * 1j
                for k in range(window_length):
                    temp += exp_core[k] * loaded_data[x * shift + k]
                transformed_data[x + freq * x_wide] = abs(temp)
        free(exp_core)
        df = np.array([w for w in transformed_data[:freq_len * x_wide]], dtype=self.dtype).reshape((freq_len, x_wide))
        free(transformed_data)
        return df


class TransformerGenerator:
    def __init__(self, freq_count, window_length, stride, dtype):
        self.freq_count = freq_count
        self.window_length = window_length
        self.stride = stride
        self.dtype = dtype
        self.generate_position = 0

    def generate_next(self, source_name, windows_count):
        window = [self.window_length, windows_count, self.stride]  # [window_length,window_number,shift]
        frequencies = np.arange(0, experiment_config["max_NQ_relative_frequency"],
                                experiment_config[
                                    "max_NQ_relative_frequency"] / self.freq_count)  # frequencies relative to Nq
        function = HDSpectrogram(source_name, windows=window, param=[frequencies], dtype=self.dtype)

        print("Position: ", self.generate_position)

        function.set_reader_byte_position(self.generate_position)
        self.generate_position += function.transform_length

        return function.transform()


def how_many(a, b):
    if a > b:
        return b
    else:
        return a % b



