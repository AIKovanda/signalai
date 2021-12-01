import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt


def gauss_convolve(arr, window_length, rel_std):
    window = signal.windows.gaussian(window_length, std=window_length * rel_std)
    window = window / np.sum(window)
    c = np.array([np.convolve(arr[i, :], window, mode='same') for i in range(arr.shape[0])])

    return c


def gauss_filter(arr, rel_std):
    return np.array([gaussian_filter1d(arr[:, i], rel_std) for i in range(arr.shape[1])]).T


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=34):
    nyq = 0.5 * fs
    if low_cut is None:
        sos = butter(order, high_cut / nyq, analog=False, btype='lowpass', output='sos')
    elif high_cut is None:
        sos = butter(order, low_cut / nyq, analog=False, btype='highpass', output='sos')
    else:
        sos = butter(order, [low_cut / nyq, high_cut / nyq], analog=False, btype='bandpass', output='sos')

    return sosfilt(sos, data)
