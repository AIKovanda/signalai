from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfilt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


def gauss_convolve(arr, window_length, rel_std):
    window = signal.windows.gaussian(window_length, std=window_length * rel_std)
    window = window / np.sum(window)
    c = np.array([np.convolve(arr[i, :], window, mode='same') for i in range(arr.shape[0])])

    return c


def gauss_filter(arr, rel_std):
    d = np.array([gaussian_filter1d(arr[:, i], rel_std) for i in range(arr.shape[1])])
    # from scipy.signal import savgol_filter
    # yhat = savgol_filter(y, 51, 3)
    return d.T


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, bandpass_params, order=5):
    lowcut, highcut, fs = bandpass_params
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def bandpass_ifft(X, bandpass_params, m=None):
    """Bandpass filtering on a real signal using inverse FFT

    Inputs
    =======

    X: 1-D numpy array of floats, the real time domain signal (time series) to be filtered
    lowcut: float, frequency components below this frequency will not pass the filter (physical frequency in unit of Hz)
    highcut: float, frequency components above this frequency will not pass the filter (physical frequency in unit of Hz)
    fs: float, the sampling frequency of the signal (physical frequency in unit of Hz)

    Notes
    =====
    1. The input signal must be real, not imaginary nor complex
    2. The filtered_signal will have only half of original amplitude. Use abs() to restore.
    3. In Numpy/Scipy, the frequencies goes from 0 to fs/2 and then from negative fs to 0.

    """
    lowcut, highcut, fs = bandpass_params
    shape = X.shape
    X = X.reshape((int(np.prod(shape)),))
    if m is None:
        m = X.size

    spectrum = scipy.fft.fft(X, n=m)
    [lowcut, highcut, fs] = map(float, [lowcut, highcut, fs])
    [low_point, high_point] = map(lambda F: F / fs * m / 2, [lowcut, highcut])

    filtered_spectrum = [spectrum[i] if low_point <= i <= high_point else 0.0 for i in range(m)]  # Filtering
    filtered_signal = scipy.fft.ifft(filtered_spectrum, n=m)  # Construct filtered signal
    return np.real(filtered_signal).reshape(shape) * 2


if __name__ == '__main__':
    data = np.load('/home/martin/Data/Datasets/201103_AE_ch1.npy')
    fs = 1562500
    lowcut = 500000
    highcut = 1562500 / 2 - 1

    Filtered_signal = bandpass_ifft(data[:128 ** 2], lowcut, highcut, fs)
    Filtered_signal2 = bandpass_ifft(data[:128 ** 2], 1, lowcut, fs)
    # transformed = butter_bandpass_filter(data[:219375], lowcut, highcut, fs)

    # transformed2 = butter_bandpass_filter(data[:20000], 1, lowcut, fs)

    plt.figure()
    rr = 1000
    sns.lineplot(x=range(rr), y=data[:rr])
    sns.lineplot(x=range(rr), y=Filtered_signal[:rr] * 2)
    sns.lineplot(x=range(rr), y=Filtered_signal2[:rr] * 2)
    # sns.lineplot(x=range(rr), y=transformed[:rr]+transformed2[:rr])
    # sns.lineplot(x=range(rr), y=transformed[:rr])
    plt.show()

    # b, a = butter_bandpass(lowcut, highcut, fs)
    # w, h = freqz(b, a, worN=2000)
    # plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
