import librosa
import numpy as np
from signalai.transformers import STFT
from taskchain.parameter import AutoParameterObject


class SignalEvaluator(AutoParameterObject):
    name = 'evaluator'

    def __init__(self, params=None):
        if params is None:
            params = {}
        self.params = params
        self.items = []

    def add_item(self, y_hat, y_true):
        self.items.append((y_hat, y_true))

    def set_items(self, items):
        self.items = items

    @property
    def stat(self):
        raise NotImplementedError


class ItemsEcho(SignalEvaluator):
    name = 'items'

    @property
    def stat(self) -> list:
        return self.items


class SpectrogramL1(SignalEvaluator):
    name = 'spectrogram_L1'

    @property
    def stat(self) -> dict[str, float]:
        print(self.name)
        l1 = []
        period = None
        stft = STFT(phase_as_meta=True, n_fft=2048, hop_length=1024)
        for y_pred, y_true in self.items:
            period = len(y_pred)
            for pred_channel, true_channel in zip(y_pred, y_true):
                pred_arr = stft(pred_channel).data_arr
                true_arr = stft(true_channel).data_arr
                l1.append(np.sum(np.abs(pred_arr - true_arr)))

        by_one = {str(i): float(np.mean(l1[i::period])) for i in range(period)}
        return {
            'all': float(np.mean(l1)),
            **by_one
        }


class SpectrogramL2(SignalEvaluator):
    name = 'spectrogram_L2'

    @property
    def stat(self) -> dict[str, float]:
        print(self.name)
        l2 = []
        period = None
        stft = STFT(phase_as_meta=True, n_fft=2048, hop_length=1024)
        for y_pred, y_true in self.items:
            period = len(y_pred)
            for pred_channel, true_channel in zip(y_pred, y_true):
                pred_arr = stft(pred_channel).data_arr
                true_arr = stft(true_channel).data_arr
                l2.append(np.sum((pred_arr - true_arr) ** 2))

        by_one = {str(i): float(np.mean(l2[i::period])) for i in range(period)}
        return {
            'all': float(np.mean(l2)),
            **by_one
        }


class MELSpectrogramL1(SignalEvaluator):
    name = 'mel_spectrogram_L1'

    @property
    def stat(self) -> dict[str, float]:
        print(self.name)
        l1 = []
        period = None
        for y_pred, y_true in self.items:
            period = len(y_pred)
            for pred_channel, true_channel in zip(y_pred, y_true):
                pred_arr = librosa.feature.melspectrogram(y=pred_channel, sr=44100)
                true_arr = librosa.feature.melspectrogram(y=true_channel, sr=44100)
                l1.append(np.sum(np.abs(pred_arr - true_arr)))

        by_one = {str(i): float(np.mean(l1[i::period])) for i in range(period)}
        return {
            'all': float(np.mean(l1)),
            **by_one
        }


class MELSpectrogramL2(SignalEvaluator):
    name = 'mel_spectrogram_L2'

    @property
    def stat(self) -> dict[str, float]:
        print(self.name)
        l2 = []
        period = None
        for y_pred, y_true in self.items:
            period = len(y_pred)
            for pred_channel, true_channel in zip(y_pred, y_true):
                pred_arr = librosa.feature.melspectrogram(y=pred_channel, sr=44100)
                true_arr = librosa.feature.melspectrogram(y=true_channel, sr=44100)
                l2.append(np.sum((pred_arr - true_arr) ** 2))

        by_one = {str(i): float(np.mean(l2[i::period])) for i in range(period)}
        return {
            'all': float(np.mean(l2)),
            **by_one
        }
