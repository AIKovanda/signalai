import librosa
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc

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


class Binary(SignalEvaluator):
    name = 'binary'

    def __init__(self, params=None):
        super().__init__(params)
        self.name = f'binary-{self.params.get("threshold", .5)}'

    @property
    def stat(self) -> dict[str, float]:
        print(self.name)
        mp = np.concatenate([i[0] for i in self.items], axis=1)
        pred = np.concatenate([i[1] for i in self.items], axis=1)
        th = self.params.get('threshold', .5)

        fpr, tpr, _ = roc_curve(mp.reshape(-1), pred.reshape(-1))
        roc_auc = auc(fpr, tpr)

        precision = precision_score(mp.reshape(-1) > .5, pred.reshape(-1) > th, zero_division=1)
        recall = recall_score(mp.reshape(-1) > .5, pred.reshape(-1) > th)
        return {
            'accuracy': accuracy_score(mp.reshape(-1) > .5, pred.reshape(-1) > th),
            'precision': precision,
            'recall': recall,
            'F1': 2 * precision * recall / (precision + recall),
            'roc_auc': roc_auc,
        }
