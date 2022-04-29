import librosa
import numpy as np
import torch
from scipy.signal import convolve2d
from sklearn.metrics import (accuracy_score, auc, precision_score,
                             recall_score, roc_curve)
from taskchain.parameter import AutoParameterObject
from taskchain.task import Config

from signalai import config
from signalai.transformers import STFT


class SignalEvaluator(AutoParameterObject):
    name = 'evaluator'

    def __init__(self, **params):
        self.params = params
        self.items = []
        self._transform = None

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


class L12(SignalEvaluator):
    name = 'L12'

    def _transform_channels(self, pred_channel: np.ndarray, true_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return pred_channel, true_channel

    @property
    def stat(self) -> dict[str, dict[str, float]]:
        print(self.name)
        l1 = []
        l2 = []
        period = None
        zero_front = self.params.get('zero_front', 0)
        zero_end = self.params.get('zero_end', 0)
        for y_true, y_pred in self.items:
            period = len(y_pred)
            for pred_channel, true_channel in zip(y_pred, y_true):
                pred_channel[:zero_front] = 0
                true_channel[:zero_front] = 0
                pred_channel[-zero_end-1:] = 0
                true_channel[-zero_end-1:] = 0
                pred_arr, true_arr = self._transform_channels(pred_channel, true_channel)
                l1.append(np.mean(np.abs(pred_arr - true_arr)))
                l2.append(np.mean((pred_arr - true_arr) ** 2))

        by_one_l1 = {str(i): float(np.mean(l1[i::period])) for i in range(period)}
        by_one_l2 = {str(i): float(np.mean(l2[i::period])) for i in range(period)}
        return {
            'l1': {'all': float(np.mean(l1)), **by_one_l1},
            'l2': {'all': float(np.mean(l2)), **by_one_l2},
        }


class SpectrogramL12(L12):
    name = 'spectrogram_L12'

    def __init__(self, **params):
        super().__init__(**params)
        self._transform = STFT(phase_as_meta=True, n_fft=2048, hop_length=1024)

    def _transform_channels(self, pred_channel: np.ndarray, true_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._transform(pred_channel).data_arr, self._transform(true_channel).data_arr


class MELSpectrogramL12(L12):
    name = 'mel_spectrogram_L12'

    def _transform_channels(self, pred_channel: np.ndarray, true_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (librosa.feature.melspectrogram(y=pred_channel, sr=44100),
                librosa.feature.melspectrogram(y=true_channel, sr=44100))


class AutoEncoderL12(L12):
    name = 'AutoEncoderL12'

    def __init__(self, **params):
        super().__init__(**params)
        config_path = config.CONFIGS_DIR / 'models' / 'autoencoder' / '1.yaml'

        conf = Config(
            config.TASKS_DIR,  # where Taskchain data should be stored
            config_path,
            global_vars=config,  # set global variables
        )
        chain = conf.chain()
        chain.set_log_level('CRITICAL')

        signal_model = chain.trained_model.value
        signal_model.load(batch='last')

        self._transform = signal_model.model.encoder
        self._transform.to(config.DEVICE).eval()

    def _transform_channels(self, pred_channel: np.ndarray, true_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            x = torch.from_numpy(pred_channel).type(torch.float32).unsqueeze(0).unsqueeze(0).to(config.DEVICE)
            y = torch.from_numpy(true_channel).type(torch.float32).unsqueeze(0).unsqueeze(0).to(config.DEVICE)
            return self._transform(x)[0][0].detach().cpu().numpy(), self._transform(y)[0][0].detach().cpu().numpy()


class Binary(SignalEvaluator):
    name = 'binary'

    def __init__(self, **params):
        super().__init__(**params)
        self.name = f'binary-t{self.params.get("threshold", .5)}'

    @property
    def stat(self) -> dict[str, float]:
        print(self.name)
        true = np.concatenate([i[0] for i in self.items], axis=1).reshape(-1)
        pred = np.concatenate([i[1] for i in self.items], axis=1).reshape(-1)
        th = self.params.get('threshold', .5)

        fpr, tpr, _ = roc_curve(true, pred)
        roc_auc = auc(fpr, tpr)

        true_map = true > .5
        pred_map = pred > th

        precision = precision_score(true_map, pred_map, zero_division=1)
        recall = recall_score(true_map, pred_map)
        return {
            'accuracy': accuracy_score(true_map, pred_map),
            'precision': precision,
            'recall': recall,
            'F1': 2 * precision * recall / (precision + recall),
            'roc_auc': roc_auc,
        }


class EBinary(Binary):
    name = 'e-binary'

    def __init__(self, **params):
        super().__init__(**params)
        self.name = (f'e-binary-t{self.params.get("threshold", .5)}-' 
                     f's{self.params.get("size", 5)}t{self.params.get("conv_threshold", 3)}')

    @property
    def stat(self) -> dict[str, float]:
        print(self.name)
        true = np.concatenate([i[0] for i in self.items], axis=1)
        pred = np.concatenate([i[1] for i in self.items], axis=1)
        th = self.params.get('threshold', .5)
        size = self.params.get("size", 5)
        conv_threshold = self.params.get("conv_threshold", 3)

        pred_map = convolve2d((pred > th).astype(int), np.ones((1, size)), 'same').reshape(-1) >= conv_threshold
        true_map = true.reshape(-1) > .5

        precision = precision_score(true_map, pred_map, zero_division=1)
        recall = recall_score(true_map, pred_map)

        return {
            'accuracy': accuracy_score(true_map, pred_map),
            'precision': precision,
            'recall': recall,
            'F1': 2 * precision * recall / (precision + recall),
        }
