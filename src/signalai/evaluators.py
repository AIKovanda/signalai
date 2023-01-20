import abc

import numpy as np
import torch
from signalai.audio_transformers import STFT
from sklearn.metrics import accuracy_score, auc, precision_score, recall_score, roc_curve
from taskchain.parameter import AutoParameterObject


class TorchEvaluator(AutoParameterObject):

    def __init__(self, **params):
        self.params = params
        self.items = []

    def reset_items(self):
        self.items = []

    @abc.abstractmethod
    def process_one(self, y_true: tuple[torch.Tensor], y_pred: tuple[torch.Tensor]):
        pass

    def process_batch(self, y_true: tuple[torch.Tensor], y_pred: tuple[torch.Tensor]):
        with torch.no_grad():
            lens = {len(i) for y in [y_true, y_pred] for i in y}
            assert len(lens) == 1, 'Tensors do not have the same length'
            for i in range(lens.pop()):
                self.process_one(tuple(item[i].detach() for item in y_true), tuple(item[i] for item in y_pred))

    @property
    @abc.abstractmethod
    def metric_value(self) -> dict:
        raise NotImplementedError

    def __call__(self, y_true: tuple[torch.Tensor], y_pred: tuple[torch.Tensor]):
        self.process_batch(y_true, y_pred)


class L2PieceWise(TorchEvaluator):

    def process_one(self, y_true: tuple[torch.Tensor], y_pred: tuple[torch.Tensor]):
        self.items.append((y_true[0] - y_pred[0])**2)

    @property
    def metric_value(self) -> dict:
        stack = torch.stack(self.items, dim=0)
        return {'L2PieceWise': torch.mean(stack, dim=0).detach().cpu().numpy()}


class L1Total(TorchEvaluator):

    def process_one(self, y_true: tuple[torch.Tensor], y_pred: tuple[torch.Tensor]):
        self.items.append(torch.abs(y_true[0] - y_pred[0]))

    @property
    def metric_value(self) -> dict:
        stack = torch.stack(self.items, dim=0)
        return {'L1Total': float(torch.mean(stack))}


class L2Total(TorchEvaluator):

    def process_one(self, y_true: tuple[torch.Tensor], y_pred: tuple[torch.Tensor]):
        self.items.append((y_true[0] - y_pred[0])**2)

    @property
    def metric_value(self) -> dict:
        stack = torch.stack(self.items, dim=0)
        return {'L2Total': float(torch.mean(stack))}


class Binary(TorchEvaluator):
    def __init__(self, **params):
        super().__init__(**params)
        self.trues = []
        self.preds = []

    def reset_items(self):
        self.trues = []
        self.preds = []

    def process_one(self, y_true: tuple[torch.Tensor], y_pred: tuple[torch.Tensor]):
        self.trues.append(y_true[0].detach().cpu().numpy())
        self.preds.append(y_pred[0].detach().cpu().numpy())

    @property
    def metric_value(self) -> dict:
        true = np.concatenate(self.trues, axis=-1).reshape(-1)
        pred = np.concatenate(self.preds, axis=-1).reshape(-1)

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


class L12(TorchEvaluator):
    name = ''

    def __init__(self, **params):
        super().__init__(**params)
        self.channels_count = None
        self.l1, self.l2 = [], []

    def reset_items(self):
        self.l1, self.l2 = [], []

    def transform_channels(self, true_channel: torch.Tensor, pred_channel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return true_channel, pred_channel

    def process_one(self, y_true: tuple[torch.Tensor], y_pred: tuple[torch.Tensor]):
        if self.channels_count is None:
            self.channels_count = y_pred[0].shape[0]
        assert self.channels_count == y_pred[0].shape[0], "y_pred has a different number of channels than previous ones"
        assert self.channels_count == y_true[0].shape[0], "y_true has a different number of channels than previous y_pred"
        zero_front = self.params.get('zero_front', 0)
        zero_end = self.params.get('zero_end', 0)
        assert zero_front >= 0 and zero_end >= 0
        true = y_true[0].detach()
        pred = y_pred[0].detach()
        with torch.no_grad():
            for true_channel, pred_channel in zip(true, pred):
                if zero_front > 0:
                    pred_channel[:zero_front] = 0
                    true_channel[:zero_front] = 0
                if zero_end > 0:
                    pred_channel[-zero_end:] = 0
                    true_channel[-zero_end:] = 0
                true_channel, pred_channel = self.transform_channels(true_channel, pred_channel)
                self.l1.append(torch.unsqueeze(torch.mean(torch.abs(pred_channel - true_channel)), 0))
                self.l2.append(torch.unsqueeze(torch.mean((pred_channel - true_channel) ** 2), 0))

    @property
    def metric_value(self) -> dict[str, float]:
        with torch.no_grad():
            by_one_l1 = {f'{self.name}l1-ch{i}': float(torch.mean(torch.cat(self.l1[i::self.channels_count]))) for i in range(self.channels_count)}
            by_one_l2 = {f'{self.name}l2-ch{i}': float(torch.mean(torch.cat(self.l2[i::self.channels_count]))) for i in range(self.channels_count)}
            return {
                f'{self.name}l1-all': float(torch.mean(torch.cat(self.l1))), **by_one_l1,
                f'{self.name}l2-all': float(torch.mean(torch.cat(self.l2))), **by_one_l2,
            }


class SpectrogramL12(L12):
    name = 'SPEC_'

    def __init__(self, **params):
        super().__init__(**params)
        self._transform = STFT(phase_as_meta=True, n_fft=2048, hop_length=1024)

    def transform_channels(self, true_channel: torch.Tensor, pred_channel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(np.abs(self._transform.process_numpy(np.expand_dims(true_channel.cpu().numpy(), 0)))),
            torch.from_numpy(np.abs(self._transform.process_numpy(np.expand_dims(pred_channel.cpu().numpy(), 0)))),
        )


class MELSpectrogramL12(L12):
    name = 'MEL_'

    def transform_channels(self, true_channel: torch.Tensor, pred_channel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import librosa
        return (
            torch.from_numpy(librosa.feature.melspectrogram(y=true_channel.cpu().numpy(), sr=44100)),
            torch.from_numpy(librosa.feature.melspectrogram(y=pred_channel.cpu().numpy(), sr=44100)),
        )

