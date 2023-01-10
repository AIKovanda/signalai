import abc

import numpy as np
import torch
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
        # print(f'{y_true=}\n{y_pred=}\n{(y_true[0] - y_pred[0])**2}')
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


class Accuracy(TorchEvaluator):

    def process_one(self, y_true: tuple[torch.Tensor], y_pred: tuple[torch.Tensor]):
        self.items.append(int(torch.abs(y_true[0] - y_pred[0]) < 0.5))

    @property
    def metric_value(self) -> dict:
        return {'Acc': float(np.mean(self.items))}


# class ItemsEcho(SignalEvaluator):
#     name = 'items'
#
#     @property
#     def stat(self) -> list:
#         return self.items


# class L12(SignalEvaluator):
#     name = 'L12'
#
#     def _transform_channels(self, pred_channel: np.ndarray, true_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         return pred_channel, true_channel
#
#     @property
#     def stat(self) -> dict[str, dict[str, float]]:
#         print(self.name)
#         l1 = []
#         l2 = []
#         period = None
#         zero_front = self.params.get('zero_front', 0)
#         zero_end = self.params.get('zero_end', 0)
#         for y_true, y_pred in self.items:
#             period = len(y_pred)
#             for pred_channel, true_channel in zip(y_pred, y_true):
#                 pred_channel[:zero_front] = 0
#                 true_channel[:zero_front] = 0
#                 pred_channel[-zero_end-1:] = 0
#                 true_channel[-zero_end-1:] = 0
#                 pred_arr, true_arr = self._transform_channels(pred_channel, true_channel)
#                 l1.append(np.mean(np.abs(pred_arr - true_arr)))
#                 l2.append(np.mean((pred_arr - true_arr) ** 2))
#
#         by_one_l1 = {str(i): float(np.mean(l1[i::period])) for i in range(period)}
#         by_one_l2 = {str(i): float(np.mean(l2[i::period])) for i in range(period)}
#         return {
#             'l1': {'all': float(np.mean(l1)), **by_one_l1},
#             'l2': {'all': float(np.mean(l2)), **by_one_l2},
#         }
#
#
# class SpectrogramL12(L12):
#     name = 'spectrogram_L12'
#
#     def __init__(self, **params):
#         super().__init__(**params)
#         self._transform = STFT(phase_as_meta=True, n_fft=2048, hop_length=1024)
#
#     def _transform_channels(self, pred_channel: np.ndarray, true_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         return self._transform(pred_channel).data_arr, self._transform(true_channel).data_arr
#
#
# class MELSpectrogramL12(L12):
#     name = 'mel_spectrogram_L12'
#
#     def _transform_channels(self, pred_channel: np.ndarray, true_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         return (librosa.feature.melspectrogram(y=pred_channel, sr=44100),
#                 librosa.feature.melspectrogram(y=true_channel, sr=44100))
#
#
# class AutoEncoderL12(L12):
#     name = 'AutoEncoderL12'
#
#     def __init__(self, **params):
#         super().__init__(**params)
#         config_path = config.CONFIGS_DIR / 'models' / 'autoencoder' / '4.yaml'
#
#         conf = Config(
#             config.TASKS_DIR,  # where Taskchain data should be stored
#             config_path,
#             global_vars=config,  # set global variables
#         )
#         chain = conf.chain()
#         chain.set_log_level('CRITICAL')
#
#         signal_model = chain.trained_model.value
#         signal_model.load(batch='last')
#
#         self._transform = signal_model.model.encoder
#         self._transform.to(config.DEVICE).eval()
#
#     def _transform_channels(self, pred_channel: np.ndarray, true_channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#         with torch.no_grad():
#             x = torch.from_numpy(pred_channel).type(torch.float32).unsqueeze(0).unsqueeze(0).to(config.DEVICE)
#             y = torch.from_numpy(true_channel).type(torch.float32).unsqueeze(0).unsqueeze(0).to(config.DEVICE)
#             return self._transform(x)[0][0].detach().cpu().numpy(), self._transform(y)[0][0].detach().cpu().numpy()
#

class Binary(TorchEvaluator):
    def __init__(self, **params):
        super().__init__(**params)
        self.trues = []
        self.preds = []

    def process_one(self, y_true: tuple[torch.Tensor], y_pred: tuple[torch.Tensor]):
        self.trues.append(y_true[0].detach().cpu().numpy())
        self.preds.append(y_pred[0].detach().cpu().numpy())

    @property
    def metric_value(self) -> dict:
        true = np.concatenate(self.trues, axis=1).reshape(-1)
        pred = np.concatenate(self.preds, axis=1).reshape(-1)
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


# class EBinary(Binary):
#     name = 'e-binary'
#
#     def __init__(self, **params):
#         super().__init__(**params)
#         self.name = (f'e-binary-t{self.params.get("threshold", .5)}-'
#                      f's{self.params.get("size", 5)}t{self.params.get("conv_threshold", 3)}')
#
#     @property
#     def stat(self) -> dict[str, float]:
#         print(self.name)
#         true = np.concatenate([i[0] for i in self.items], axis=1)
#         pred = np.concatenate([i[1] for i in self.items], axis=1)
#         th = self.params.get('threshold', .5)
#         size = self.params.get("size", 5)
#         conv_threshold = self.params.get("conv_threshold", 3)
#
#         pred_map = convolve2d((pred > th).astype(int), np.ones((1, size)), 'same').reshape(-1) >= conv_threshold
#         true_map = true.reshape(-1) > .5
#
#         precision = precision_score(true_map, pred_map, zero_division=1)
#         recall = recall_score(true_map, pred_map)
#
#         return {
#             'accuracy': accuracy_score(true_map, pred_map),
#             'precision': precision,
#             'recall': recall,
#             'F1': 2 * precision * recall / (precision + recall),
#         }
