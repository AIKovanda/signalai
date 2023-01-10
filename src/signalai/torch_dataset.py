import numpy as np
from torch.utils.data import Dataset

from signalai.time_series_gen import Transformer


class TorchDataset(Dataset, Transformer):
    takes = 'dict'

    def _process(self, input_: dict[str, np.ndarray]) -> list[tuple[np.ndarray], tuple[np.ndarray]]:
        return [
            tuple(input_[key] for key in self.config['take_dict'][purpose])
            for purpose in ['inputs', 'outputs']
        ]

    def transform_taken_length(self, length: int) -> int:
        return length

    def is_infinite(self) -> bool:
        return False

    def _build(self):
        assert 'take_dict' in self.config

    def __len__(self):
        try:
            return min(self.config['max_length'], super().__len__())
        except ValueError:
            return self.config['max_length']

    def __getitem__(self, idx):
        return self._getitem(idx)
