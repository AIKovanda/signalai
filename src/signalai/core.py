import abc
from pathlib import Path


class SignalModel(abc.ABC):
    def __init__(self, model, signal_generator, model_type, training_params, save_dir):
        super().__init__()
        self.model = model
        self.signal_generator = signal_generator
        self.model_type = model_type
        self.training_params = training_params
        self.save_dir = Path(save_dir) if save_dir is not None else None

        self.evaluator = None

        self.criterion = None
        self.optimizer = None

    def train_on_generator(self):
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        if self.evaluator is None:
            eval_info = self.training_params.get("evaluator", None)
            if eval_info is not None:
                assert False
                # self.evaluator = get_instance(eval_info["class"], {
                #     "gen_gen": self.gen_generator})
        
        self._train_on_generator()

    @abc.abstractmethod
    def train_on_batch(self, x, y):
        pass

    @abc.abstractmethod
    def eval_on_batch(self, x):
        pass

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass

    @abc.abstractmethod
    def _train_on_generator(self):
        pass

    @abc.abstractmethod
    def get_optimizer(self):
        pass

    @abc.abstractmethod
    def get_criterion(self):
        pass
