import abc
from pathlib import Path


class SignalModel(abc.ABC):
    def __init__(self, model, gen_generator, model_type, training_params, save_dir):
        super().__init__()
        self.model = model
        self.gen_generator = gen_generator
        self.model_type = model_type
        self.training_params = training_params
        self.save_dir = Path(save_dir) if save_dir is not None else None

        self.train_gen = None
        self.evaluator = None

        self.criterion = None
        self.optimizer = None

    def train_on_generator(self):
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        if self.train_gen is None:
            self.train_gen = self.gen_generator.get_generator(
                "train", log=0, batch_size=self.training_params["batch_size"]
            )
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
