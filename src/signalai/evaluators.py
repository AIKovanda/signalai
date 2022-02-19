from taskchain.parameter import AutoParameterObject


class SignalEvaluator(AutoParameterObject):
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.params = params
        self.items = []

    def add_item(self, y_hat, y_true):
        self.items.append((y_hat, y_true))

    def add_batch(self, y_hat_batch, y_true_batch):
        for (y_hat, y_true) in zip(y_hat_batch, y_true_batch):
            self.items.append((y_hat, y_true))

    @property
    def stat(self) -> dict:
        raise NotImplementedError


class EventStat(SignalEvaluator):
    @property
    def stat(self) -> dict:
        return {'data': self.items}

