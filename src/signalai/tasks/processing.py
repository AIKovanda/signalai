from pathlib import Path

from taskchain import Parameter, InMemoryData, DirData
from taskchain.task import Task

from signalai.tasks.data_preparation import TrainSignalGenerator
from signalai.core import SignalModel
from signalai.torch_core import TorchSignalModel


def init_model(signal_model_config, save_dir=None, signal_generator=None, training_params=None):

    if signal_model_config['signal_model_type'] == 'torch_signal_model':
        signal_model = TorchSignalModel(
            signal_generator=signal_generator,
            training_params=training_params,
            save_dir=save_dir,
            **signal_model_config,
        )
    else:
        raise NotImplementedError(f'{signal_model_config["signal_model_type"]} type of model is not implemented yet!')
    return signal_model


class TrainModel(Task):

    class Meta:
        input_tasks = [TrainSignalGenerator]
        parameters = [
            Parameter('signal_model_config'),
            Parameter('training_params'),
        ]

    def run(self, train_signal_generator, signal_model_config, training_params) -> DirData:
        dir_data = self.get_data_object()
        signal_model = init_model(signal_model_config, dir_data.dir, train_signal_generator, training_params)

        signal_model.train_on_generator()
        return dir_data


class TrainedModel(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [TrainModel]
        parameters = [
            Parameter('signal_model_config'),
        ]

    def run(self, train_model, signal_model_config) -> SignalModel:
        signal_model = init_model(signal_model_config, save_dir=Path(train_model))
        model_path = Path(train_model) / 'saved_model' / 'epoch_last.pth'
        signal_model.load(model_path)
        return signal_model
