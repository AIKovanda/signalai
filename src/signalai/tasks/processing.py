from pathlib import Path

from signalai.config import DEVICE
from signalai.core import SignalModel
from signalai.tasks.data_preparation import TrainSignalGenerator

from taskchain import Parameter, InMemoryData, DirData
from taskchain.task import Task

from signalai.torch_core import TorchSignalModel


def init_model(signal_model_config, save_dir=None, gen_generator=None, training_params=None):
    if signal_model_config['signal_model_type'] == 'torch_signal_model':
        signal_model = TorchSignalModel(
            model=signal_model_config['model'].to(DEVICE),
            gen_generator=gen_generator,
            model_type=signal_model_config['signal_model_type'],
            training_params=training_params,
            save_dir=save_dir
        )
    else:
        raise NotImplementedError(f'{signal_model_config["signal_model_type"]} type of model is not implemented yet!')
    return signal_model


class TrainModel(Task):

    class Meta:
        input_tasks = [TrainSignalGenerator]
        parameters = [
            Parameter('signal_model_config'),
            Parameter('training_params')
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
        signal_model = init_model(signal_model_config)
        model_path = Path(train_model) / 'saved_model' / 'last.pth'
        signal_model.load(model_path)
        return signal_model
