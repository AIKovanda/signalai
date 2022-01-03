from pathlib import Path

from taskchain import Parameter, InMemoryData, DirData
from taskchain.task import Task

from signalai.tasks.data_preparation import TrainSeriesProcessor
from signalai.core import SignalModel
from signalai.torch_core import TorchSignalModel


def init_model(signal_model_config, save_dir=None, series_processor=None, processing_fs=44100, **training_params):

    if signal_model_config['signal_model_type'] == 'torch_signal_model':
        signal_model = TorchSignalModel(
            series_processor=series_processor,
            training_params=training_params,
            save_dir=save_dir,
            processing_fs=processing_fs,
            **signal_model_config,
        )
    else:
        raise NotImplementedError(f'{signal_model_config["signal_model_type"]} type of model is not implemented yet!')
    return signal_model


class TrainModel(Task):

    class Meta:
        input_tasks = [TrainSeriesProcessor]
        parameters = [
            Parameter('signal_model_config'),
            Parameter('batches'),
            Parameter('criterion'),
            Parameter('optimizer'),
            Parameter('batch_size', default=1, ignore_persistence=True),
            Parameter('echo_step', default=500, ignore_persistence=True),
            Parameter('save_step', default=1000, ignore_persistence=True),
            Parameter('average_losses_to_print', default=100, ignore_persistence=True),
            Parameter('loss_lambda', default='lambda x, y, crit: crit(x, y)'),
            Parameter('test', default=False),
            Parameter('processing_fs', default=None),
        ]

    def run(self,
            train_series_processor, signal_model_config,
            batches, criterion, optimizer,
            batch_size, echo_step, save_step, average_losses_to_print, loss_lambda, test, processing_fs,
            ) -> DirData:
        if test:
            batches = 10
        train_series_processor.set_processing_fs(processing_fs)
        dir_data = self.get_data_object()
        signal_model = init_model(
            signal_model_config, dir_data.dir, train_series_processor, processing_fs=processing_fs,
            batch_size=batch_size, batches=batches, echo_step=echo_step, save_step=save_step,
            average_losses_to_print=average_losses_to_print, loss_lambda=loss_lambda,
            criterion=criterion, optimizer=optimizer,
        )

        signal_model.train_on_generator()
        return dir_data


class TrainedModel(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [TrainModel]
        parameters = [
            Parameter('signal_model_config'),
            Parameter('processing_fs', default=None),
        ]

    def run(self, train_model, signal_model_config, processing_fs) -> SignalModel:
        signal_model = init_model(signal_model_config, save_dir=Path(train_model), processing_fs=processing_fs)
        model_path = Path(train_model) / 'saved_model' / 'epoch_last.pth'
        signal_model.load(model_path)
        return signal_model
