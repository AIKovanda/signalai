from pathlib import Path

from taskchain import DirData, InMemoryData, Parameter
from taskchain.task import Task
from tqdm import tqdm

from signalai.core import SignalModel
from signalai.evaluators import SignalEvaluator
from signalai.tasks.data_preparation import (TestSeriesProcessor,
                                             TrainSeriesProcessor)
from signalai.torch_core import TorchSignalModel


def init_model(signal_model_config, save_dir=None, processing_fs=None):

    if signal_model_config['signal_model_type'] == 'torch_signal_model':
        signal_model = TorchSignalModel(
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
            Parameter('early_stopping_at', default=None),
            Parameter('early_stopping_min', default=0, dont_persist_default_value=True),
            Parameter('early_stopping_regression', default=None),
            Parameter('batch_size', default=1),
            Parameter('echo_step', default=500, ignore_persistence=True),
            Parameter('save_step', default=1000, ignore_persistence=True),
            Parameter('average_losses_to_print', default=100, ignore_persistence=True),
            Parameter('loss_lambda', default='lambda x, y, crit: crit(x, y)'),
            Parameter('test', default=False),
            Parameter('processing_fs', default=None),
        ]

    def run(self,
            train_series_processor, signal_model_config,
            batches, criterion, optimizer, early_stopping_at, early_stopping_min, early_stopping_regression,
            batch_size, echo_step, save_step, average_losses_to_print, loss_lambda, test, processing_fs,
            ) -> DirData:
        if test:
            batches = 10
        train_series_processor.load_to_ram(purpose='train')
        train_series_processor.load_to_ram(purpose='valid')
        train_series_processor.set_processing_fs(processing_fs)
        dir_data = self.get_data_object()
        signal_model = init_model(
            signal_model_config, dir_data.dir, processing_fs=processing_fs,
        )
        training_params = {
            'batch_size': batch_size,
            'batches': batches,
            'echo_step': echo_step,
            'save_step': save_step,
            'average_losses_to_print': average_losses_to_print,
            'loss_lambda': loss_lambda,
        }
        signal_model.set_criterion(criterion)
        signal_model.set_optimizer(optimizer)

        signal_model.train_on_generator(
            train_series_processor,
            training_params=training_params,
            early_stopping_at=early_stopping_at,
            early_stopping_min=early_stopping_min,
            early_stopping_regression=early_stopping_regression,
        )
        train_series_processor.free_ram(purpose='train')
        train_series_processor.free_ram(purpose='valid')
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
        model_path = Path(train_model) / 'saved_model' / '0' / 'epoch_last.pth'
        signal_model.load(model_path)
        return signal_model


class EvaluateModel(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [TestSeriesProcessor, TrainedModel]
        parameters = [
            Parameter('evaluators'),
            Parameter('eval_batches', default=None),
            Parameter('eval_batch_size', default=1),
            Parameter('eval_post_transform', default=True),
            Parameter('processing_fs', default=None),
        ]

    def run(self, test_series_processor, trained_model: SignalModel, evaluators: list[SignalEvaluator],
            eval_batches, eval_batch_size, eval_post_transform, processing_fs) -> dict:

        assert len(evaluators) > 0, "There is no evaluator!"
        test_series_processor.load_to_ram(purpose='test')
        test_series_processor.set_processing_fs(processing_fs)
        evaluation_params = {
            'batch_size': eval_batch_size,
            'batches': eval_batches,
        }
        items = list(tqdm(
                trained_model.eval_on_generator(
                    test_series_processor, evaluation_params, post_transform=eval_post_transform,
                ), total=eval_batches*eval_batch_size))

        test_series_processor.free_ram(purpose='test')

        for evaluator in evaluators:
            evaluator.set_items(items)

        return {evaluator.name: evaluator.stat for evaluator in evaluators}


class EchoInfo(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [
            Parameter("signal_model_config"),
            Parameter("processing_fs", default=None),
        ]

    def run(self, processing_fs, signal_model_config) -> tuple[int, int]:
        return signal_model_config['target_signal_length'], processing_fs
