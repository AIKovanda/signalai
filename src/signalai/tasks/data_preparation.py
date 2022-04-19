from taskchain import InMemoryData, Parameter, Task

from signalai.tasks.datasets import DatasetManipulator
from signalai.timeseries import Logger, SeriesDatasetsKeeper, TorchDataset


class KeeperLoader(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetManipulator]
        parameters = [
            Parameter("load_to_ram", default=False, ignore_persistence=True),
        ]

    def run(self) -> SeriesDatasetsKeeper:
        logger = Logger(name="KeeperLoader", verbose=0, save=False)
        keeper = SeriesDatasetsKeeper(
            datasets_config=self.input_tasks['dataset_manipulator'].value,
            logger=logger,
        )

        return keeper


class TaskSeriesProcessor(Task):
    class Meta:
        abstract = True

    def run(self) -> TorchDataset:
        split_name = self.meta.split_name
        logger = Logger(name=f"{split_name.capitalize()}SignalGenerator", verbose=0, save=False)

        return TorchDataset(
            processor_config=self.parameters[split_name],
            keeper=self.input_tasks['keeper_loader'].value,
            logger=logger,
        )


class TrainSeriesProcessor(TaskSeriesProcessor):
    class Meta:
        data_class = InMemoryData
        input_tasks = [KeeperLoader]
        parameters = [
            Parameter("train", default=None),
            Parameter("load_to_ram", default=False, ignore_persistence=True),
        ]
        split_name = 'train'


class ValidSeriesProcessor(TaskSeriesProcessor):
    class Meta:
        data_class = InMemoryData
        input_tasks = [KeeperLoader]
        parameters = [
            Parameter("valid", default=None),
            Parameter("load_to_ram", default=False, ignore_persistence=True),
        ]
        split_name = 'valid'


class TestSeriesProcessor(TaskSeriesProcessor):
    class Meta:
        data_class = InMemoryData
        input_tasks = [KeeperLoader]
        parameters = [
            Parameter("test", default=None),
            Parameter("load_to_ram", default=False, ignore_persistence=True),
        ]
        split_name = 'test'
