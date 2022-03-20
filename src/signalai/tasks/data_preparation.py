from signalai.tasks.datasets import DatasetManipulator
from signalai.timeseries import SeriesDatasetsKeeper, SeriesProcessor, Logger
from taskchain import Parameter, InMemoryData, Task


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
        if self.parameters['load_to_ram']:
            keeper.load_to_ram()

        return keeper


class TaskSeriesProcessor(Task):
    class Meta:
        abstract = True

    def run(self) -> SeriesProcessor:
        split_name = self.meta.split_name
        logger = Logger(name=f"{split_name.capitalize()}SignalGenerator", verbose=0, save=False)

        return SeriesProcessor(
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

