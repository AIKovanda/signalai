from signalai.timeseries import SeriesDatasetsKeeper, SeriesProcessor, Logger
from taskchain import Parameter, InMemoryData, Task


class TaskSeriesProcessor(Task):
    class Meta:
        abstract = True
        input_tasks = []

    def run(self) -> SeriesProcessor:
        if self.parameters['tryout']:
            split_name = "trial"
            split_range = self.parameters['trial']
        else:
            split_name = self.meta.split_name
            split_range = self.parameters['split'].get(split_name)

        assert self.parameters.get(split_name) is not None, f"Config missing for {split_name}"
        logger = Logger(name=f"{split_name.capitalize()}SignalGenerator", verbose=0)
        keeper = SeriesDatasetsKeeper(
            datasets_config=self.parameters['datasets'],
            split_range=split_range,
            logger=logger,
        )
        if self.parameters['load_to_ram']:
            keeper.load_to_ram()

        return SeriesProcessor(
            processor_config=self.parameters[split_name],
            keeper=keeper,
            logger=logger,
        )


class TrainSeriesProcessor(TaskSeriesProcessor):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [
            Parameter("datasets"),
            Parameter("train", default=None),
            Parameter("load_to_ram", default=False, ignore_persistence=True),
            Parameter("split", default={}),
            Parameter("tryout", default=False),
            Parameter("trial", default=None, ignore_persistence=True),
        ]
        split_name = 'train'


class ValidSeriesProcessor(TaskSeriesProcessor):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [
            Parameter("datasets"),
            Parameter("valid", default=None),
            Parameter("load_to_ram", default=False, ignore_persistence=True),
            Parameter("split", default={}),
            Parameter("tryout", default=False),
            Parameter("trial", default=None, ignore_persistence=True),
        ]
        split_name = 'valid'


class TestSeriesProcessor(TaskSeriesProcessor):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [
            Parameter("datasets"),
            Parameter("test", default=None),
            Parameter("load_to_ram", default=False, ignore_persistence=True),
            Parameter("split", default={}),
            Parameter("tryout", default=False),
            Parameter("trial", default=None, ignore_persistence=True),
        ]
        split_name = 'test'

