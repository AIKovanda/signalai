from taskchain import InMemoryData, Parameter, Task

from signalai.time_series_gen import TimeSeriesGen, make_graph
from signalai.torch_dataset import TorchDataset
from signalai.tasks.datasets import DatasetDict


class TimeSeriesGenGraph(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDict]
        parameters = [
            Parameter("data_graph"),
            Parameter("generators", default={}),
        ]

    def run(self, dataset_dict: dict, generators: dict, data_graph: dict) -> dict[str, TimeSeriesGen]:
        graph = make_graph(
            time_series_gens={**dataset_dict, **generators},
            structure=data_graph,
        )
        return graph


class TaskTimeSeriesGen(Task):
    class Meta:
        abstract = True

    def run(self) -> TorchDataset:
        take_dict = self.parameters[self.meta.split_name]
        take = {val for i in ['inputs', 'outputs'] for val in take_dict[i]}
        relevant_graph = {
            key: val for key, val in self.input_tasks['time_series_gen_graph'].value.items() if key in take}
        params = {
            'take_dict': take_dict,
            'max_length': self.parameters['max_length'],
        }
        td = TorchDataset(**params).take_input(relevant_graph)
        td.build()
        return td


class TrainTimeSeriesGen(TaskTimeSeriesGen):
    class Meta:
        data_class = InMemoryData
        input_tasks = [TimeSeriesGenGraph]
        parameters = [
            Parameter("train_gen", default=None),
            Parameter("max_length", default=10**15),
        ]
        split_name = 'train_gen'


class ValidTimeSeriesGen(TaskTimeSeriesGen):
    class Meta:
        data_class = InMemoryData
        input_tasks = [TimeSeriesGenGraph]
        parameters = [
            Parameter("valid_gen", default=None),
            Parameter("max_length", default=10**15),
        ]
        split_name = 'valid_gen'


class TestTimeSeriesGen(TaskTimeSeriesGen):
    class Meta:
        data_class = InMemoryData
        input_tasks = [TimeSeriesGenGraph]
        parameters = [
            Parameter("test_gen", default=None),
            Parameter("max_length", default=10**15),
        ]
        split_name = 'test_gen'
