from taskchain import InMemoryData, Parameter, Task

from signalai.tasks.datasets import DatasetDict
from signalai.time_series_gen import make_graph
from signalai.torch_dataset import TorchDataset


class TaskTimeSeriesGen(Task):
    class Meta:
        abstract = True

    def run(self) -> TorchDataset:
        graph = make_graph(
            time_series_gens={**self.input_tasks['dataset_dict'].value, **self.parameters['generators']},
            structure=self.parameters[f'data_graph_{self.meta.split_name}'],
        )
        take_dict = self.parameters[f'{self.meta.split_name}_gen']
        take = {val for i in ['inputs', 'outputs'] for val in take_dict[i]}
        relevant_graph = {
            key: val for key, val in graph.items() if key in take}
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
        input_tasks = [DatasetDict]
        parameters = [
            Parameter("data_graph_train"),
            Parameter("train_gen"),
            Parameter("max_length", default=10**15),
            Parameter("generators", default={}),
        ]
        split_name = 'train'


class ValidTimeSeriesGen(TaskTimeSeriesGen):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDict]
        parameters = [
            Parameter("data_graph_valid"),
            Parameter("valid_gen"),
            Parameter("max_length", default=10**15),
            Parameter("generators", default={}),
        ]
        split_name = 'valid'


class TestTimeSeriesGen(TaskTimeSeriesGen):
    class Meta:
        data_class = InMemoryData
        input_tasks = [DatasetDict]
        parameters = [
            Parameter("data_graph_test"),
            Parameter("test_gen"),
            Parameter("max_length", default=10**15),
            Parameter("generators", default={}),
        ]
        split_name = 'test'
