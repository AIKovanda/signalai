from taskchain import Parameter, InMemoryData, Task


class DatasetManipulator(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [Parameter("datasets")]

    def run(self, datasets) -> list:
        return datasets
