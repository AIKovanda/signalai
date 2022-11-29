from taskchain import InMemoryData, Parameter, Task


class DatasetDict(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = []
        parameters = [Parameter("datasets")]

    def run(self, datasets: dict) -> dict:
        return datasets
