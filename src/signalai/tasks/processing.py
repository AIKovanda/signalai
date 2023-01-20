from pathlib import Path

from taskchain import DirData, InMemoryData, Parameter
from taskchain.task import Task

from signalai.core import TimeSeriesModel
from signalai.tasks.data_preparation import TestTimeSeriesGen, TrainTimeSeriesGen, ValidTimeSeriesGen
from signalai.time_series_gen import make_graph


class TrainModel(Task):
    class Meta:
        input_tasks = [TrainTimeSeriesGen, ValidTimeSeriesGen]
        parameters = [
            Parameter('model'),
            Parameter('generators', default={}, ignore_persistence=True),
            Parameter('transform_graph', default={}, ignore_persistence=True),
        ]

    def run(self, train_time_series_gen, valid_time_series_gen, model: TimeSeriesModel, generators: dict,
            transform_graph: dict) -> DirData:
        graph = make_graph(
            time_series_gens=generators,
            structure=transform_graph,
        )
        if 'pre_transform' in graph:
            model.set_pre_transform(graph['pre_transform'])
        if 'post_transform' in graph:
            model.set_post_transform(graph['post_transform'])
        dir_data = self.get_data_object()
        model.set_path(dir_data.dir)
        batch_history, eval_history = model.train_on_generator(
            train_time_series_gen,
            valid_time_series_gen,
        )
        batch_history.to_parquet(dir_data.dir / 'batch_history.parquet')
        eval_history.to_parquet(dir_data.dir / 'eval_history.parquet')
        return dir_data


class TrainedModel(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [TrainModel]
        parameters = [
            Parameter('model'),
            Parameter('generators', default={}, ignore_persistence=True),
            Parameter('transform_graph', default={}, ignore_persistence=True),
        ]

    def run(self, train_model, model: TimeSeriesModel, generators: dict, transform_graph: dict) -> TimeSeriesModel:
        graph = make_graph(
            time_series_gens=generators,
            structure=transform_graph,
        )
        model.save_dir = Path(train_model)
        if 'pre_transform' in graph:
            model.set_pre_transform(graph['pre_transform'])
        if 'post_transform' in graph:
            model.set_post_transform(graph['post_transform'])
        model.load(weights_path=Path(train_model) / 'saved_model' / '0' / 'epoch_last.pth')
        return model


class EvaluateModel(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [TestTimeSeriesGen, TrainedModel]
        parameters = [
            Parameter('evaluators'),
            Parameter('evaluation_params', default={}),
        ]

    def run(self, test_time_series_gen, trained_model: TimeSeriesModel, evaluators: list,
            evaluation_params: dict) -> dict:

        assert len(evaluators) > 0, "There is no evaluator!"
        return trained_model.eval_on_generator(test_time_series_gen, evaluators, evaluation_params, use_tqdm=True)


class EvaluateValidModel(Task):
    class Meta:
        data_class = InMemoryData
        input_tasks = [ValidTimeSeriesGen, TrainedModel]
        parameters = [
            Parameter('evaluators'),
            Parameter('evaluation_params', default={}),
        ]

    def run(self, valid_time_series_gen, trained_model: TimeSeriesModel, evaluators: list,
            evaluation_params: dict) -> dict:

        assert len(evaluators) > 0, "There is no evaluator!"
        return trained_model.eval_on_generator(valid_time_series_gen, evaluators, evaluation_params, use_tqdm=True)
