from pathlib import Path

from taskchain import DirData, InMemoryData, Parameter
from taskchain.task import Task
from tqdm import tqdm

from signalai.core import TimeSeriesModel
from signalai.tasks.data_preparation import TestTimeSeriesGen, TrainTimeSeriesGen
from signalai.time_series_gen import make_graph


class TrainModel(Task):
    class Meta:
        input_tasks = [TrainTimeSeriesGen]
        parameters = [
            Parameter('model'),
        ]

    def run(self, train_time_series_gen, model: TimeSeriesModel) -> DirData:
        dir_data = self.get_data_object()
        model.set_path(dir_data.dir)
        model.train_on_generator(train_time_series_gen).to_parquet(dir_data.dir / 'metric_history.parquet')
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
            Parameter('eval_batches', default=None),
            Parameter('eval_batch_size', default=1),
            Parameter('eval_post_transform', default=True),
            Parameter('processing_fs', default=None),
        ]

    def run(self, test_time_series_gen, trained_model: TimeSeriesModel, evaluators,
            eval_batches, eval_batch_size, eval_post_transform, processing_fs) -> dict:

        assert len(evaluators) > 0, "There is no evaluator!"
        test_time_series_gen.load_to_ram(purpose='test')
        test_time_series_gen.set_processing_fs(processing_fs)
        evaluation_params = {
            'batch_size': eval_batch_size,
            'batches': eval_batches,
        }
        items = list(tqdm(
                trained_model.eval_on_generator(
                    test_time_series_gen, evaluation_params, post_transform=eval_post_transform,
                ), total=eval_batches*eval_batch_size))

        test_time_series_gen.free_ram(purpose='test')

        for evaluator in evaluators:
            evaluator.set_items(items)

        return {evaluator.name: evaluator.stat for evaluator in evaluators}
