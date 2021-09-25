import numpy as np
from taskorganizer.pipeline import PipelineTask



class TrainModel(PipelineTask):

    def run(self, data_generator, cde: np.ndarray):
        print("TrainModel running")
        return data_generator.dot(data_generator) + cde


class TrainedModel(PipelineTask):

    def run(self, train_model):
        print("TrainedModel running")
        return train_model
