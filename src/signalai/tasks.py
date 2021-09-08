import numpy as np
from taskorganizer.pipeline import PipelineTask


class InitA(PipelineTask):
    def run(self):
        return np.random.random((4, 4))


class A(PipelineTask):

    def run(self, init_a: np.ndarray, cde):
        print("a running")
        return init_a + cde


class B(PipelineTask):

    def run(self, init_a, a: np.ndarray):
        print("b running")
        return a.dot(a) + init_a
