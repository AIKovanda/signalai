import numpy as np

from timeseries import TimeSeriesGenerator


def test_random_run():
    def params_func():
        return np.random.random()

    def run_func(x):
        return np.exp(x)

    t = TimeSeriesGenerator(params_func, run_func)
    assert t.next_item == 0
    t10 = t[10]
    assert t.next_item == 11
    t.reset_item()
    assert t.next_item == 0
    for i in range(11):
        t_value = t[i]

    assert t10 == t_value
    assert t.next_item == 11


def test_random():
    def params_func():
        return np.random.random()
    t = TimeSeriesGenerator(params_func)
    assert t.next_item == 0
    t10 = t[10]
    assert t.next_item == 11
    t.reset_item()
    assert t.next_item == 0
    for i in range(11):
        t_value = t[i]

    assert t10 == t_value
    assert t.next_item == 11
