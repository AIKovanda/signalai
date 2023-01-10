from functools import lru_cache

import numpy as np
import yaml

from time_series import TimeSeries, Signal
from signalai.time_series_gen import TimeSeriesSum, TimeSeriesGen, LambdaTransformer, make_graph, TimeSeriesHolder


class SinusGenerator(TimeSeriesGen):
    def _next_epoch(self):
        self.config['epoch'] = 1 + self.config.get('epoch', 0)

    def _build(self):
        assert 'linspace_params' in self.config
        assert 'amplitude' in self.config
        assert 'built' not in self.config
        self.config['built'] = True
        self.history = []

    def __len__(self):
        assert ValueError

    def _getitem(self, item: int) -> TimeSeries:
        if len(self.history) > 0:
            assert self.history[-1] != item
        self.history.append(item)
        return Signal(
            data_arr=self.config['amplitude'] * np.sin(item+np.linspace(*self.config['linspace_params'])),
            fs=self.config.get('fs'),
        )

    def is_infinite(self) -> bool:
        return True

    def __call__(self, input_):
        raise ValueError('SinusGenerator does not take inputs.')


class FiniteSinusGenerator(SinusGenerator):
    def __len__(self):
        return self.config['len_']

    def _build(self):
        assert 'len_' in self.config
        super()._build()

    def is_infinite(self) -> bool:
        return False

    def _getitem(self, item: int) -> TimeSeries:
        if len(self.history) > 0:
            assert self.history[-1] != item
        self.history.append(item)
        return Signal(
            data_arr=self.config['amplitude'] * np.sin(item+np.linspace(*self.config['linspace_params'])),
            fs=self.config.get('fs'),
        )


class NotUsed(FiniteSinusGenerator):
    def _build(self):
        assert False, 'This should not be run'


def test_graph():
    tsgs = {
        'sg0': SinusGenerator(linspace_params=[0, 4, 100], amplitude=2),
        'sg1': FiniteSinusGenerator(linspace_params=[0, 4, 100], amplitude=1, len_=50),
        'l0': LambdaTransformer(lambda_w='w*2+1'),
        'l1': LambdaTransformer(lambda_w='w**2'),
        'sum': TimeSeriesSum(),
        'not_used': NotUsed(),
    }
    cfg = yaml.load("""
        =Z:
          l1:
            l0:
              sg1
        =K:
          sg0
        =X:
          sum:
            - Z
            - K
        =Y:
          l1:
            Z
        """, yaml.FullLoader)

    graph = make_graph(tsgs, cfg)
    assert 'Z' in graph and isinstance(graph['Z'], TimeSeriesGen)
    assert 'X' in graph and isinstance(graph['X'], TimeSeriesGen)
    x_gen = graph['X']
    y_gen = graph['Y']
    k_gen = graph['K']
    z_gen = graph['Z']
    x_gen.build()
    x_gen.build()
    tsgs['l0'].config['lambda'] = 'None'
    tsgs['l1'].config['lambda'] = 'None'
    assert k_gen.is_infinite()
    assert not x_gen.is_infinite()
    assert len(x_gen) == 50
    assert z_gen.getitem(0) == Signal(data_arr=(np.sin(np.linspace(0, 4, 100))*2+1)**2)
    assert x_gen.getitem(0) == Signal(data_arr=(np.sin(np.linspace(0, 4, 100))*2+1)**2+2*np.sin(np.linspace(0, 4, 100)))
    assert z_gen.getitem(0) == Signal(data_arr=(np.sin(np.linspace(0, 4, 100))*2+1)**2)
    assert y_gen.getitem(0) == Signal(data_arr=((np.sin(np.linspace(0, 4, 100))*2+1)**2)**2)
    assert tsgs['sg1'].history == [0]

    assert y_gen.getitem(1) == Signal(data_arr=((np.sin(1+np.linspace(0, 4, 100))*2+1)**2)**2)
    assert x_gen.getitem(1) == Signal(data_arr=(np.sin(1+np.linspace(0, 4, 100))*2+1)**2+2*np.sin(1+np.linspace(0, 4, 100)))
    assert y_gen.getitem(0) == Signal(data_arr=((np.sin(np.linspace(0, 4, 100))*2+1)**2)**2)
    assert tsgs['sg1'].history == [0, 1, 0]
    assert 'epoch' not in tsgs['sg0'].config
    x_gen.next_epoch()
    assert tsgs['sg0'].config['epoch'] == 1


def test_transformer():
    tsgs = {
        'l0': LambdaTransformer(lambda_w='w*2+1'),
        'l1': LambdaTransformer(lambda_w='np.sum(w.data_arr/2)', apply_to_ts=True),
    }
    cfg = yaml.load("""
        =Z:
          l1:
            l0
        """, yaml.FullLoader)

    graph = make_graph(tsgs, cfg)
    assert 'Z' in graph and isinstance(graph['Z'], TimeSeriesGen)
    z_gen = graph['Z']
    s = TimeSeries(data_arr=np.array([[2, 2.5],
                                      [1, 2.5]]),
                   time_map=np.array([[0, 1],
                                      [0, 0]]),
                   meta={'a': 1, 'b': 'asdf'}, fs=50)
    assert z_gen.process(s) == 10.


def test_holder():
    holder = TimeSeriesHolder(timeseries=[
        Signal(data_arr=np.arange(0, 10)+0.1, meta={'a': 0}),
        Signal(data_arr=np.arange(0, 17)+0.2, meta={'a': 1}),
        Signal(data_arr=np.arange(0, 13)+0.3, meta={'a': 2}),
    ])
    holder.build()
    assert holder.index_list == [(0, None), (1, None), (2, None)]
    assert len(holder) == 3
    assert holder.getitem(2) == Signal(data_arr=np.arange(0, 13)+0.3, meta={'a': 2})
    holder.set_taken_length(5)
    assert holder.index_list == [(0, 0), (0, 5), (1, 0), (1, 4), (1, 8), (1, 12), (2, 0), (2, 4), (2, 8)]
    assert len(holder) == 9
    for i in range(9):
        assert len(holder.getitem(i)) == 5
    print(holder.getitem(4).data_arr)
    assert holder.getitem(0) == Signal(data_arr=np.arange(0, 5)+0.1, meta={'a': 0})
    assert holder.getitem(4) == Signal(data_arr=np.arange(8, 13)+0.2, meta={'a': 1})
    holder.next_epoch()
    assert holder.index_list == [(0, 2), (0, 2), (1, 2), (1, 4), (1, 7), (1, 9), (2, 2), (2, 4), (2, 5)]


def test_holder_with_priority():
    holder = TimeSeriesHolder(timeseries=[
        Signal(data_arr=np.arange(0, 10)+0.1, meta={'a': 0}),
        Signal(data_arr=np.arange(0, 17)+0.2, meta={'a': 1}),
        Signal(data_arr=np.arange(0, 13)+0.3, meta={'a': 2}),
    ], priorities=[1, 3, 1])
    holder.build()
    assert holder.index_list == [(0, None), (1, None), (1, None), (1, None), (2, None)]
    assert len(holder) == 5
    assert holder.getitem(4) == Signal(data_arr=np.arange(0, 13)+0.3, meta={'a': 2})
    holder.priorities = [3, 2, 2]
    holder.set_taken_length(5)
    assert holder.index_list == [(0, 0), (0, 2), (0, 5), (1, 0), (1, 12), (2, 0), (2, 8)]
    assert len(holder) == 7
    for i in range(7):
        assert len(holder.getitem(i)) == 5
    assert holder.getitem(0) == Signal(data_arr=np.arange(0, 5)+0.1, meta={'a': 0})
    assert holder.getitem(4) == Signal(data_arr=np.arange(12, 17)+0.2, meta={'a': 1})
    assert holder.current_offset == 0
    holder.next_epoch()
    assert holder.current_offset == 1/2
    assert holder.index_list == [(0, 2), (0, 2), (0, 2), (1, 2), (1, 9), (2, 2), (2, 5)]
    assert len(holder) == 7
    for i in range(7):
        assert len(holder.getitem(i)) == 5


