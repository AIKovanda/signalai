import numpy as np
import yaml

from time_series import TimeSeries, Signal
from signalai.time_series_gen import TimeSeriesSum, TimeSeriesGen, LambdaTransformer, make_graph


class SinusGenerator(TimeSeriesGen):
    def _set_epoch(self, epoch_id: int):
        self.config['epoch'] = epoch_id

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
    x_gen.set_epoch(1)
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

