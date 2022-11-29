import abc
from functools import lru_cache
from itertools import chain

import numpy as np
from taskchain.parameter import AutoParameterObject

from signalai.time_series import TimeSeries, sum_time_series


class TimeSeriesGen(AutoParameterObject, abc.ABC):
    def __init__(self, **config):
        self.input_ts_gen_args: list[TimeSeriesGen] = []
        self.input_ts_gen_kwargs: dict[any, TimeSeriesGen] = {}
        self.taken_length = None
        self.config = config
        self.data = None
        self.built = False
        self.getitem = lru_cache(1)(self._getitem)

    def _set_epoch(self, epoch_id: int):
        pass

    def set_epoch(self, epoch_id: int):
        self._set_epoch(epoch_id)
        for i in chain(self.input_ts_gen_args, self.input_ts_gen_kwargs.values()):
            i.set_epoch(epoch_id)

    def _build(self):  # expensive loading/building function for data
        pass

    def build(self):
        if not self.built:
            self._build()
            self.built = True
            for i in chain(self.input_ts_gen_args, self.input_ts_gen_kwargs.values()):
                i.build()

    def set_taken_length(self, length):
        if self.taken_length is None:
            self.taken_length = length
        else:
            if self.taken_length != length:
                raise ValueError(
                    f'This generator already gives length of {self.taken_length}, length {length} cannot '
                    f'be set instead. To bypass, restart taken_length value by calling reset_taken_length().')

        self._set_child_taken_length(length)

    def _set_child_taken_length(self, length):
        pass

    def reset_taken_length(self):
        self.taken_length = None
        for i in chain(self.input_ts_gen_args, self.input_ts_gen_kwargs.values()):
            i.reset_taken_length()

    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _getitem(self, item: int) -> TimeSeries:
        pass

    @abc.abstractmethod
    @lru_cache(1)
    def is_infinite(self) -> bool:
        pass


class Transformer(TimeSeriesGen):
    takes = None  # 'time_series', 'list', 'dict'

    def process(self, input_: TimeSeries | list[TimeSeries] | dict[str, TimeSeries]) -> TimeSeries | np.ndarray:
        if self.takes == 'time_series':
            if len(self.input_ts_gen_args) > 0:
                return self._process(self.input_ts_gen_args[0].process(input_))
            return self._process(input_)
        elif self.takes == 'list':
            if len(self.input_ts_gen_args) > 0:
                return self._process([tsg.process(input_) for tsg in self.input_ts_gen_args])
            return self._process(input_)
        elif self.takes == 'dict':
            if len(self.input_ts_gen_kwargs) > 0:
                return self._process({key: tsg.process(input_) for key, tsg in self.input_ts_gen_kwargs.items()})
            return self._process(input_)
        else:
            raise ValueError(f'Wrongly defined class, takes must be time_series/list/dict, not {self.takes}')

    @abc.abstractmethod
    def _process(self, input_: TimeSeries | list[TimeSeries] | dict[str, TimeSeries]) -> TimeSeries | np.ndarray:
        pass

    @abc.abstractmethod
    @lru_cache(1)
    def transform_taken_length(self, length: int) -> int:
        pass

    def _set_child_taken_length(self, length):
        needed_length = self.transform_taken_length(length)
        for i in chain(self.input_ts_gen_args, self.input_ts_gen_kwargs.values()):
            i.set_taken_length(needed_length)

    def __len__(self):
        lens = {len(tsg) for tsg in chain(self.input_ts_gen_args, self.input_ts_gen_kwargs.values()) if not tsg.is_infinite()}
        if len(lens) > 1:
            raise ValueError('Generators appear to be of different lengths!')
        elif len(lens) == 0:
            raise ValueError('No length is available!')
        return lens.pop()

    def _getitem(self, item: int) -> TimeSeries:
        if self.takes == 'time_series':
            return self._process(self.input_ts_gen_args[0].getitem(item))
        elif self.takes == 'list':
            return self._process([tsg.getitem(item) for tsg in self.input_ts_gen_args])
        elif self.takes == 'dict':
            return self._process({key: tsg.getitem(item) for key, tsg in self.input_ts_gen_kwargs.items()})
        else:
            raise ValueError(f'Wrongly defined class, takes must be time_series/list/dict, not {self.takes}')

    @lru_cache(1)
    def is_infinite(self) -> bool:
        return all([tsg.is_infinite() for tsg in chain(self.input_ts_gen_args, self.input_ts_gen_kwargs.values())])

    def take_input(self, input_: TimeSeriesGen | list[TimeSeriesGen] | dict[str, TimeSeriesGen]):
        transformer = type(self)(**self.config)
        if self.takes == 'time_series':
            assert isinstance(input_, TimeSeriesGen), f'Wrong input type {type(input_)}. Needed {self.takes}.'
            transformer.input_ts_gen_args.append(input_)
        elif self.takes == 'list':
            assert isinstance(input_, list), f'Wrong input type {type(input_)}. Needed {self.takes}.'
            assert all([isinstance(i, TimeSeriesGen) for i in input_])
            transformer.input_ts_gen_args = input_
        elif self.takes == 'dict':
            assert isinstance(input_, dict), f'Wrong input type {type(input_)}. Needed {self.takes}.'
            assert all([isinstance(val, TimeSeriesGen) and isinstance(key, str) for key, val in input_.items()])
            transformer.input_ts_gen_kwargs = input_
        else:
            raise ValueError(f'Wrongly defined class, takes must be time_series/list/dict, not {self.takes}')
        return transformer


class TimeSeriesSum(Transformer):
    takes = 'list'

    def transform_taken_length(self, length: int) -> int:
        pass

    def _process(self, ts_list: list[TimeSeries]) -> TimeSeries:
        return sum_time_series(*ts_list)


class LambdaTransformer(Transformer):
    takes = 'time_series'

    def _build(self):
        assert 'lambda_w' in self.config
        self.__getitem__ = lru_cache(50)(self._getitem)

    def transform_taken_length(self, length: int) -> int:
        return length

    def _process(self, ts: TimeSeries) -> TimeSeries:
        func = eval("lambda w: "+self.config['lambda_w'])
        if self.config.get('apply_to_ts', False):
            return func(ts)

        return ts.apply(func)


def _graph_node(
        time_series_gens: dict[str, TimeSeriesGen | Transformer],
        structure: dict | list | str,
) -> TimeSeriesGen | list[TimeSeriesGen] | dict[str, TimeSeriesGen]:

    if isinstance(structure, str):
        return time_series_gens[structure]
    elif isinstance(structure, list):
        return [_graph_node(time_series_gens, i) for i in structure]
    elif isinstance(structure, dict):
        if len(structure) == 1 and (base_tsg_name := list(structure.keys())[0])[0] != '=':
            base_tsg: Transformer = time_series_gens[base_tsg_name]
            return base_tsg.take_input(_graph_node(time_series_gens, structure[base_tsg_name]))
        else:
            resulting_dict = {}
            for key, val in structure.items():
                assert key[0] == '=', 'Names in dict must start with =.'
                resulting_dict[key[1:]] = _graph_node(time_series_gens, val)
            return resulting_dict
    else:
        raise TypeError(f"Type '{type(structure)}' of structure is not supported. Repair your config.")


def make_graph(time_series_gens: dict[str, TimeSeriesGen], structure: dict) -> dict[str, TimeSeriesGen | Transformer]:
    output_gens = {}
    for key, val in structure.items():
        assert key[0] == '=', 'Names in dict must start with =.'
        output_gens[key[1:]] = _graph_node({**time_series_gens, **output_gens}, val)

    return output_gens
