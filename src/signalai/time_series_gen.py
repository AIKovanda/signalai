import abc
from functools import lru_cache
from itertools import chain, count

import numpy as np
from taskchain.parameter import AutoParameterObject

from signalai.time_series import sum_time_series, TimeSeries


def offset_generator():
    yield 1/2
    for pow_ in count(2):
        for i in range(1, 2**(pow_ - 1), 2):
            yield i / 2**pow_
            yield 0.5 + i / 2**pow_


class TimeSeriesGen(AutoParameterObject, abc.ABC):
    def __init__(self, **config):
        self.input_ts_gen_args: list[TimeSeriesGen] = []
        self.input_ts_gen_kwargs: dict[any, TimeSeriesGen] = {}
        self.taken_length = None
        self.epoch_id = 0
        self.config = config
        self.built = False
        self.getitem = lru_cache(1)(self._getitem)

    def _next_epoch(self):
        pass

    def next_epoch(self):
        self.epoch_id += 1
        self._next_epoch()
        for i in chain(self.input_ts_gen_args, self.input_ts_gen_kwargs.values()):
            i.next_epoch()

    def _build(self):  # expensive loading/building function for data
        pass

    def build(self):
        if not self.built:
            self._build()
            self.built = True
            for i in chain(self.input_ts_gen_args, self.input_ts_gen_kwargs.values()):
                i.build()

            assert all([i.built for i in chain(self.input_ts_gen_args, self.input_ts_gen_kwargs.values())])

    def _set_taken_length(self, length):
        pass

    def set_taken_length(self, length):
        assert length is not None
        if self.taken_length is None:
            self.taken_length = length
        else:
            if self.taken_length != length:
                raise ValueError(
                    f'This generator already gives length of {self.taken_length}, length {length} cannot '
                    f'be set instead. To bypass, restart taken_length value by calling reset_taken_length().')

        self._set_taken_length(length)
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

    def get_random_item(self):
        return self._getitem(np.random.randint(0, len(self)))


class TimeSeriesHolder(TimeSeriesGen):

    def __init__(self, timeseries: list[TimeSeries] = None, priorities: list[int] = None, **config):
        super().__init__(**config)
        self.timeseries = timeseries
        self.priorities = priorities
        self.offset_generator = offset_generator()
        self.current_offset = 0
        self.index_list: list[tuple[int, int | None]] = []

    def _set_taken_length(self, length):
        self._build_index_list()

    def __len__(self):
        return len(self.index_list)

    def _next_epoch(self):
        self.current_offset = next(self.offset_generator)
        self._build_index_list()

    def _build(self):
        self._build_index_list()

    def _build_index_list(self):
        assert self.timeseries is not None
        self.index_list = []
        if self.taken_length is None:
            priorities = self.priorities if self.priorities is not None else [1]*len(self.timeseries)
            for i, priority in zip(range(len(self.timeseries)), priorities):
                self.index_list += [(i, None)] * priority
        else:
            for i, ts in enumerate(self.timeseries):
                ts_len = len(ts)
                interval_count = int(np.ceil(ts_len / self.taken_length)) if self.priorities is None else self.priorities[i]
                if self.current_offset == 0:
                    end = ts_len - self.taken_length
                else:
                    end = ts_len + self.taken_length * (self.current_offset - 2)
                for j in np.linspace(self.current_offset*self.taken_length, end, interval_count):
                    self.index_list.append((i, int(j)))

    def is_infinite(self) -> bool:
        return False

    def map_index(self, idx: int) -> tuple[int, int | None]:
        return self.index_list[idx]

    def _get_ts(self, ts_id: int, interval: tuple[int, int] = None) -> TimeSeries:
        if interval is not None:
            return self.timeseries[ts_id].crop(interval)
        return self.timeseries[ts_id]

    def _getitem(self, idx: int) -> TimeSeries:
        ts_id, position = self.map_index(idx)
        if self.taken_length:
            assert position is not None, 'Position in signal must be defined when specifying the taken length!'
            return self._get_ts(ts_id, (position, position+self.taken_length))

        assert position is None, 'Position in signal is defined without specifying the taken length!'
        return self._get_ts(ts_id)

    def __eq__(self, other):
        if self.timeseries != other.timeseries:
            if self.timeseries is not None or other.timeseries is not None:
                return False
        if self.priorities != other.priorities:
            if self.priorities is not None or other.priorities is not None:
                return False
        if self.current_offset != other.current_offset:
            if self.current_offset is not None or other.current_offset is not None:
                return False
        if self.index_list != other.index_list:
            if self.index_list is not None or other.index_list is not None:
                return False

        return True


class Transformer(TimeSeriesGen):
    takes = None  # 'time_series', 'list', 'dict'

    def process(self, input_: TimeSeries | list[TimeSeries] | dict[str, TimeSeries]) -> TimeSeries | np.ndarray:
        if self.takes == 'time_series':
            transform_chance = self.config.get('transform_chance', 1.)
            if len(self.input_ts_gen_args) > 0:
                if np.random.rand() > transform_chance:
                    return self.input_ts_gen_args[0].process(input_)
                return self._process(self.input_ts_gen_args[0].process(input_))

            if np.random.rand() > transform_chance:
                return input_
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
        assert length is not None
        needed_length = self.transform_taken_length(length)
        assert needed_length is not None, type(self)
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
            input_ = self.input_ts_gen_args[0].getitem(item)
            transform_chance = self.config.get('transform_chance', 1.)
            skip_transform = np.random.rand() > transform_chance and self.transform_taken_length(
                self.taken_length) == self.taken_length

            if skip_transform:
                return input_
            return self._process(input_)
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
        return length

    def _process(self, ts_list: list[TimeSeries]) -> TimeSeries:
        return sum_time_series(ts_list)


class ChannelPruner(Transformer):
    takes = 'time_series'

    def _build(self):
        assert 'choose_channels' in self.config

    def transform_taken_length(self, length: int) -> int:
        return length

    def _process(self, ts: TimeSeries) -> TimeSeries:
        choose_channels = eval(self.config.get("choose_channels"))
        return ts.take_channels(channels=choose_channels)


class TimeMapScale(Transformer):
    """
    channels: list containing list of integers
    """
    takes = 'time_series'

    def _process(self, x: TimeSeries) -> np.ndarray:
        first_crop = self.config.get("first_crop")
        target_length = self.config.get("target_length")
        if target_length is None:
            target_length = x.time_map.shape[-1] * eval(str(self.config.get("scale")))
        time_map = x.time_map.astype(int)
        if first_crop is not None:
            time_map = time_map[..., first_crop[0]: time_map.shape[-1] - first_crop[1]]
        # nearest
        return time_map[:, np.round(np.linspace(0, time_map.shape[-1] - 1, int(target_length))).astype(int)]

    def transform_taken_length(self, length: int) -> int:
        # here it does not make sense, but it would be hard to make alternative
        return length


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
