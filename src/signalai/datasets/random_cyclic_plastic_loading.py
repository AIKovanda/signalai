import abc

import numpy as np
from numba import jit

from signalai.time_series import TimeSeries
from signalai.time_series_gen import TimeSeriesGen

SQR23 = np.sqrt(2. / 3.)
SQR32 = np.sqrt(1.5)


class RandomCyclicPlasticLoadingParams:
    def __init__(self, params: dict = None, scaled_params: np.ndarray = None, depspc_r=None, use_kiso=True):
        assert (params is None) ^ (scaled_params is None)  # xor
        self.params = params if params is not None else self.unscale_params(scaled_params)
        self.depspc_r = depspc_r
        self.use_kiso = use_kiso

    @property
    @abc.abstractmethod
    def scaled_params(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def unscale_params(self, scaled_params: np.ndarray) -> dict:
        pass

    @staticmethod
    def _uniform_params(interval: tuple[float, float]) -> tuple[float, float]:
        a, b = interval
        assert b > a
        return (a + b) / 2, (b - a) / np.sqrt(12)

    def _normalize_uniform(self, num, interval: tuple[float, float]) -> float:
        mu, sigma = self._uniform_params(interval)
        return (num - mu) / sigma

    def _denormalize_uniform(self, num, interval: tuple[float, float]) -> float:
        mu, sigma = self._uniform_params(interval)
        return num * sigma + mu

    @classmethod
    def generate(cls, depspc_r=None, depspc_r_number: int = None, use_kiso=True):
        pass


class RandomCyclicPlasticLoadingParams1D(RandomCyclicPlasticLoadingParams):
    @property
    def scaled_params(self) -> np.ndarray:
        if self.use_kiso:
            return np.array([
                self._normalize_uniform(self.params['k0'], (150, 250)),
                self._normalize_uniform(self.params['kap'][0], (100, 10000)),
                self._normalize_uniform(1 / self.params['kap'][1], (30, 150)),
                self._normalize_uniform(np.log(self.params['c'][0]), (np.log(1000), np.log(10000))),
                self._normalize_uniform(self.params['a'][0], (250, 350)),
            ])
        return np.array([
            self._normalize_uniform(np.log(self.params['c'][0]), (np.log(1000), np.log(10000))),
            self._normalize_uniform(self.params['a'][0], (250, 350)),
        ])

    def unscale_params(self, scaled_params: np.ndarray) -> dict:
        assert len(scaled_params) == 5
        if self.use_kiso:
            return {
                'k0': self._denormalize_uniform(scaled_params[0], (150, 250)),
                'kap': [self._denormalize_uniform(scaled_params[1], (100, 10000)),
                        1 / self._denormalize_uniform(scaled_params[2], (30, 150))],
                'c': np.exp(np.expand_dims(self._denormalize_uniform(scaled_params[3], (np.log(1000), np.log(10000))), 0)),
                'a': np.expand_dims(self._denormalize_uniform(scaled_params[4], (250, 350)), 0),
            }
        return {
            'c': np.exp(np.expand_dims(self._denormalize_uniform(scaled_params[0], (np.log(1000), np.log(10000))), 0)),
            'a': np.expand_dims(self._denormalize_uniform(scaled_params[1], (250, 350)), 0),
        }

    @classmethod
    def generate(cls, depspc_r=None, depspc_r_number: int = None, use_kiso=True):
        params = {
            'c': np.exp(np.random.uniform(np.log(1000), np.log(10000), size=1)),
            'a': np.random.uniform(250, 350, size=1),
        }
        if use_kiso:
            params.update({
                'k0': np.random.uniform(150, 250),
                'kap': [np.random.uniform(100, 10000), 1 / np.random.uniform(30, 150)],  # MPa
            })
        return RandomCyclicPlasticLoadingParams1D(
            params=params,
            depspc_r=np.random.uniform(0.0005, 0.005, depspc_r_number) if depspc_r is None else np.array(depspc_r),
            use_kiso=use_kiso,
        )


class RandomCyclicPlasticLoadingParams4D(RandomCyclicPlasticLoadingParams):

    @property
    def scaled_params(self) -> np.ndarray:
        if self.use_kiso:
            return np.array([
                self._normalize_uniform(self.params['k0'], (150, 250)),
                self._normalize_uniform(self.params['kap'][0], (100, 10000)),
                self._normalize_uniform(1 / self.params['kap'][1], (30, 150)),
                (np.log(self.params['c'][0]) - 8.07521718) / 0.64220986,
                (np.log(self.params['c'][1]) - 6.66012441) / 0.70230589,
                (np.log(self.params['c'][2]) - 5.75443793) / 0.82282555,
                (np.log(self.params['c'][3]) - 4.83443022) / 0.71284407,
                *[(i - 75) / 42.8 for i in self.params['a']],
            ])
        return np.array([
            (np.log(self.params['c'][0]) - 8.07521718) / 0.64220986,
            (np.log(self.params['c'][1]) - 6.66012441) / 0.70230589,
            (np.log(self.params['c'][2]) - 5.75443793) / 0.82282555,
            (np.log(self.params['c'][3]) - 4.83443022) / 0.71284407,
            *[(i - 75) / 42.8 for i in self.params['a']],
        ])

    def unscale_params(self, scaled_params: np.ndarray) -> dict:
        len_a = int((len(scaled_params) - 3) / 2)
        assert len_a == 4
        if self.use_kiso:
            return {
                'k0': self._denormalize_uniform(scaled_params[0], (150, 250)),
                'kap': [self._denormalize_uniform(scaled_params[1], (100, 10000)),
                        1 / self._denormalize_uniform(scaled_params[2], (30, 150))],
                'c': np.exp(np.array([
                    scaled_params[3] * 0.64220986 + 8.07521718,
                    scaled_params[4] * 0.70230589 + 6.66012441,
                    scaled_params[5] * 0.82282555 + 5.75443793,
                    scaled_params[6] * 0.71284407 + 4.83443022,
                ])),
                'a': scaled_params[7:] * 42.8 + 75,
            }
        return {
            'c': np.exp(np.array([
                scaled_params[0] * 0.64220986 + 8.07521718,
                scaled_params[1] * 0.70230589 + 6.66012441,
                scaled_params[2] * 0.82282555 + 5.75443793,
                scaled_params[3] * 0.71284407 + 4.83443022,
            ])),
            'a': scaled_params[4:] * 42.8 + 75,
        }

    @classmethod
    def generate(cls, depspc_r=None, depspc_r_number: int = None, use_kiso=True):
        a = np.random.uniform(0, 1, size=4)
        a /= np.sum(a)
        params = {
            'c': np.sort([np.exp(np.random.uniform(np.log(1000), np.log(10000))),
                          *np.exp(np.random.uniform(np.log(50), np.log(2000), size=3))])[::-1],
            'a': np.random.uniform(250, 350) * a,
        }
        if use_kiso:
            params.update({
                'k0': np.random.uniform(150, 250),
                'kap': [np.random.uniform(100, 10000), 1 / np.random.uniform(30, 150)]
            })
        return RandomCyclicPlasticLoadingParams4D(
            params=params,
            depspc_r=np.random.uniform(0.0005, 0.005, depspc_r_number) if depspc_r is None else np.array(depspc_r),
            use_kiso=use_kiso,
        )


@jit(nopython=True)
def _random_cyclic_plastic_loading(kap: np.ndarray, k0: float, a: np.ndarray, c: np.ndarray,
                                   depspc_r: np.ndarray, steps: int, use_kiso=True):
    epspc = np.zeros(len(depspc_r) * steps + 1, dtype=np.float32)
    if use_kiso:
        kiso = np.zeros(len(depspc_r) * steps + 1, dtype=np.float32)
    alp = np.zeros((len(a), len(depspc_r) * steps + 1), dtype=np.float32)

    for i, next_depspc_r in enumerate(depspc_r):
        D = (-1) ** i
        for j in range(1, steps + 1):
            epspc[i * steps + j] = epspc[i * steps] + j * next_depspc_r / steps
            depspc = j * next_depspc_r / steps

            if use_kiso:
                kiso[i * steps + j] = D / kap[1] * (1 - (1 - kap[1] * k0) * np.exp(
                    -SQR32 * kap[0] * kap[1] * epspc[i * steps + j]))
            alp[:, i * steps + j] = D * a - (D * a - alp[:, i * steps]) * np.exp(-c * depspc)

    sig = SQR23 * np.sum(alp, axis=0) + kiso if use_kiso else SQR23 * np.sum(alp, axis=0)
    return sig, epspc


@jit(nopython=True)
def _random_cyclic_plastic_loading_equidistant(kap: np.ndarray, k0: float, a: np.ndarray, c: np.ndarray,
                                               depspc_r: np.ndarray, distance: float, use_kiso=True):
    N = int(np.sum(depspc_r) // distance) + 1
    if use_kiso:
        kiso = np.zeros(N, dtype=np.float32)
    alp = np.zeros((len(a), N), dtype=np.float32)
    depspc = 0.
    D = 1
    alp0 = np.zeros_like(a)
    prev_seg_id = 0
    epspc_rcum = np.cumsum(depspc_r)
    for i in range(N):
        epspc = i * distance
        act_seg_id = np.argmin(epspc >= epspc_rcum)
        if act_seg_id > prev_seg_id:  # Update back-stress
            alp0 = D * a - (D * a - alp0) * np.exp(-c * depspc_r[act_seg_id])
            D *= -1
            # Subtract accumulated previous segment to get sub-increment of a new segment
            depspc -= depspc_r[prev_seg_id]
            prev_seg_id = act_seg_id
        depspc += distance
        alp[:, i] = D * a - (D * a - alp0) * np.exp(-c * depspc)
        if use_kiso:
            kiso[i] = D / kap[1] * (1 - (1 - kap[1] * k0) * np.exp(-SQR32 * kap[0] * kap[1] * epspc))
    if use_kiso:
        return SQR23 * np.sum(alp, axis=0) + kiso  # Remove kiso to eliminate signal jumps
    return SQR23 * np.sum(alp, axis=0)


def random_cyclic_plastic_loading(rcpl_params: RandomCyclicPlasticLoadingParams, mode: str, use_kiso=True, **kwargs) -> TimeSeries:
    func = {
        'equidistant': _random_cyclic_plastic_loading_equidistant,
        'scaled': _random_cyclic_plastic_loading,
    }[mode]
    a = np.array(rcpl_params.params['a'], dtype=np.float32)
    c = np.array(rcpl_params.params['c'], dtype=np.float32)
    assert len(a) == len(c), 'a and c must be the same size'
    assert rcpl_params.depspc_r[0] != 0, 'Leading zero is forbidden'
    kap = np.array(rcpl_params.params['kap'], dtype=np.float32) if use_kiso else np.zeros(2)
    k0 = float(rcpl_params.params['k0']) if use_kiso else 0
    out = func(
        kap=kap,
        k0=k0,
        a=a,
        c=c,
        depspc_r=np.array(rcpl_params.depspc_r, dtype=np.float32),
        use_kiso=use_kiso,
        **kwargs
    )
    if isinstance(out, tuple):
        sig, epspc = out
    else:
        sig, epspc = out, None
    return TimeSeries(
        data_arr=sig,
        meta={'rcpl_params': rcpl_params, 'epspc': epspc},
    )


class RandomCyclicPlasticLoadingGen(TimeSeriesGen):

    def __len__(self):
        raise ValueError

    def _build(self):
        assert 'dim' in self.config
        assert 'depspc_r' in self.config or 'depspc_r_number' in self.config
        assert 'rcpl_kwargs' in self.config  # use_kiso, steps or distance
        assert self.config['mode'] in ['equidistant', 'scaled']

    def _getitem(self, _: int) -> TimeSeries:
        classes = {
            1: RandomCyclicPlasticLoadingParams1D,
            4: RandomCyclicPlasticLoadingParams4D,
        }

        use_kiso = self.config.get('use_kiso', True)
        model_params = classes[self.config['dim']].generate(
            depspc_r=self.config.get("depspc_r"),
            depspc_r_number=self.config.get("depspc_r_number"),
            use_kiso=use_kiso,
        )
        return random_cyclic_plastic_loading(model_params, mode=self.config['mode'], use_kiso=use_kiso, **self.config["rcpl_kwargs"])

    def is_infinite(self) -> bool:
        return True
