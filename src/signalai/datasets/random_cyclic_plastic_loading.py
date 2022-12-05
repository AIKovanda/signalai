import abc

import numpy as np
from numba import jit

from signalai.time_series import TimeSeries
from signalai.time_series_gen import TimeSeriesGen

SQR23 = np.sqrt(2. / 3.)
SQR32 = np.sqrt(1.5)


class RandomCyclicPlasticLoadingParams:
    def __init__(self, params: dict = None, scaled_params: np.ndarray = None, depspc_r=None):
        assert (params is None) ^ (scaled_params is None)  # xor
        self.params = params if params is not None else self.unscale_params(scaled_params)
        self.depspc_r = depspc_r

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
    def generate(cls, depspc_r=None, depspc_r_number: int = None):
        pass


class RandomCyclicPlasticLoadingParams1D(RandomCyclicPlasticLoadingParams):
    @property
    def scaled_params(self) -> np.ndarray:
        return np.array([
            self._normalize_uniform(self.params['k0'], (150, 250)),
            self._normalize_uniform(self.params['kap'][0], (100, 10000)),
            self._normalize_uniform(1 / self.params['kap'][1], (30, 150)),
            self._normalize_uniform(np.log(self.params['c'][0]), (np.log(1000), np.log(10000))),
            self._normalize_uniform(self.params['a'][0], (250, 350)),
        ])

    def unscale_params(self, scaled_params: np.ndarray) -> dict:
        assert len(scaled_params) == 5
        return {
            'k0': self._denormalize_uniform(scaled_params[0], (150, 250)),
            'kap': [self._denormalize_uniform(scaled_params[1], (100, 10000)),
                    1 / self._denormalize_uniform(scaled_params[2], (30, 150))],
            'c': np.exp(np.expand_dims(self._denormalize_uniform(scaled_params[3], (np.log(1000), np.log(10000))), 0)),
            'a': np.expand_dims(self._denormalize_uniform(scaled_params[4], (250, 350)), 0),
        }

    @classmethod
    def generate(cls, depspc_r=None, depspc_r_number: int = None):
        return RandomCyclicPlasticLoadingParams1D(params={
            # 'E': np.random.uniform(160, 210),  # GPa
            'k0': np.random.uniform(150, 250),
            'kap': [np.random.uniform(100, 10000), 1 / np.random.uniform(30, 150)],  # MPa
            'c': np.exp(np.random.uniform(np.log(1000), np.log(10000), size=1)),
            'a': np.random.uniform(250, 350, size=1),
        },
            depspc_r=np.random.uniform(0.0005, 0.005, depspc_r_number) if depspc_r is None else np.array(depspc_r),
        )


class RandomCyclicPlasticLoadingParams4D(RandomCyclicPlasticLoadingParams):

    @property
    def scaled_params(self) -> np.ndarray:
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

    def unscale_params(self, scaled_params: np.ndarray) -> dict:
        len_a = int((len(scaled_params) - 3) / 2)
        assert len_a == 4
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

    @classmethod
    def generate(cls, depspc_r=None, depspc_r_number: int = None):
        a = np.random.uniform(0, 1, size=4)
        a /= np.sum(a)
        return RandomCyclicPlasticLoadingParams4D(params={
            # 'E': np.random.uniform(160, 210),  # GPa
            'k0': np.random.uniform(150, 250),
            'kap': [np.random.uniform(100, 10000), 1 / np.random.uniform(30, 150)],  # MPa
            'c': np.sort([np.exp(np.random.uniform(np.log(1000), np.log(10000))),
                          *np.exp(np.random.uniform(np.log(50), np.log(2000), size=3))])[::-1],
            'a': np.random.uniform(250, 350) * a,
        },
            depspc_r=np.random.uniform(0.0005, 0.005, depspc_r_number) if depspc_r is None else np.array(depspc_r),
        )


@jit(nopython=True)
def _random_cyclic_plastic_loading(kap: np.ndarray, k0: float, a: np.ndarray, c: np.ndarray, depspc_r: np.ndarray, steps: int):
    epspc = np.zeros(len(depspc_r) * steps + 1, dtype=np.float32)
    kiso = np.zeros(len(depspc_r) * steps + 1, dtype=np.float32)
    alp = np.zeros((len(a), len(depspc_r) * steps + 1), dtype=np.float32)

    for i, next_depspc_r in enumerate(depspc_r):  # next_depspc_r does not start with 0!
        D = (-1) ** i
        for j in range(1, steps + 1):
            epspc[i * steps + j] = epspc[i * steps] + j * next_depspc_r / steps
            depspc = j * next_depspc_r / steps

            kiso[i * steps + j] = D / kap[1] * (1 - (1 - kap[1] * k0) * np.exp(
                -SQR32 * kap[0] * kap[1] * epspc[i * steps + j]))
            alp[:, i * steps + j] = D * a - (D * a - alp[:, i * steps]) * np.exp(-c * depspc)

    sig = SQR23 * np.sum(alp, axis=0) + kiso
    return epspc, sig


def random_cyclic_plastic_loading(rcpl_params: RandomCyclicPlasticLoadingParams4D, steps: int) -> TimeSeries:
    a = np.array(rcpl_params.params['a'], dtype=np.float32)
    c = np.array(rcpl_params.params['c'], dtype=np.float32)
    assert len(a) == len(c), 'a and c must be the same size'
    assert rcpl_params.depspc_r[0] != 0, 'Leading zero is forbidden'
    kap = np.array(rcpl_params.params['kap'], dtype=np.float32)
    k0 = float(rcpl_params.params['k0'])
    epspc, sig = _random_cyclic_plastic_loading(
        kap=kap,
        k0=k0,
        a=a,
        c=c,
        depspc_r=np.array(rcpl_params.depspc_r, dtype=np.float32),
        steps=int(steps),
    )
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
        assert 'steps' in self.config

    def _getitem(self, _: int) -> TimeSeries:
        classes = {
            1: RandomCyclicPlasticLoadingParams1D,
            4: RandomCyclicPlasticLoadingParams4D,
        }

        model_params = classes[self.config['dim']].generate(
            depspc_r=self.config.get("depspc_r"),
            depspc_r_number=self.config.get("depspc_r_number"),
        )
        return random_cyclic_plastic_loading(model_params, steps=self.config["steps"])

    def is_infinite(self) -> bool:
        return True
