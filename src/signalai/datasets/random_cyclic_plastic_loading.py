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
        self.use_kiso = use_kiso
        self.params = params if params is not None else self.unscale_params(scaled_params)
        self.depspc_r = depspc_r

    @classmethod
    def scaling_coefficients(cls) -> dict:
        pass

    @property
    def scaled_params(self) -> np.ndarray:
        scaling_coef = self.scaling_coefficients()
        values = []
        if self.use_kiso:
            values += [
                self.scale(self.params['k0'], scaling_coef['k0']),
                self.scale(self.params['kap'][0], scaling_coef['kap'][0]),
                self.scale(1 / self.params['kap'][1], scaling_coef['kap'][1]),
            ]
        values += [self.scale(np.log(c), coef) for c, coef in zip(self.params['c'], scaling_coef['c'])]
        values += [self.scale(a, scaling_coef['a']) for a in self.params['a']]
        return np.array(values)

    def unscale_params(self, scaled_params: np.ndarray) -> dict:
        scaling_coef = self.scaling_coefficients()
        c_len = len(scaling_coef['c'])
        kiso = {}
        start_id = 0
        if self.use_kiso:
            start_id = 3
            kiso = {
                'k0': self.unscale(scaled_params[0], scaling_coef['k0']),
                'kap': [self.unscale(scaled_params[1], scaling_coef['kap'][0]),
                        1 / self.unscale(scaled_params[2], scaling_coef['kap'][1])],
            }
        assert len(scaled_params) == start_id + 2 * c_len
        return {
            **kiso,
            'c': np.exp(np.array(
                [self.unscale(scaled_params[i + start_id], coef) for i, coef in enumerate(scaling_coef['c'])])),
            'a': np.array([self.unscale(a, scaling_coef['a']) for a in scaled_params[start_id + c_len:]]),
        }

    @staticmethod
    def _uniform_params(interval: tuple[float, float]) -> tuple[float, float]:
        a, b = interval
        assert b > a
        return (a + b) / 2, (b - a) / np.sqrt(12)

    @staticmethod
    def scale(num, coefs: tuple[float, float]) -> float:
        return (num - coefs[0]) / coefs[1]

    @staticmethod
    def unscale(num, coefs: tuple[float, float]) -> float:
        return num *coefs[1] + coefs[0]

    @classmethod
    def generate(cls, depspc_r=None, depspc_r_number: int = None, use_kiso=True):
        scaling_coef = cls.scaling_coefficients()
        c_len = len(scaling_coef['c'])
        a = np.random.uniform(0, 1, size=c_len)
        a /= np.sum(a)
        a = np.random.uniform(250, 350) * a
        params = {
            'c': np.sort([np.exp(np.random.uniform(np.log(1000), np.log(10000))),
                          *np.exp(np.random.uniform(np.log(50), np.log(2000), size=c_len-1))])[::-1],
            'a': a,
        }
        if use_kiso:
            params.update({
                'k0': np.random.uniform(150, 250),
                'kap': [np.random.uniform(100, 10000), 1 / np.random.uniform(30, 150)]
            })
        return cls(
            params=params,
            depspc_r=np.random.uniform(0.0005, 0.005, depspc_r_number) if depspc_r is None else np.array(depspc_r),
            use_kiso=use_kiso,
        )


class RandomCyclicPlasticLoadingParams1D(RandomCyclicPlasticLoadingParams):

    @classmethod
    def scaling_coefficients(cls) -> dict:
        return {
            'k0': cls._uniform_params((150, 250)),
            'kap': (cls._uniform_params((100, 10000)), cls._uniform_params((30, 150))),
            'c': (cls._uniform_params((np.log(1000), np.log(10000))),),
            'a': cls._uniform_params((250, 350)),
        }


class RandomCyclicPlasticLoadingParams2D(RandomCyclicPlasticLoadingParams):

    @classmethod
    def scaling_coefficients(cls) -> dict:
        return {
            'k0': cls._uniform_params((150, 250)),
            'kap': (cls._uniform_params((100, 10000)), cls._uniform_params((30, 150))),
            'c': ((8.06, 0.65), (5.75, 1.05)),
            'a': (150, 73.5),
        }


class RandomCyclicPlasticLoadingParams3D(RandomCyclicPlasticLoadingParams):

    @classmethod
    def scaling_coefficients(cls) -> dict:
        return {
            'k0': cls._uniform_params((150, 250)),
            'kap': (cls._uniform_params((100, 10000)), cls._uniform_params((30, 150))),
            'c': ((8.06, 0.65), (6.36, 0.86), (5.15, 0.87)),
            'a': (100, 54.9),
        }


class RandomCyclicPlasticLoadingParams4D(RandomCyclicPlasticLoadingParams):

    @classmethod
    def scaling_coefficients(cls) -> dict:
        return {
            'k0': cls._uniform_params((150, 250)),
            'kap': (cls._uniform_params((100, 10000)), cls._uniform_params((30, 150))),
            'c': ((8.07521718, 0.64220986), (6.66012441, 0.70230589), (5.75443793, 0.82282555), (4.83443022, 0.71284407)),
            'a': (75, 42.8),
        }


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
        use_kiso = self.config.get('use_kiso', True)
        model_params = eval(f"RandomCyclicPlasticLoadingParams{self.config['dim']}D").generate(
            depspc_r=self.config.get("depspc_r"),
            depspc_r_number=self.config.get("depspc_r_number"),
            use_kiso=use_kiso,
        )
        return random_cyclic_plastic_loading(model_params, mode=self.config['mode'], use_kiso=use_kiso, **self.config["rcpl_kwargs"])

    def is_infinite(self) -> bool:
        return True
