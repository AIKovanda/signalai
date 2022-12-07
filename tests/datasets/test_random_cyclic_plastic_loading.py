import numpy as np
from matplotlib import pyplot as plt

from datasets.random_cyclic_plastic_loading import (
    RandomCyclicPlasticLoadingGen,
    RandomCyclicPlasticLoadingParams1D,
    RandomCyclicPlasticLoadingParams4D,
)


def test_norm():
    for i in range(1, 12):
        rcpl_params = RandomCyclicPlasticLoadingParams4D.generate(12)
        assert rcpl_params._denormalize_uniform(rcpl_params._normalize_uniform(i, [-1, 13]), [-1, 13]) == i
        assert np.abs(np.exp(rcpl_params._denormalize_uniform(rcpl_params._normalize_uniform(np.log(i), [-1, 13]), [-1, 13])) - i) < 1e-8


def test_params():
    for class_ in [RandomCyclicPlasticLoadingParams1D, RandomCyclicPlasticLoadingParams4D]:
        for i in range(10):
            rcpl_params = class_.generate(12)
            scaled = rcpl_params.scaled_params
            print(scaled)
            assert np.sum(np.abs(scaled - class_(scaled_params=scaled).scaled_params)) < 1e-15, scaled - class_(scaled_params=scaled).scaled_params


def test_generation():
    for dim in [1, 4]:
        for use_kiso in [True, False]:
            for mode, conf in [('equidistant', {'distance': 0.0001}), ('scaled', {'steps': 12})]:
                rtsg = RandomCyclicPlasticLoadingGen(
                    depspc_r_number=12, mode=mode, use_kiso=use_kiso, rcpl_kwargs=conf, dim=dim,
                )
                sigs = []
                params = []
                for i in range(10000):
                    item = rtsg.getitem(i)
                    sigs.append(item.data_arr)
                    assert item.data_arr.shape[-1] > 1, item.data_arr
                    params.append(item.meta['rcpl_params'].scaled_params)

                sigs = np.concatenate(sigs, axis=-1)
                params = np.stack(params)
                print(f'Dim: {dim}, use_kiso: {int(use_kiso)}, mode: {mode}')
                print('mu_abs', np.mean(np.abs(sigs)))
                print('mu', np.mean(sigs))
                print('std', np.std(sigs))
                print('mu_params', np.mean(params, axis=0))
                print('std_params', np.std(params, axis=0))
                print('-'*25)


def test_plot():
    rtsg = RandomCyclicPlasticLoadingGen(depspc_r_number=12, rcpl_kwargs={'steps': 12}, dim=4)
    plt.figure()
    for i in range(6):
        print('-' * 12, i)
        item = rtsg.getitem(i)
        plt.plot(item.meta['epspc'], item.data_arr[0], linewidth=0.2)

    plt.xlabel('$\\epsilon_p^c$')
    plt.ylabel('$\\sigma$')
    plt.savefig('out.svg')

    plt.figure()
    for i in range(6):
        print('-' * 12, i)
        item = rtsg.getitem(i)
        plt.plot(list(range(len(item.meta['epspc']))), item.data_arr[0], linewidth=0.2)

    plt.plot(list(range(len(item.meta['epspc']))), np.zeros_like(item.meta['epspc']), 'b.', markersize=1)
    plt.ylabel('$\\sigma$')
    plt.savefig('out_scaled.svg')
