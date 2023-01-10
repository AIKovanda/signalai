import os
from pathlib import Path

import numpy as np

from signalai.datasets.file_loader import FileLoader, GlobFileLoader
from time_series import Signal


def test_simple_load():
    temp_dir = Path('temp')
    temp_dir.mkdir(exist_ok=True)
    for i in range(1, 10):
        np.save(str((temp_dir / f'{i}.npy').absolute()), np.arange(0, 20)+i/10)
    all_file_structure = [
        {'channels': ['1.npy', '2.npy', '3.npy'], "meta": {'c': 1}},
        {'channels': ['4.npy', '5.npy', '6.npy'], "meta": {'c': 1}},
        {'channels': ['7.npy', '8.npy', '9.npy'], "meta": {'c': 1}},
    ]
    holder = FileLoader(base_dir=temp_dir, all_file_structure=all_file_structure)
    holder.build()
    assert len(holder) == 3
    assert len(holder.timeseries) == 3
    assert holder.priorities is None
    for i in range(3):
        assert holder.timeseries[i] == Signal(
            data_arr=np.vstack([np.arange(0, 20)+(1+3*i)/10, np.arange(0, 20)+(2+3*i)/10, np.arange(0, 20)+(3+3*i)/10]),
            meta={'c': 1})

    all_file_structure = [
        {'channels': ['1.npy', '2.npy', '3.npy'], "meta": {'c': 1}, 'fs': 50, "priority": 5},
        {'channels': ['4.npy', '5.npy', '6.npy'], "meta": {'c': 1}, 'fs': 50, "priority": 1},
        {'channels': ['7.npy', '8.npy', '9.npy'], "meta": {'c': 1}, 'fs': 50, "priority": 2},
    ]
    holder = FileLoader(base_dir=temp_dir, all_file_structure=all_file_structure)
    holder.build()
    assert len(holder) == 8
    assert len(holder.timeseries) == 3
    assert holder.priorities == [5, 1, 2]
    for i in range(3):
        assert holder.timeseries[i] == Signal(
            data_arr=np.vstack([np.arange(0, 20)+(1+3*i)/10, np.arange(0, 20)+(2+3*i)/10, np.arange(0, 20)+(3+3*i)/10]),
            meta={'c': 1}, fs=50)

    holder = FileLoader(base_dir=temp_dir, all_file_structure=all_file_structure, num_workers=3)
    holder.build()
    assert len(holder) == 8
    assert len(holder.timeseries) == 3
    assert holder.priorities == [5, 1, 2]
    for i in range(3):
        assert holder.timeseries[i] == Signal(
            data_arr=np.vstack([np.arange(0, 20)+(1+3*i)/10, np.arange(0, 20)+(2+3*i)/10, np.arange(0, 20)+(3+3*i)/10]),
            meta={'c': 1}, fs=50)

    all_file_structure = [
        {'channels': ['1.npy', '2.npy', '3.npy'], "meta": {'c': 1}, 'fs': 50, "priority": 5, "interval": [2, 10]},
        {'channels': ['4.npy', '5.npy', '6.npy'], "meta": {'c': 1}, 'fs': 50, "priority": 1, "interval": [3, 11]},
        {'channels': ['7.npy', '8.npy', '9.npy'], "meta": {'c': 1}, 'fs': 50, "priority": 2, "interval": [4, 12]},
    ]
    holder = FileLoader(base_dir=temp_dir, all_file_structure=all_file_structure)
    holder.build()
    assert len(holder) == 8
    assert len(holder.timeseries) == 3
    assert holder.priorities == [5, 1, 2]
    for i in range(3):
        assert holder.timeseries[i] == Signal(
            data_arr=np.vstack([np.arange(2+i, i+10)+(1+3*i)/10, np.arange(2+i, i+10)+(2+3*i)/10, np.arange(2+i, i+10)+(3+3*i)/10]),
            meta={'c': 1}, fs=50)

    all_file_structure = [
        {'channels': ['1.npy', '2.npy', '3.npy'], "meta": {'c': 1}, 'fs': 50, "relative_priority": 1, "interval": [2, 8]},
        {'channels': ['4.npy', '5.npy', '6.npy'], "meta": {'c': 1}, 'fs': 50, "relative_priority": 1, "interval": [3, 12]},
        {'channels': ['7.npy', '8.npy', '9.npy'], "meta": {'c': 1}, 'fs': 50, "relative_priority": 1, "interval": [2, 20]},
    ]
    holder = FileLoader(base_dir=temp_dir, all_file_structure=all_file_structure)
    holder.build()
    assert len(holder) == 3
    assert len(holder.timeseries) == 3
    assert holder.priorities is None

    holder.set_taken_length(5)
    assert len(holder) == 12
    assert len(holder.timeseries) == 3
    assert holder.priorities == [4, 4, 4]

    os.system(f"rm -r '{temp_dir.absolute()}'")


def test_generation():
    temp_dir = Path('/dev/shm/.temp_test')
    os.system(f'rm -r "{temp_dir.absolute()}"')
    temp_dir.mkdir()
    for i in range(1000, 1100):
        np.save(str(temp_dir / f'{i}.npy'), np.ones(20 + i % 2 + i % 3) * i)

    eg = FileLoader(
        base_dir=temp_dir, all_file_structure=[{
          'channels': [f'{i}.npy'],
          'target_dtype': 'float16',
          'relative_priority': 1,
          'fs': 1562500,
        } for i in range(1000, 1100)])
    geg = GlobFileLoader(
        base_dir=temp_dir, file_structure={
          'target_dtype': 'float16',
          'relative_priority': 1,
          'fs': 1562500,
        })
    assert eg == geg
    eg.build()
    geg.build()
    for i in range(1000, 1100):
        assert np.all(eg.timeseries[i-1000].data_arr == i), i
        assert len(eg.timeseries[i-1000]) == 20 + i % 2 + i % 3, i
        assert eg.timeseries[i-1000].fs == 1562500
        assert eg.timeseries[i-1000].data_arr.dtype == 'float16'
        assert np.all(geg.timeseries[i-1000].data_arr == i), i
    assert eg == geg
    assert len(geg.timeseries) == 100, type(geg.timeseries)
    assert len(eg.timeseries[2]) == 20
    os.system(f'rm -r "{temp_dir.absolute()}"')
