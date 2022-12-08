import os
from pathlib import Path

import numpy as np

from signalai.datasets.file_loader import FileLoader
from time_series import Signal


def test_simple_load():
    temp_dir = Path('temp')
    temp_dir.mkdir(exist_ok=True)
    for i in range(1, 10):
        np.save(str((temp_dir / f'{i}.npy').absolute()), np.arange(0, 20)+i/10)
    file_structure = [
        {'channels': ['1.npy', '2.npy', '3.npy'], "meta": {'c': 1}},
        {'channels': ['4.npy', '5.npy', '6.npy'], "meta": {'c': 1}},
        {'channels': ['7.npy', '8.npy', '9.npy'], "meta": {'c': 1}},
    ]
    holder = FileLoader(base_dir=temp_dir, file_structure=file_structure)
    holder.build()
    assert len(holder) == 3
    assert len(holder.timeseries) == 3
    assert holder.priorities is None
    for i in range(3):
        assert holder.timeseries[i] == Signal(
            data_arr=np.vstack([np.arange(0, 20)+(1+3*i)/10, np.arange(0, 20)+(2+3*i)/10, np.arange(0, 20)+(3+3*i)/10]),
            meta={'c': 1})

    file_structure = [
        {'channels': ['1.npy', '2.npy', '3.npy'], "meta": {'c': 1}, 'fs': 50, "priority": 5},
        {'channels': ['4.npy', '5.npy', '6.npy'], "meta": {'c': 1}, 'fs': 50, "priority": 1},
        {'channels': ['7.npy', '8.npy', '9.npy'], "meta": {'c': 1}, 'fs': 50, "priority": 2},
    ]
    holder = FileLoader(base_dir=temp_dir, file_structure=file_structure)
    holder.build()
    assert len(holder) == 8
    assert len(holder.timeseries) == 3
    assert holder.priorities == [5, 1, 2]
    for i in range(3):
        assert holder.timeseries[i] == Signal(
            data_arr=np.vstack([np.arange(0, 20)+(1+3*i)/10, np.arange(0, 20)+(2+3*i)/10, np.arange(0, 20)+(3+3*i)/10]),
            meta={'c': 1}, fs=50)

    holder = FileLoader(base_dir=temp_dir, file_structure=file_structure, num_workers=3)
    holder.build()
    assert len(holder) == 8
    assert len(holder.timeseries) == 3
    assert holder.priorities == [5, 1, 2]
    for i in range(3):
        assert holder.timeseries[i] == Signal(
            data_arr=np.vstack([np.arange(0, 20)+(1+3*i)/10, np.arange(0, 20)+(2+3*i)/10, np.arange(0, 20)+(3+3*i)/10]),
            meta={'c': 1}, fs=50)

    os.system(f"rm -r '{temp_dir.absolute()}'")
