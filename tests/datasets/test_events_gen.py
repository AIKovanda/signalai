import os
from pathlib import Path

import numpy as np
from signalai.datasets.events_gen import EventsGen


def test_generation():
    temp_dir = Path('/dev/shm/.temp_test')
    os.system(f'rm -r "{temp_dir.absolute()}"')
    temp_dir.mkdir()
    (temp_dir / '1').mkdir()
    (temp_dir / '2').mkdir()
    for i in range(1000, 1100):
        np.save(str(temp_dir / '1' / f'{i}.npy'), np.ones(20 + i % 2 + i % 3))
        np.save(str(temp_dir / '2' / f'{i}.npy'), np.ones(20 + i % 2 + i % 3) * 2)

    eg = EventsGen(
        paths={'1': temp_dir / '1', '2': temp_dir / '2'}, file_structure={
            'target_dtype': 'float16',
            'relative_priority': 1,
            'fs': 1562500,
        }, event_count_range=[0, 5], start_arange=[0, 128], event_length_arange=[10, 11],
    )
    eg.set_taken_length(1024)
    eg.build()
    assert len(eg.input_ts_gen_kwargs) == 2
    assert len(eg.input_ts_gen_kwargs['1'].timeseries) == 100
    assert eg.input_ts_gen_kwargs['1'].timeseries[0].data_arr[0, 0] == 1
    assert len(eg.input_ts_gen_kwargs['2'].timeseries) == 100
    assert eg.input_ts_gen_kwargs['2'].timeseries[0].data_arr[0, 0] == 2

    for i in range(400):
        item = eg.getitem(i).data_arr
        assert item.shape == (2, 1024)
        assert np.all(item[:, 138:] == 0)
        fake_item = np.zeros_like(item)
        np_where = np.where(np.logical_and(item[:, :-1] != 0, item[:, 1:] == 0))
        for x, y in zip(*np_where):
            fake_item[x, y-9: y+1] = x + 1  # y-th is still non-zero

        assert len(np_where[0]) <= 5, np_where[0]
        assert np.all(item == fake_item)

    os.system(f'rm -r "{temp_dir.absolute()}"')
