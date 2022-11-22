import numpy as np
import pytest

from timeseries import Signal, TimeSeries


def test_operators():
    s1 = TimeSeries(data_arr=np.array([[1, 2, 3, 3.5],
                                       [2, 3, 3.5, 1]]),
                    time_map=np.array([[1, 0, 1, 0],
                                       [0, 1, 0, 1]]),
                    meta={'a': 0, 'b': 'asdf'}, fs=50)

    s2 = TimeSeries(data_arr=np.array([[4, 5, 6, 6.5],
                                       [5, 6, 6.5, 4]]),
                    time_map=np.array([[0, 1, 1, 0],
                                       [0, 0, 1, 1]]),
                    meta={'a': 1, 'b': 'asdf'}, fs=50)

    assert np.all(s1.time_map == np.array([[True, False, True, False], [False, True, False, True]]))
    assert np.all(s2.time_map == np.array([[False, True, True, False], [False, False, True, True]]))

    assert s1 + s2 == TimeSeries(data_arr=np.array([[5, 7, 9, 10],
                                                    [7, 9, 10, 5]]),
                                 time_map=np.array([[1, 1, 1, 0],
                                                    [0, 1, 1, 1]]),
                                 meta={'b': 'asdf'}, fs=50)

    assert s1 & s2 == TimeSeries(data_arr=np.array([[1, 2, 3, 3.5, 4, 5, 6, 6.5],
                                                    [2, 3, 3.5, 1, 5, 6, 6.5, 4]]),
                                 time_map=np.array([[1, 0, 1, 0, 0, 1, 1, 0],
                                                    [0, 1, 0, 1, 0, 0, 1, 1]]),
                                 meta={'b': 'asdf'}, fs=50)

    assert s1 | s2 == TimeSeries(data_arr=np.array([[1, 2, 3, 3.5],
                                                    [2, 3, 3.5, 1],
                                                    [4, 5, 6, 6.5],
                                                    [5, 6, 6.5, 4]]),
                                 time_map=np.array([[1, 0, 1, 0],
                                                    [0, 1, 0, 1],
                                                    [0, 1, 1, 0],
                                                    [0, 0, 1, 1]]),
                                 meta={'b': 'asdf'}, fs=50)


def test_crop():
    s1 = TimeSeries(data_arr=np.array([[1, 2, 3, 3.5, 4, 5, 6, 6.5],
                                       [2, 3, 3.5, 1, 5, 6, 6.5, 4]]),
                    time_map=np.array([[1, 0, 1, 0, 0, 1, 1, 0],
                                       [0, 1, 0, 1, 0, 0, 1, 1]]),
                    meta={'b': 'asdf'}, fs=50)

    assert s1.crop([2, 5]) == TimeSeries(data_arr=np.array([[3, 3.5, 4],
                                                            [3.5, 1, 5]]),
                                         time_map=np.array([[1, 0, 0],
                                                            [0, 1, 0]]),
                                         meta={'b': 'asdf'}, fs=50)


def test_different_len_signals():
    s1 = Signal(data_arr=np.array([1, 2, 3]), time_map=np.array([1, 0, 1]), meta={'a': 0, 'b': 'asdf'}, fs=50)
    s2 = Signal(data_arr=np.array([4, 5, 6, 6.5]), time_map=np.array([0, 1, 1, 0]), meta={'a': 1, 'b': 'asdf'}, fs=50)

    joined_s = s1 & s2

    assert np.all(joined_s.data_arr == np.array([1, 2, 3, 4, 5, 6, 6.5]))
    assert np.all(joined_s.time_map == np.array([True, False, True, False, True, True, False]))
    assert joined_s.fs == 50
    assert joined_s.meta == {'b': 'asdf'}

    with pytest.raises(ValueError) as exec_info:
        s1 + s2
    with pytest.raises(ValueError) as exec_info:
        s1 | s2


def test_different_freq_signals():
    s1 = Signal(data_arr=np.array([1, 2, 3]), time_map=np.array([1, 0, 1]), meta={'a': 0, 'b': 'asdf'}, fs=55)
    s2 = Signal(data_arr=np.array([4, 5, 6]), time_map=np.array([0, 1, 1]), meta={'a': 1, 'b': 'asdf'}, fs=49)

    with pytest.raises(ValueError) as exec_info:
        s1 + s2
    with pytest.raises(ValueError) as exec_info:
        s1 & s2
    with pytest.raises(ValueError) as exec_info:
        s1 | s2

    s1 = Signal(data_arr=np.array([1, 2, 3]), time_map=np.array([1, 0, 1]), meta={'a': 0, 'b': 'asdf'}, fs=55)
    s2 = Signal(data_arr=np.array([4, 5, 6]), time_map=np.array([0, 1, 1]), meta={'a': 1, 'b': 'asdf'}, fs=None)

    with pytest.raises(ValueError) as exec_info:
        s1 + s2
    with pytest.raises(ValueError) as exec_info:
        s1 & s2
    with pytest.raises(ValueError) as exec_info:
        s1 | s2

    with pytest.raises(ValueError) as exec_info:
        s2 + s1
    with pytest.raises(ValueError) as exec_info:
        s2 & s1
    with pytest.raises(ValueError) as exec_info:
        s2 | s1

    s1 = Signal(data_arr=np.array([1, 2, 3]), time_map=np.array([1, 0, 1]), meta={'a': 0, 'b': 'asdf'}, fs=None)
    s2 = Signal(data_arr=np.array([4, 5, 6]), time_map=np.array([0, 1, 1]), meta={'a': 1, 'b': 'asdf'}, fs=None)
    s1 + s2
    s1 & s2
    s1 | s2


def test_take_channels():
    s1 = Signal(data_arr=np.array([[0, 0, 0.01, 0.250, 5, 0, 5, 0.0001, 0],
                                   [0, 0, 0.01, 0.001, 5, 0, 5, 0.5, 0.001]]),
                time_map=np.array([[0, 0, 1, 0, 1, 0, 1, 0, 1],
                                   [0, 1, 0, 1, 0, 1, 0, 1, 0]]),
                meta={'a': 0, 'b': 'asdf'}, fs=50)

    assert s1.channels_count == 2

    assert s1.take_channels() == s1
    assert s1.take_channels([1, 1]) == Signal(data_arr=np.array([[0, 0, 0.01, 0.001, 5, 0, 5, 0.5, 0.001],
                                                                 [0, 0, 0.01, 0.001, 5, 0, 5, 0.5, 0.001]]),
                                              time_map=np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0],
                                                                 [0, 1, 0, 1, 0, 1, 0, 1, 0]]),
                                              meta={'a': 0, 'b': 'asdf'}, fs=50)

    assert s1.take_channels([1, [0, 1]]) == Signal(data_arr=np.array([[0, 0, 0.01, 0.001, 5, 0, 5, 0.5, 0.001],
                                                                      [0, 0, 0.02, 0.251, 10, 0, 10, 0.5001, 0.001]]),
                                                   time_map=np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0],
                                                                      [0, 1, 1, 1, 1, 1, 1, 1, 1]]),
                                                   meta={'a': 0, 'b': 'asdf'}, fs=50)


def test_trim():
    s1 = Signal(data_arr=np.array([[0, 0, 0.01, 0.250, 5, 0, 5, 0.0001, 0],
                                   [0, 0, 0.01, 0.001, 5, 0, 5, 0.5, 0.001]]),
                time_map=np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],
                                   [0, 1, 0, 1, 0, 1, 0, 1, 0]]),
                meta={'a': 0, 'b': 'asdf'}, fs=50)
    s2 = s1.trim(0.02)
    s1.trim_(0.02)
    assert s1 == s2

    assert np.all(s1.data_arr == np.array([[0.250, 5, 0, 5, 0.0001],
                                           [0.001, 5, 0, 5, 0.5]]))
    assert np.all(s1.time_map == np.array([[False, True, False, True, False],
                                           [True, False, True, False, True]]))
    assert s1.fs == 50
    assert s1.meta == {'a': 0, 'b': 'asdf'}


def test_margin_interval():
    s1 = Signal(data_arr=np.array([[0, 0, 0.01, 0.250, 5, 0, 5, 0.0001, 0],
                                   [0, 0, 0.01, 0.001, 5, 0, 5, 0.5, 0.001]]),
                time_map=np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],
                                   [0, 1, 0, 1, 0, 1, 0, 1, 0]]),
                meta={'a': 0, 'b': 'asdf'}, fs=50)

    assert s1.margin_interval(9, 2) == Signal(data_arr=np.array([[0, 0, 0, 0, 0.01, 0.250, 5, 0, 5],
                                                                 [0, 0, 0, 0, 0.01, 0.001, 5, 0, 5]]),
                                              time_map=np.array([[0, 0, 1, 0, 1, 0, 1, 0, 1],
                                                                 [0, 0, 0, 1, 0, 1, 0, 1, 0]]),
                                              meta={'a': 0, 'b': 'asdf'}, fs=50)

    assert s1.margin_interval(13, 2) == Signal(data_arr=np.array([[0, 0, 0, 0, 0.01, 0.250, 5, 0, 5, 0.0001, 0, 0, 0],
                                                                  [0, 0, 0, 0, 0.01, 0.001, 5, 0, 5, 0.5, 0.001, 0,
                                                                   0]]),
                                               time_map=np.array([[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
                                                                  [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]]),
                                               meta={'a': 0, 'b': 'asdf'}, fs=50)

    assert s1.margin_interval(4, -3) == Signal(data_arr=np.array([[0.250, 5, 0, 5],
                                                                  [0.001, 5, 0, 5]]),
                                               time_map=np.array([[0, 1, 0, 1],
                                                                  [1, 0, 1, 0]]),
                                               meta={'a': 0, 'b': 'asdf'}, fs=50)

    assert s1.margin_interval(8, -3) == Signal(data_arr=np.array([[0.250, 5, 0, 5, 0.0001, 0, 0, 0],
                                                                  [0.001, 5, 0, 5, 0.5, 0.001, 0, 0]]),
                                               time_map=np.array([[0, 1, 0, 1, 0, 1, 0, 0],
                                                                  [1, 0, 1, 0, 1, 0, 0, 0]]),
                                               meta={'a': 0, 'b': 'asdf'}, fs=50)
