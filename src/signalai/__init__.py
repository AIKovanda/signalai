from pkg_resources import get_distribution

__version__ = get_distribution('signalai').version

from signalai.timeseries import (SeriesDatasetsKeeper, SeriesProcessor, Signal,
                                 Signal2D, read_audio, read_bin, read_npy)
