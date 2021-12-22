from pkg_resources import get_distribution

__version__ = get_distribution('signalai').version

from signalai.timeseries import Signal, Signal2D, SeriesProcessor, SeriesDatasetsKeeper, read_npy, read_bin, read_audio
