from pkg_resources import get_distribution

__version__ = get_distribution('signalai').version

from signalai.signal import Signal, SignalProcessor, SignalDatasetsKeeper, read_npy, read_bin, read_audio
