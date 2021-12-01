from pkg_resources import get_distribution

__version__ = get_distribution('taskchain').version

from signalai.signal import Signal, SignalProcessor, SignalDatasetsKeeper
