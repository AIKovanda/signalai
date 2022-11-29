from pkg_resources import get_distribution

__version__ = get_distribution('signalai').version

from signalai.time_series import (Signal, Signal2D, read_audio, read_bin, read_npy)
