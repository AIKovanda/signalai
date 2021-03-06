import librosa
import numpy as np
from pedalboard import \
    Chorus as PBChorus  # in the magnitude frequency response
from pedalboard import \
    Gain as \
    PBGain  # Increase or decrease the volume of a signal by applying a gain value (in decibels).
from pedalboard import \
    Phaser as \
    PBPhaser  # A 6 stage phaser that modulates first order all-pass filters to create sweeping notches
from pedalboard import \
    Reverb as \
    PBReverb  # Performs a simple reverb effect on a stream of audio data.

from signalai.timeseries import Signal, Signal2D, TimeSeries, Transformer
from signalai.tools.filters import butter_bandpass_filter
from signalai.tools.utils import by_channel

RETURN_CLASS = {
    2: Signal,
    3: Signal2D,
}


class Standardizer(Transformer):
    in_dim = 2

    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        mean = self.evaluated_params.get('mean', 0)
        std = self.evaluated_params.get('std', 1)

        new_x = (x - np.mean(x)) / np.std(x)
        return (new_x * std) + mean

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class Gain(Transformer):
    """
    gain_db: gain in decibels, can be negative
    """
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, fs: int) -> np.ndarray:
        gain_db = self._get_parameter_uniform('gain_db', default=[-10, 10])
        return PBGain(gain_db=gain_db)(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class Phaser(Transformer):
    """
    Does not take parameters.
    """
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, fs: int) -> np.ndarray:
        return PBPhaser()(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class Chorus(Transformer):
    """
    centre_delay_ms: between 7 and 8 is reminds Chorus the most
    """
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, fs: int) -> np.ndarray:
        centre_delay_ms = self._get_parameter_uniform('centre_delay_ms', default=[7., 8.])
        return PBChorus(centre_delay_ms=centre_delay_ms)(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class Reverb(Transformer):
    """
    Parameters in range [0,1]:
        room_size
        damping
        wet_level
        dry_level
        freeze_mode
    """
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, fs: int) -> np.ndarray:
        room_size = self._get_parameter_uniform('room_size', default=[0., 1.])
        damping = self._get_parameter_uniform('damping', default=[0., 1.])
        wet_level = self._get_parameter_uniform('wet_level', default=[0., 1.])
        dry_level = self._get_parameter_uniform('dry_level', default=[0., 1.])
        freeze_mode = self._get_parameter_uniform('freeze_mode', default=[0., .1])
        return PBReverb(
            room_size=room_size,
            damping=damping,
            wet_level=wet_level,
            dry_level=dry_level,
            freeze_mode=freeze_mode,
        )(input_array=x, sample_rate=fs)  # todo .astype(x.dtype)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class BandPassFilter(Transformer):
    """
    low_cut - None or frequency
    high_cut - None or frequency
    """
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, fs: int) -> np.ndarray:
        low_cut = self.evaluated_params.get('low_cut')
        high_cut = self.evaluated_params.get('high_cut')

        return butter_bandpass_filter(x, low_cut, high_cut, fs)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        return Signal(
            data_arr=self.transform_npy(x.data_arr, fs=x.fs),
            time_map=x.time_map,
            meta=x.meta,
            logger=x.logger,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class ChannelJoiner(Transformer):
    """
    channels: list containing list of integers
    """
    in_dim = 2

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        choose_channels = self.evaluated_params.get("choose_channels")
        if choose_channels is not None:
            channels = choose_channels[np.random.choice(len(choose_channels))]
        else:
            channels = self.evaluated_params.get("channels", [list(range(x.data_arr.shape[0]))])
        return x.take_channels(channels=channels)

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class Lambda(Transformer):
    """
    channels: list containing list of integers
    """

    def transform(self, x: TimeSeries) -> TimeSeries:
        return self.evaluated_params.get("lambda")(x)

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class TimeMapScale(Transformer):
    """
    channels: list containing list of integers
    """

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        target_length = self.evaluated_params.get("target_length")
        if target_length is None:
            target_length = len(x) * self.evaluated_params.get("scale")

        time_map = x.time_map.astype(int)
        # nearest
        new_time_map = time_map[:, np.round(np.linspace(0, time_map.shape[-1] - 1, int(target_length))).astype(int)]
        return TimeSeries(time_map=new_time_map, meta=x.meta)

    def original_signal_length(self, length: int, fs: int = None) -> int:  # todo: another approach
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class STFT(Transformer):
    in_dim = 2

    @by_channel
    def transform_npy(self, x: np.ndarray, split_complex=False) -> np.ndarray:
        center = self.evaluated_params.get('center', False)
        n_fft = self.evaluated_params.get('n_fft', 256)
        hop_length = self.evaluated_params.get('hop_length')
        # _, _, Zxx = scipy_signal.stft(x[0], fs=fs)
        Zxx = librosa.stft(x[0], center=center, n_fft=n_fft, hop_length=hop_length)
        if split_complex:
            return np.moveaxis(Zxx.view('(2,)float32'), -1, 0)

        return np.expand_dims(Zxx, 0)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        if self.evaluated_params.get('phase_as_meta', True):
            Zxx = self.transform_npy(x.data_arr, split_complex=False)
            data_arr = np.abs(Zxx)  # magnitude
            meta = x.meta.copy() | {'phase': np.angle(Zxx)}
        else:
            data_arr = self.transform_npy(x.data_arr, split_complex=True)
            meta = x.meta

        return Signal2D(
            data_arr=data_arr,
            time_map=x.time_map,  # todo
            meta=meta,
            logger=x.logger,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True


class ISTFT(Transformer):
    in_dim = 3

    @by_channel
    def transform_npy(self, x: np.ndarray) -> np.ndarray:
        # return scipy_signal.istft(x)[1]
        center = self.evaluated_params.get('center', False)
        hop_length = self.evaluated_params.get('hop_length')
        return np.expand_dims(librosa.istft(x[0], hop_length=hop_length, center=center), 0)

    def transform_timeseries(self, x: TimeSeries) -> TimeSeries:
        meta = x.meta.copy()  # todo: without copy
        phase = meta.pop('phase', None)
        data_arr = x.data_arr

        if self.evaluated_params.get('phase_as_meta', True):
            assert phase is not None, "Phase must be included in the input information"
            Zxx = data_arr * np.exp(1j * phase)
        else:
            Zxx = data_arr[0::2] + 1j * data_arr[1::2]

        s_data_arr = self.transform_npy(Zxx)
        return Signal(
            data_arr=s_data_arr,
            time_map=x.time_map,  # todo
            meta=meta,
            logger=x.logger,
            fs=x.fs,
        )

    def original_signal_length(self, length: int, fs: int = None) -> int:
        return length

    @property
    def keeps_signal_length(self) -> bool:
        return True
