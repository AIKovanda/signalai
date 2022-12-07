from signalai.time_series import TimeSeries
from time_series_old import SeriesTrack


class SimpleStrategy(SeriesTrack):

    def next(self, length: int) -> TimeSeries:
        chosen_key = self._choose_class_key()
        return self.takers[chosen_key].next(length=length)


