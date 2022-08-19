from typing import Union

import numpy as np

from signalai.timeseries import MultiSeries, SeriesTrack, TimeSeries


class SimpleStrategy(SeriesTrack):

    def next(self, length: int) -> Union[TimeSeries, MultiSeries]:
        chosen_key = self._choose_class_key()
        return self.takers[chosen_key].next(length=length)


class ToneStrategy(SeriesTrack):

    def next(self, length: int) -> Union[TimeSeries, MultiSeries]:
        """
        Tone compose, randomly taking tones into a MultiSeries.
        """
        count_min, count_max = self.params.get('tones_count_range', (0, 10))
        tones_count = np.random.choice(range(count_min, count_max + 1))
        chosen_classes = [self.relevant_classes[np.random.choice(len(self.relevant_classes))]
                          for _ in range(tones_count)]

        possible_starting_index = np.arange(*self.params.get('start_arange', [1]))
        possible_tone_length = np.arange(*self.params.get('tone_length_arange', [length]))

        series_dict = {}
        for class_name in self.relevant_classes:
            class_count = chosen_classes.count(class_name)
            if class_count == 0:
                continue

            final_intervals = []

            for i in range(class_count):
                starting_index = np.random.choice(possible_starting_index)
                tone_length = np.random.choice(possible_tone_length)
                ending_index = min(starting_index + tone_length, length)  # not to be over the edge
                for j, interval in enumerate(final_intervals):
                    if interval[0] <= starting_index <= interval[1]:
                        break
                    if interval[0] <= ending_index <= interval[1]:
                        break
                    if starting_index <= interval[0] and ending_index >= interval[1]:
                        break
                else:
                    final_intervals.append([starting_index, ending_index])

            assert len(final_intervals) > 0, "Something is wrong with chosen tone intervals."
            # print("____________")
            # print(class_name, final_intervals)
            series_dict[class_name] = MultiSeries(
                series=[
                    self.takers[class_name].next(interval[1] - interval[0]).margin_interval(
                        interval_length=length,
                        start_id=interval[0],
                    ) for interval in final_intervals]
            ).sum_channels()
            # print(sum(interval[1] - interval[0] for interval in final_intervals)*2)
            # print(np.sum(series_dict[class_name].time_map))
            # print(np.sum(series_dict[class_name].data_arr[0] != 0)*2)
            # print(np.sum(series_dict[class_name].data_arr[1] != 0)*2)

        # if len(series_dict) == 0:  # todo: zero signals
        #     print(len(self.relevant_classes))
        #     return MultiSeries(series=[Signal(data_arr=np.zeros((len(self.relevant_classes), length)), fs=self.fs)], class_order=self.relevant_classes)

        return MultiSeries(series=series_dict, class_order=self.relevant_classes)
