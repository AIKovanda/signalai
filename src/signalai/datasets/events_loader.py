from pathlib import Path

import pandas as pd

from signalai import read_audio
from signalai.time_series_gen import TimeSeriesGen
from time_series import TimeSeries


class EventsGen(TimeSeriesGen):
    def __len__(self):
        pass

    def _getitem(self, item: int) -> TimeSeries:
        pass

    def is_infinite(self) -> bool:
        return True

    def get_class_objects(self) -> list:
        generated_result = []
        class_structure = self.params["class_structure"]

        for superclass_name, structure in class_structure.items():

            all_csv = pd.read_csv(structure['classes_file'])
            unique_tones = all_csv['tone'].drop_duplicates().to_list()

            files = list(Path('/'.join(structure['filename'].split('/')[:-1])).glob(
                structure['filename'].split('/')[-1]))
            assert len(files) > 0, 'There is no file!'
            all_signals = {}
            for file in files:
                all_signal = read_audio(file, dtype=self.params.get("target_dtype"))
                assert all_signal.fs is not None
                all_signal.update_meta(self.params.get("meta", {}))
                all_signals[str(file)] = all_signal

            for unique_tone in unique_tones:
                sub_df = all_csv.query(f"tone=='{unique_tone}'")
                signals = []
                for row in sub_df.itertuples():
                    for s_name, all_signal in all_signals.items():
                        s = all_signal.crop(interval=(row.sample_start, row.sample_end))
                        # s.trim_(threshold=1e-7)  # found around 1000 empty samples in the beginning
                        s.update_meta({'force': row.force, 'origin': s_name})
                        signals.append(s)

                assert len(signals) > 0, 'Something went wrong here!'
                # generated_result.append(SeriesClass(
                #     series=signals, class_name=unique_tone, superclass_name=superclass_name, logger=self.logger,
                #     purpose=self.purpose,
                # ))

        return generated_result
