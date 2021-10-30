import numpy as np


class SignalIndexer:
    def __init__(self, df, max_signal_length, next_after_samples=False, log=0):
        self.df = df.drop_duplicates(subset="filename_id").reset_index(drop=True)
        self.max_signal_length = max_signal_length
        self.next_after_samples = next_after_samples
        self.p = self.df.interval_length.to_numpy()
        self.total_interval_length = np.sum(self.p)
        self.p = self.p / np.sum(self.p)
        self.indexing_end = {
            int(i.Index): max(0, int(i.interval_length)-self.max_signal_length) for i in self.df.itertuples()
        }
        self.id_now = 0
        self.file_now = 0
        self.log = log

    def __next__(self):
        if self.next_after_samples:
            id_ = self.id_now
            filename_id = self.df.loc[self.file_now, "filename_id"]
            self.id_now += self.next_after_samples
            if self.id_now > self.indexing_end[self.file_now]:
                self.id_now = 0
                if self.log > 0:
                    print("new cycle")

                self.file_now += 1
                if self.file_now >= len(self.indexing_end):
                    self.file_now = 0
            return filename_id, id_

        else:
            random_df_index = np.random.choice(self.df.index.to_list(), p=self.p)
            chosen_filename_id = self.df.at[random_df_index, 'filename_id']
            chosen_relative_start_id = np.random.choice(self.indexing_end[random_df_index]+1)
            return chosen_filename_id, chosen_relative_start_id
