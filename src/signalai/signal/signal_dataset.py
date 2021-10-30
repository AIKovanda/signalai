from signalai.signal.signal_indexer import SignalIndexer


class SignalDataset:
    def __init__(self, df, signal_loader, max_signal_length, split=None, next_after_samples=False, log=0):
        self.df = df
        self.signal_loader = signal_loader
        self.max_signal_length = max_signal_length
        self.next_after_samples = next_after_samples
        self.split = split
        if self.split is not None:
            self.df = self.df.query(f"split=='{self.split}'")
        self.log = log
        self.signal_indexer = SignalIndexer(self.df, self.max_signal_length, self.next_after_samples, log=self.log)
        self.total_interval_length = self.signal_indexer.total_interval_length

    def __next__(self):
        chosen_filename_id, relative_start = next(self.signal_indexer)
        return self.signal_loader.load(
            chosen_filename_id,
            start_relative=relative_start,
            max_interval_length=self.max_signal_length
        ), relative_start
