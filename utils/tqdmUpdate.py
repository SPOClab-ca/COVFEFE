from tqdm import tqdm

class TqdmUpdate(tqdm):
    def update(self, done, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.n = done
        super().refresh()