import torch
from torch.utils.data import IterableDataset

class SyntheticStreamingDataset(IterableDataset):
    """Generates samples lazily instead of materializing the dataset in RAM."""
    def __init__(self, samples=100_000, image_size=28, classes=10):
        self.samples, self.image_size, self.classes = samples, image_size, classes
    def __iter__(self):
        for _ in range(self.samples):
            yield torch.randn(1, self.image_size, self.image_size), torch.randint(0, self.classes, (1,)).item()
