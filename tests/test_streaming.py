from torch.utils.data import DataLoader

from syntrix.data.streaming import SyntheticStreamingDataset


def test_streaming(): assert sum(x.shape[0] for x,_ in DataLoader(SyntheticStreamingDataset(10),batch_size=2))==10
