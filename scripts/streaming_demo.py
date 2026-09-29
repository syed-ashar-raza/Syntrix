from torch.utils.data import DataLoader
from syntrix.data.streaming import SyntheticStreamingDataset
loader=DataLoader(SyntheticStreamingDataset(100_000),batch_size=64); seen=0
for x,y in loader:
    seen+=x.size(0)
    if seen>=1024: break
print({"streamed_samples":seen,"dataset_materialized":False})
