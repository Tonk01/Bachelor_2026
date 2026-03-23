from torch.utils.data import DataLoader
from src.data.dataset import ValveDataset


ds = ValveDataset("src/data/raw/raw1", "src/data/raw/raw2")

print("dataset size", len(ds))

loader = DataLoader(ds, batch_size = 4, shuffle = False)

batch = next(iter(loader))

print("x batch shape", batch["x"].shape)
print("y batch shape", batch["y"].shape)
print("meta keys", batch["meta"].keys())
print("eventids", batch["meta"]["eventid"])
print("paths[0]", batch["meta"]["path"][0])

# PYTHONPATH=. python scripts/test_dataset.py

