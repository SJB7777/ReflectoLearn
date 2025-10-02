import h5py
import torch
from torch.utils.data import Dataset


class NlayerDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.len = len(f["R"])  # 길이만 저장

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as f:
            R = f["R"][idx].astype("float32")
            n_layer = f["n_layer"][idx].astype("int64")
        return torch.tensor(R), torch.tensor(n_layer)
