import h5py

import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


class ThicknessDataset(Dataset):
    """
    Dataset for thickness regression given n_layer.
    HDF5 format assumed:
        /R -> reflectivity curves (N, q_points)
        /thickness -> (N, max_layers) array of target thicknesses
        /n_layer -> (N,) array with number of layers per sample

    Only samples with exact n_layer are included.
    """

    def __init__(self, h5_path: str, transform=None):
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform

        # 안전하게 Path 처리
        self.h5_path = Path(h5_path)

        # Dataset 초기화 시점에서 n_layer 필터링
        with h5py.File(self.h5_path, "r") as f:
            self.input_dim = f["R"].shape[1]  # q_points 길이
            self._len = f["R"].shape[0]  # 전체 샘플 수
            self.n_layer = f["thickness"].shape[1]

    def __len__(self):
        return self._len

    def __getitem__(self, idx):

        with h5py.File(self.h5_path, "r") as f:
            R = f["R"][idx].astype(np.float32)
            thickness = f["thickness"][idx].astype(np.float32)

        R = torch.tensor(R, dtype=torch.float32)
        y = torch.tensor(thickness, dtype=torch.float32)

        if self.transform:
            R = self.transform(R)

        return R, y
