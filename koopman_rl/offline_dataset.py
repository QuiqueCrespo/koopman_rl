"""
Offline dataset loader for LEWM HDF5 files.

Expected HDF5 layout (decompressed from reacher.tar.zst etc.):
    /pixels   [N, H, W, C]  uint8
    /action   [N, action_dim]  float32 (or float64)

Each item is a consecutive (obs_t, obs_{t+1}, action_t) triple.
Dataset length = N - 1.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class LEWMDataset(Dataset):
    """
    Wraps a decompressed LEWM HDF5 file.

    Args:
        data_path: path to the .h5 file
        obs_key:   HDF5 key for pixel observations (default "pixels")
        act_key:   HDF5 key for actions (default "action")
    """

    def __init__(self, data_path: str, obs_key: str = "pixels", act_key: str = "action"):
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required: pip install h5py")

        with h5py.File(data_path, "r") as f:
            # Load everything into RAM as numpy arrays for fast random access.
            # For very large datasets, replace with mmap or lazy h5py reads.
            obs  = f[obs_key][:]   # [N, H, W, C] uint8
            acts = f[act_key][:]   # [N, action_dim]

        # Convert pixels: [N, H, W, C] uint8 → [N, C, H, W] float32 in [0, 1]
        self.obs  = torch.from_numpy(obs).permute(0, 3, 1, 2).float().div_(255.0)
        self.acts = torch.from_numpy(acts.astype(np.float32))
        self._len = len(self.obs) - 1   # consecutive pairs

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        return self.obs[idx], self.obs[idx + 1], self.acts[idx]
