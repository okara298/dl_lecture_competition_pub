import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from glob import glob

def scale_eeg(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.T).T
    return data_scaled

def baseline_correction(data):
    baseline = np.mean(data[:, :100], axis=1, keepdims=True)  # 最初の100サンプルをベースラインとして使用
    data_corrected = data - baseline
    return data_corrected

def preprocess_eeg(data):
    data = baseline_correction(data)
    data = scale_eeg(data)
    return data

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = np.load(X_path)
        X = preprocess_eeg(X)
        X = torch.from_numpy(X).float()  # Ensure the data type is float32
        
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path)).long()
        
        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            
            return X, y, subject_idx
        else:
            return X, subject_idx
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]
