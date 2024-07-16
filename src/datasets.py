import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from glob import glob


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

        # 正規化のための平均と標準偏差を計算
        self.mean = self.calculate_mean()
        self.std = self.calculate_std()
        
    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        # Xデータの読み込みと正規化、ベースライン補正
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path))

        # ベースライン補正を適用
        X = X - X[:, 0, None]  # 各チャネルのベースラインを最初の値で補正
        
        # 正規化を適用
        X = (X - self.mean) / self.std

        # 被験者情報の読み込み
        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path))
        
        if self.split in ["train", "val"]:
            # ラベルの読み込み
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path))
            
            return X, y, subject_idx
        else:
            return X, subject_idx

    def calculate_mean(self):
        # データセット全体の平均を計算
        all_X = []
        for i in range(self.num_samples):
            X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
            X = np.load(X_path)
            all_X.append(X)
        all_X = np.concatenate(all_X, axis=1)
        return np.mean(all_X)

    def calculate_std(self):
        # データセット全体の標準偏差を計算
        all_X = []
        for i in range(self.num_samples):
            X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
            X = np.load(X_path)
            all_X.append(X)
        all_X = np.concatenate(all_X, axis=1)
        return np.std(all_X)
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]
