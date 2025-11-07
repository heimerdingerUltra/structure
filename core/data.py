import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, Iterator, Optional
import math


class MemoryMappedDataset(Dataset):
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[callable] = None
    ):
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        
        self.transform = transform
        
        self.X.share_memory_()
        self.y.share_memory_()
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.X[idx], self.y[idx]
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x, y


class DistributedStratifiedSampler(Sampler):
    
    def __init__(
        self,
        targets: np.ndarray,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_bins: int = 10
    ):
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0
        
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        bins = np.linspace(targets.min(), targets.max(), num_bins + 1)
        self.bin_indices = [
            np.where((targets >= bins[i]) & (targets < bins[i + 1]))[0]
            for i in range(num_bins)
        ]
        
        if not self.drop_last:
            self.bin_indices[-1] = np.where(targets >= bins[-2])[0]
        
        self.num_samples = sum(len(idx) for idx in self.bin_indices) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        all_indices = []
        for bin_idx in self.bin_indices:
            if self.shuffle:
                perm = torch.randperm(len(bin_idx), generator=g).tolist()
                bin_idx = bin_idx[perm]
            all_indices.extend(bin_idx.tolist())
        
        if self.shuffle:
            perm = torch.randperm(len(all_indices), generator=g).tolist()
            all_indices = [all_indices[i] for i in perm]
        
        all_indices = all_indices[:self.total_size]
        
        indices = all_indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch


class Augmentation:
    
    @staticmethod
    def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
        if alpha <= 0:
            return x, y
        
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y
    
    @staticmethod
    def cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if alpha <= 0:
            return x, y
        
        lam = np.random.beta(alpha, alpha)
        batch_size, n_features = x.size()
        
        cut_size = int(n_features * (1 - lam))
        cut_start = np.random.randint(0, n_features - cut_size + 1)
        cut_end = cut_start + cut_size
        
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = x.clone()
        mixed_x[:, cut_start:cut_end] = x[index, cut_start:cut_end]
        
        lam = 1 - (cut_size / n_features)
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y
    
    @staticmethod
    def feature_dropout(x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        if p <= 0 or not x.requires_grad:
            return x
        
        mask = torch.bernoulli(
            torch.full_like(x, 1 - p)
        )
        
        return x * mask / (1 - p)
    
    @staticmethod
    def gaussian_noise(x: torch.Tensor, std: float = 0.02) -> torch.Tensor:
        if std <= 0:
            return x
        
        noise = torch.randn_like(x) * std
        return x + noise


class DataModule:
    
    def __init__(
        self,
        batch_size: int = 512,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        use_stratified_sampling: bool = True
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.use_stratified_sampling = use_stratified_sampling
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        n = len(X)
        indices = np.random.RandomState(seed).permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        return (
            (X[train_idx], y[train_idx]),
            (X[val_idx], y[val_idx]),
            (X[test_idx], y[test_idx])
        )
    
    def create_loaders(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
        augment: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        transform = None
        if augment:
            def train_transform(x):
                x = Augmentation.gaussian_noise(x, std=0.02)
                x = Augmentation.feature_dropout(x, p=0.05)
                return x
            transform = train_transform
        
        train_dataset = MemoryMappedDataset(X_train, y_train, transform=transform)
        val_dataset = MemoryMappedDataset(X_val, y_val)
        test_dataset = MemoryMappedDataset(X_test, y_test)
        
        if self.use_stratified_sampling:
            train_sampler = DistributedStratifiedSampler(
                y_train,
                num_replicas=1,
                rank=0,
                shuffle=True,
                num_bins=10
            )
            shuffle_train = False
        else:
            train_sampler = None
            shuffle_train = True
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=shuffle_train if train_sampler is None else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader
    
    def kfold_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        seed: int = 42
    ) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
        
        n = len(X)
        indices = np.random.RandomState(seed).permutation(n)
        
        fold_size = n // n_folds
        
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([
                indices[:val_start],
                indices[val_end:]
            ])
            
            yield (
                (X[train_idx], y[train_idx]),
                (X[val_idx], y[val_idx])
            )
