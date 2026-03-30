from abc import ABC, abstractmethod
import random

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import tiktoken
from datasets import load_dataset

from config import ModelConfig


class LLMDataset(Dataset, ABC):
    def __init__(self, config: ModelConfig):
        self.enc     = tiktoken.get_encoding(config.encoding_name)
        self.seq_len = config.seq_len

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor:
        # returns: tokens (seq_len + 1,) — model splits into input/target
        ...


class OpenWebTextDataset(LLMDataset):
    def __init__(self, config: ModelConfig, split: str = "train"):
        super().__init__(config)
        self.data = load_dataset("Skylion007/openwebtext", split=split)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # skip documents shorter than one full window; rare but avoids padding
        tokens = self.enc.encode(self.data[idx]["text"])
        while len(tokens) < self.seq_len + 1:
            idx    = (idx + 1) % len(self.data)
            tokens = self.enc.encode(self.data[idx]["text"])

        start = random.randint(0, len(tokens) - self.seq_len - 1)
        chunk = tokens[start : start + self.seq_len + 1]
        return torch.tensor(chunk, dtype=torch.long)              # (seq_len + 1,)


def get_dataset(config: ModelConfig, split: str = "train") -> LLMDataset:
    return OpenWebTextDataset(config, split=split)


def build_dataloader(
    dataset: LLMDataset,
    batch_size: int,    # per-GPU
    rank: int,
    world_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> tuple[DataLoader, DistributedSampler]:
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return loader, sampler


if __name__ == "__main__":
    data = load_dataset("Skylion007/openwebtext", split='train')