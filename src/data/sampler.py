from __future__ import annotations

import random
from typing import Iterator

from torch.utils.data import BatchSampler

class BucketBatchSampler(BatchSampler):
    def __init__(self, lengths: list[int], batch_size: int, drop_last: bool = False, shuffle: bool = True) -> None:
        if batch_size <= 0:
            raise ValueError("batch size must be > 0")
        
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
    
    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.lengths)))

        indices.sort(key=lambda index: self.lengths[index])

        window_size = self.batch_size * 100
        shuffle_indices = []

        for start in range(0, len(indices), window_size):
            window = indices[start:start + window_size]
            
            if self.shuffle:
                random.shuffle(window)

            shuffle_indices.extend(window)

        batches: list[list[int]] = []
        for start in range(0, len(shuffle_indices), self.batch_size):
            batch = shuffle_indices[start:start + self.batch_size]

            if len(batch) < self.batch_size and self.drop_last:
                continue
                
            batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        n = len(self.lengths)

        if self.drop_last:
            return n // self.batch_size

        return (n + self.batch_size - 1) // self.batch_size
