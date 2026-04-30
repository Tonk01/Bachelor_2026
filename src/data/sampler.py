from __future__ import annotations

import random
from typing import Iterator
from collections import Counter

from torch.utils.data import BatchSampler

class BucketBatchSampler(BatchSampler):
    def __init__(self, 
        lengths: list[int], 
        batch_size: int, drop_last: 
        bool = False, 
        shuffle: bool = True,
        max_length_spread: int = 20_000,
        shuffle_window_multipler: int = 1,
        debug: bool = False,
        ) -> None:

        if batch_size <= 0:
            raise ValueError("batch size must be > 0")
        
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.max_length_spread = max_length_spread
        self.shuffle_window_multiplier = shuffle_window_multipler
        self.debug = debug

    def batch_size_for_length(self, length: int) -> int:
        if length > 200_000:
            return min(self.batch_size, 6)
        
        if length > 100_000:
            return min(self.batch_size, 8)
        
        return self.batch_size
    
    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda index: self.lengths[index])

        window_size = self.batch_size
        shuffle_indices = []

        for start in range(0, len(indices), window_size):
            window = indices[start:start + window_size]
            
            if self.shuffle:
                random.shuffle(window)

            shuffle_indices.extend(window)

        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_min: int | None = None
        current_max: int | None = None
        current_target_batchsize: int | None = None

        for index in shuffle_indices:
            length = self.lengths[index]
            sample_target_batchsize = self.batch_size_for_length(length)

            if not current_batch:
                current_batch = [index]
                current_min = length
                current_max = length
                current_target_batchsize = sample_target_batchsize
                continue

            next_min = min(current_min, length)
            next_max = max(current_max, length)
            next_target_batchsize = min(current_target_batchsize, sample_target_batchsize)

            batch_full = (len(current_batch) + 1) > next_target_batchsize
            spread_too_large = (next_max - next_min) > self.max_length_spread

            if batch_full or spread_too_large:
                if len(current_batch) == current_target_batchsize or not self.drop_last:
                    batches.append(current_batch)

                current_batch = [index]
                current_min = length
                current_max = length
                current_target_batchsize = sample_target_batchsize
            else: 
                current_batch.append(index)
                current_min = next_min
                current_max = next_max
                current_target_batchsize = next_target_batchsize

        if current_batch and (len(current_batch) == current_target_batchsize or not self.drop_last):
            batches.append(current_batch)

        if self.shuffle:
            random.shuffle(batches)

        # for debugging of bucketing and batches. 
        if self.debug and batches:
            all_spreads = [
                max(self.lengths[i] for i in batch) - min(self.lengths[i] for i in batch)
                for batch in batches
                if batch
            ]

            batch_sizes = [len(batch) for batch in batches]
            batch_size_counts = Counter(batch_sizes)

            print("\nBucketBatchSampler debug:")
            print(f"  batches: {len(batches)}")
            print(f"  target_batch_size: {self.batch_size}")
            print(f"  min_actual_batch_size: {min(batch_sizes)}")
            print(f"  max_actual_batch_size: {max(batch_sizes)}")
            print(f"  shuffle_window: {window_size}")
            print(f"  max_length_spread_allowed: {self.max_length_spread:,}")
            print(f"  max_actual_spread: {max(all_spreads):,}")
            print(f"  mean_actual_spread: {sum(all_spreads) / len(all_spreads):,.0f}")

            print("  actual_batch_size_counts:")
            for size in sorted(batch_size_counts):
                print(f"    n={size}: {batch_size_counts[size]} batches")

            for batch in batches[:20]:
                lens = [self.lengths[i] for i in batch]
                print(f"  example: {min(lens):,} - {max(lens):,} | n={len(batch)}")

            print()

        yield from batches

    def __len__(self) -> int:
        n = len(self.lengths)

        if self.drop_last:
            return n // self.batch_size

        return (n + self.batch_size - 1) // self.batch_size
