from src.data.sampler import BucketBatchSampler


def test_bucket_batch_sampler_groups_by_sorted_lengths_without_shuffle() -> None:
    sampler = BucketBatchSampler(
        lengths=[30, 10, 20, 40],
        batch_size=2,
        shuffle=False,
    )

    assert list(sampler) == [[1, 2], [0, 3]]


def test_bucket_batch_sampler_yields_each_index_once_with_shuffle() -> None:
    sampler = BucketBatchSampler(
        lengths=[30, 10, 20, 40, 50],
        batch_size=2,
        shuffle=True,
    )

    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert sorted(flattened) == [0, 1, 2, 3, 4]
    assert len(flattened) == len(set(flattened))


def test_bucket_batch_sampler_drop_last() -> None:
    sampler = BucketBatchSampler(
        lengths=[30, 10, 20, 40, 50],
        batch_size=2,
        drop_last=True,
        shuffle=False,
    )

    batches = list(sampler)

    assert batches == [[1, 2], [0, 3]]
    assert len(sampler) == 2
