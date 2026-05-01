from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.collate import valve_collate
from src.models.CNN_model import ValveEventCNN
from src.data.sampler import BucketBatchSampler
from src.train.loss import MaskedBCELoss

@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  
    workers: int = 0
    prefetch_factor: int | None = 2
    persistent_workers: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels: int = 1
    dropout: float = 0.1
    
def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        "x": batch["x"].to(device, non_blocking=True),
        "y": batch["y"].to(device, non_blocking=True),
        "mask": batch["mask"].to(device, non_blocking=True),
        "lengths": batch["lengths"].to(device, non_blocking=True),
        "meta": batch["meta"],
    }

def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    prefetch_factor: int | None,
    persistent_workers: bool,
) -> DataLoader:
    loader_kwargs = {
        "collate_fn": valve_collate,
        "pin_memory": torch.cuda.is_available(),
        "num_workers": workers,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    
    if hasattr(dataset, "estimated_lengths"):
        batch_sampler = BucketBatchSampler(
            lengths=dataset.estimated_lengths,
            batch_size=batch_size,
            drop_last=False,
            shuffle=shuffle,
            max_length_spread=20_000,
            shuffle_window_multipler=1,
            debug=False,
        )

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **loader_kwargs,
    )

def train_one_epoch(
    model: ValveEventCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MaskedBCELoss,
    device: torch.device,
) -> float:
    model.train()

    running_loss = 0.0
    num_batches = 0


    progress_bar = tqdm(loader, desc="Training", leave=False)
    print()

    for step, batch in enumerate(progress_bar, start=1):
        if batch is None:
            continue

        batch = move_batch_to_device(batch, device)

        x = batch["x"]
        y = batch["y"]
        mask = batch["mask"]

        if step == 1:
            print("First batch x device", x.device)
            print("Model device", next(model.parameters()).device)

        optimizer.zero_grad(set_to_none=True)
        
        preds = model(x)
        loss = criterion(preds, y, mask)

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        num_batches += 1

        progress_bar.set_postfix({
            "loss:": f"{loss.item():.6f}"
        })

    if num_batches == 0:
        raise ValueError("Training loader produced zero batches")
    
    return running_loss / num_batches

@torch.no_grad()
def validate_one_epoch(
    model: ValveEventCNN,
    loader: DataLoader,
    criterion: MaskedBCELoss,
    device: torch.device,
) -> float:
    model.eval()

    running_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(loader, desc="Validation", leave=False)
    for batch in progress_bar:
        if batch is None:
            continue

        batch = move_batch_to_device(batch, device)

        x = batch["x"]
        y = batch["y"]
        mask = batch["mask"]

        preds = model(x)
        loss = criterion(preds, y, mask)

        running_loss += float(loss.item())
        num_batches += 1

        progress_bar.set_postfix({
            "val_loss": f"{loss.item():.6f}"
        })

    if num_batches == 0:
        raise ValueError("Validation loader produced 0 batches")
    
    return running_loss / num_batches


def fit(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: TrainConfig | None = None,
) -> tuple[ValveEventCNN, dict[str, list[float]]]:
    cfg = config or TrainConfig()
    device = torch.device(cfg.device)

    if cfg.batch_size <= 0:
        raise ValueError("batch size must be greater than 0")

    if cfg.epochs <= 0:
        raise ValueError("epochs must be greater than 0")
    
    if cfg.learning_rate <= 0:
        raise ValueError("learning rate must be greater than 0")
    
    if cfg.weight_decay < 0:
        raise ValueError("weight decay must be >= 0")
    
    if cfg.workers < 0:
        raise ValueError("workers must be >= 0")

    if cfg.prefetch_factor is not None and cfg.prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be > 0 when provided")

    if not 0.0 <= cfg.dropout < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0)")

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        workers=cfg.workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
    )

    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        workers=cfg.workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
    )

    model = ValveEventCNN(
        in_channels=cfg.in_channels,
        dropout=cfg.dropout,
    ).to(device)
    criterion = MaskedBCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    print("Using device", device)
    if device.type == "cuda":
        print("GPU", torch.cuda.get_device_name(0))
    print(
        "DataLoader workers",
        cfg.workers,
        "| prefetch_factor",
        cfg.prefetch_factor if cfg.workers > 0 else None,
        "| persistent_workers",
        cfg.persistent_workers if cfg.workers > 0 else False,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_seconds": [],
        "val_seconds": [],
        "epoch_seconds": [],
    }

    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    fit_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        train_seconds = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        val_seconds = time.perf_counter() - val_start
        epoch_seconds = time.perf_counter() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_seconds"].append(train_seconds)
        history["val_seconds"].append(val_seconds)
        history["epoch_seconds"].append(epoch_seconds)

        print(
            f"epoch={epoch}/{cfg.epochs} "
            f"train_loss={train_loss:.6f} - val_loss={val_loss:.6f} "
            f"train_time={train_seconds:.1f}s val_time={val_seconds:.1f}s "
            f"epoch_time={epoch_seconds:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state_dict is None:
        raise ValueError("No best model state was captured during training")

    history["total_seconds"] = [time.perf_counter() - fit_start]
    model.load_state_dict(best_state_dict)
    return model, history
