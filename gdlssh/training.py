from __future__ import annotations

import csv
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .fs import ensure_dir
from .importing import import_from_string

LOGGER = logging.getLogger(__name__)


class LossLogger:
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path).expanduser()
        self.rows: list[tuple[int, float, float]] = []

    def append(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self.rows.append((epoch, train_loss, val_loss))
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            writer.writerows(self.rows)


def build_model(factory_spec: str, kwargs: dict) -> torch.nn.Module:
    factory = import_from_string(factory_spec)
    return factory(**kwargs)


def build_loss(loss_spec: str, kwargs: dict) -> torch.nn.Module:
    factory = import_from_string(loss_spec)
    return factory(**kwargs)


def train_model(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs: int, checkpoint_dir: str | Path, checkpoint_prefix: str, loss_csv_path: str | Path, device: torch.device) -> None:
    checkpoint_root = ensure_dir(checkpoint_dir)
    loss_logger = LossLogger(loss_csv_path)
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_steps = 0
        for invar, outvar in train_loader:
            inputs = invar.squeeze(0).to(device=device, dtype=torch.float32)
            targets = outvar.squeeze(0).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item())
            train_steps += 1
        model.eval()
        val_total = 0.0
        val_steps = 0
        with torch.no_grad():
            for invar, outvar in val_loader:
                inputs = invar.squeeze(0).to(device=device, dtype=torch.float32)
                targets = outvar.squeeze(0).to(device=device, dtype=torch.float32)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                val_total += float(loss.item())
                val_steps += 1
        train_loss = train_total / max(train_steps, 1)
        val_loss = val_total / max(val_steps, 1)
        loss_logger.append(epoch, train_loss, val_loss)
        checkpoint_path = checkpoint_root / f"{checkpoint_prefix}_weights_epoch{epoch}"
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, checkpoint_path)
        LOGGER.info("epoch=%s train_loss=%.6f val_loss=%.6f saved=%s", epoch, train_loss, val_loss, checkpoint_path)
