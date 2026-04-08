import argparse
import copy
import csv
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.append("src")

import torch
from torch.utils.data import DataLoader

from src.config_utils import ensure_dir, load_config
from src.logging_utils import configure_logging
from src.mlflow_utils import MLflowTracker
from src.oceantaco import batch_to_model_tensors, build_dataset, get_collate_fn
from src.pytorch_losses import torch_masked_mse
from src.simvp_model import SimVP_Model_no_skip_configurable

LOGGER = logging.getLogger(__name__)


class LossLogger:
    def __init__(self, filename: Path):
        self.filename = filename
        self.rows = []

    def log(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self.rows.append((epoch, train_loss, val_loss))
        with self.filename.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            writer.writerows(self.rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimVP on OceanTACO data.")
    parser.add_argument(
        "--config",
        default="configs/oceantaco.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config):
    grid_cfg = config["data"]["grid"]
    model_cfg = config["model"]
    sequence_length = int(config["data"]["sequence_length"])
    input_channels = len(config["data"]["inputs"])
    target_channels = len(config["data"]["targets"])

    return SimVP_Model_no_skip_configurable(
        in_shape=(sequence_length, input_channels, int(grid_cfg["height"]), int(grid_cfg["width"])),
        out_channels=target_channels,
        model_type=model_cfg["type"],
        hid_S=int(model_cfg["hidden_spatial"]),
        hid_T=int(model_cfg["hidden_temporal"]),
        N_S=int(model_cfg["spatial_depth"]),
        N_T=int(model_cfg["temporal_depth"]),
        drop=float(model_cfg["drop"]),
        drop_path=float(model_cfg["drop_path"]),
    )


def evaluate(model, dataloader, device, use_amp, config):
    model.eval()
    total_loss = 0.0
    steps = 0
    skipped_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch_to_model_tensors(batch, config)
            if inputs is None:
                skipped_batches += 1
                continue
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                predictions = model(inputs)
                loss = torch_masked_mse(predictions, targets)
            total_loss += float(loss.item())
            steps += 1
    return total_loss / max(steps, 1), skipped_batches


def main():
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config)
    training_cfg = config["training"]
    tracker = MLflowTracker(config, stage="training")

    set_seed(int(training_cfg["seed"]))
    LOGGER.info("Loaded training config from %s", config["__config_path__"])

    weights_dir = ensure_dir(config["paths"]["weights_dir"], config)
    logs_dir = ensure_dir(config["paths"]["logs_dir"], config)
    LOGGER.info("Artifacts will be written to weights_dir=%s logs_dir=%s", weights_dir, logs_dir)
    tracker.start_run(run_name=training_cfg.get("checkpoint_name"))
    try:
        tracker.log_config()

        train_dataset = build_dataset(config, "train")
        val_dataset = build_dataset(config, "validation")
        collate_fn = get_collate_fn()

        if len(train_dataset) == 0:
            raise RuntimeError("No OceanTACO training samples matched the current configuration.")
        if len(val_dataset) == 0:
            raise RuntimeError("No OceanTACO validation samples matched the current configuration.")
        LOGGER.info("Loaded datasets: train=%s samples validation=%s samples", len(train_dataset), len(val_dataset))
        tracker.log_metrics(
            {
                "train_dataset_size": float(len(train_dataset)),
                "validation_dataset_size": float(len(val_dataset)),
            },
            step=0,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(training_cfg["batch_size"]),
            shuffle=bool(training_cfg["shuffle"]),
            num_workers=int(training_cfg["num_workers"]),
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(training_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(training_cfg["num_workers"]),
            collate_fn=collate_fn,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = bool(training_cfg["amp"]) and device.type == "cuda"
        model = build_model(config).to(device)
        LOGGER.info("Using device=%s amp=%s", device, use_amp)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

        checkpoint_name = training_cfg["checkpoint_name"]
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
        logger = LossLogger(logs_dir / f"{checkpoint_name}_losses.csv")
        start_epoch = 0

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            LOGGER.info("Resumed checkpoint from %s at epoch=%s", checkpoint_path, start_epoch)

        epochs = int(training_cfg["epochs"])
        LOGGER.info("Starting training for %s epochs", epochs - start_epoch)
        for epoch in range(start_epoch, epochs):
            model.train()
            total_train_loss = 0.0
            steps = 0
            skipped_train_batches = 0
            LOGGER.info("Epoch %s/%s started", epoch + 1, epochs)

            for batch in train_loader:
                inputs, targets = batch_to_model_tensors(batch, config)
                if inputs is None:
                    skipped_train_batches += 1
                    continue
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    predictions = model(inputs)
                    loss = torch_masked_mse(predictions, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += float(loss.item())
                steps += 1

            train_loss = total_train_loss / max(steps, 1)
            val_loss, skipped_val_batches = evaluate(model, val_loader, device, use_amp, config)
            logger.log(epoch, train_loss, val_loss)
            tracker.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "skipped_train_batches": float(skipped_train_batches),
                    "skipped_val_batches": float(skipped_val_batches),
                    "completed_train_batches": float(steps),
                },
                step=epoch,
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config_path": config["__config_path__"],
                "config_snapshot": copy.deepcopy(config),
            }
            checkpoint_path = weights_dir / f"{checkpoint_name}_epoch{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            if config.get("tracking", {}).get("mlflow", {}).get("log_checkpoints", True):
                tracker.log_artifact(checkpoint_path, artifact_subdir="checkpoints")
            LOGGER.info(
                "Epoch %s/%s finished: train_loss=%.6f val_loss=%.6f skipped_train_batches=%s skipped_val_batches=%s checkpoint=%s",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                skipped_train_batches,
                skipped_val_batches,
                checkpoint_path,
            )
    finally:
        checkpoint_name = training_cfg["checkpoint_name"]
        loss_log_path = logs_dir / f"{checkpoint_name}_losses.csv"
        if loss_log_path.exists() and config.get("tracking", {}).get("mlflow", {}).get("log_loss_csv", True):
            tracker.log_artifact(loss_log_path, artifact_subdir="logs")
            LOGGER.info("Saved loss log to %s", loss_log_path)
        tracker.end_run()
        LOGGER.info("Training run finished")


if __name__ == "__main__":
    main()
