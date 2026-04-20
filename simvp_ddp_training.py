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
from src.oceantaco import batch_to_model_tensors, build_dataset, get_collate_fn, get_reserved_input_rules
from src.pytorch_losses import torch_masked_mse
from src.simvp_model import build_simvp_model_from_config, describe_model_config

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
    return build_simvp_model_from_config(config)


def build_dataloader(dataset, batch_size, shuffle, num_workers, collate_fn, device, training_cfg):
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "collate_fn": collate_fn,
        "pin_memory": bool(training_cfg.get("pin_memory", device.type == "cuda")),
    }

    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = bool(training_cfg.get("persistent_workers", True))
        prefetch_factor = training_cfg.get("prefetch_factor")
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(**loader_kwargs)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(int(training_cfg["seed"]))
    LOGGER.info("Loaded training config from %s", config["__config_path__"])
    reserved_rules = get_reserved_input_rules(config)
    if reserved_rules:
        LOGGER.info("Reserved input rules enabled: %s", reserved_rules)

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

        train_loader = build_dataloader(
            dataset=train_dataset,
            batch_size=training_cfg["batch_size"],
            shuffle=training_cfg["shuffle"],
            num_workers=training_cfg["num_workers"],
            collate_fn=collate_fn,
            device=device,
            training_cfg=training_cfg,
        )
        val_loader = build_dataloader(
            dataset=val_dataset,
            batch_size=training_cfg["batch_size"],
            shuffle=False,
            num_workers=training_cfg["num_workers"],
            collate_fn=collate_fn,
            device=device,
            training_cfg=training_cfg,
        )

        use_amp = bool(training_cfg["amp"]) and device.type == "cuda"
        model = build_model(config).to(device)
        model_metadata = describe_model_config(config)
        LOGGER.info("Using device=%s amp=%s model=%s", device, use_amp, model_metadata)
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
        validation_every_epochs = max(1, int(training_cfg.get("validation_every_epochs", 1)))
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
            ran_validation = ((epoch - start_epoch + 1) % validation_every_epochs == 0) or (epoch == epochs - 1)
            if ran_validation:
                val_loss, skipped_val_batches = evaluate(model, val_loader, device, use_amp, config)
            else:
                val_loss = float("nan")
                skipped_val_batches = 0
            logger.log(epoch, train_loss, val_loss)
            tracker.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss if ran_validation else 0.0,
                    "skipped_train_batches": float(skipped_train_batches),
                    "skipped_val_batches": float(skipped_val_batches),
                    "completed_train_batches": float(steps),
                    "ran_validation": float(ran_validation),
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
                "model_metadata": copy.deepcopy(model_metadata),
            }
            checkpoint_path = weights_dir / f"{checkpoint_name}_epoch{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            if config.get("tracking", {}).get("mlflow", {}).get("log_checkpoints", True):
                tracker.log_artifact(checkpoint_path, artifact_subdir="checkpoints")
            LOGGER.info(
                "Epoch %s/%s finished: train_loss=%.6f val_loss=%s skipped_train_batches=%s skipped_val_batches=%s checkpoint=%s",
                epoch + 1,
                epochs,
                train_loss,
                f"{val_loss:.6f}" if ran_validation else "skipped",
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
