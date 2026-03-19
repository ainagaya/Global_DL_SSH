#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gdlssh.config import load_config
from gdlssh.io_utils import list_files
from gdlssh.logging_utils import configure_logging
from gdlssh.tfrecords import SequenceSchema, TfrecordTorchDataset
from gdlssh.training import build_loss, build_model, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SimVP model using configuration-driven orchestration.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    schema = SequenceSchema(
        batch_size=config["sequence"]["batch_size"],
        n_t=config["sequence"]["n_t"],
        grid_size=config["geometry"]["grid_size"],
        n_obs_max=config["sequence"]["n_obs_max"],
        domain_x_m=config["geometry"]["domain_x_m"],
        domain_y_m=config["geometry"]["domain_y_m"],
        mean_ssh=config["normalization"]["mean_ssh"],
        std_ssh=config["normalization"]["std_ssh"],
        mean_sst=config["normalization"]["mean_sst"],
        std_sst=config["normalization"]["std_sst"],
    )
    train_files = list_files(config["paths"]["training_tfrecord_dir"], ".tfrecord")
    val_files = list_files(config["paths"]["validation_tfrecord_dir"], ".tfrecord")
    train_loader = DataLoader(TfrecordTorchDataset(train_files, schema), batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(TfrecordTorchDataset(val_files, schema), batch_size=1, shuffle=False, num_workers=0)
    model = build_model(config["legacy"]["model_factory"], config["train"]["model_kwargs"])
    criterion = build_loss(config["loss"]["import_path"], config["loss"].get("kwargs", {}))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=config["train"]["num_epochs"],
        checkpoint_dir=config["paths"]["weights_dir"],
        checkpoint_prefix=config["train"]["experiment_name"],
        loss_csv_path=Path(config["paths"]["logs_dir"]) / f"{config['train']['experiment_name']}_losses.csv",
        device=device,
    )


if __name__ == "__main__":
    main()
