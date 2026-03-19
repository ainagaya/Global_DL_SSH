#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gdlssh.config import load_config
from gdlssh.importing import import_from_string
from gdlssh.inference import InferenceStats, load_checkpoint, predict_all_regions, refactor_predictions_by_day
from gdlssh.logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict SSH fields from preprocessed regional inputs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    stats = InferenceStats(
        mean_ssh=config["normalization"]["mean_ssh"],
        std_ssh=config["normalization"]["std_ssh"],
        mean_sst=config["normalization"]["mean_sst"],
        std_sst=config["normalization"]["std_sst"],
    )
    model_factory = import_from_string(config["legacy"]["model_factory"])
    model = model_factory(**config["train"]["model_kwargs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(config["paths"]["weights_dir"]) / f"{config['predict']['experiment_name']}_weights_epoch{config['predict']['checkpoint_epoch']}"
    model = load_checkpoint(model.to(device), checkpoint, device)
    region_files = [Path(config["paths"]["testing_dir"]) / f"input_data_region{region}.npy" for region in config["predict"]["available_regions"]]
    predict_all_regions(
        model=model,
        region_files=region_files,
        output_dir=config["paths"]["preds_dir"],
        experiment_name=config["predict"]["experiment_name"],
        n_t=config["sequence"]["n_t"],
        batch_size=config["predict"]["batch_size"],
        stats=stats,
        time_index=config["predict"]["prediction_time_index"],
        use_sst=True,
        num_workers=config["runtime"]["workers"],
    )
    prediction_paths = [Path(config["paths"]["preds_dir"]) / f"{config['predict']['experiment_name']}region{region}.npy" for region in config["predict"]["available_regions"]]
    refactor_predictions_by_day(
        region_prediction_paths=prediction_paths,
        output_dir=config["paths"]["refactored_preds_dir"],
        experiment_name=config["predict"]["experiment_name"],
        start_date=config["merge"]["prediction_dates"][0],
        num_days=config["predict"]["num_days"],
    )


if __name__ == "__main__":
    main()
