from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .fs import ensure_dir
from .normalization import normalize_nonzero_np

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceStats:
    mean_ssh: float
    std_ssh: float
    mean_sst: float
    std_sst: float


class RegionSequenceDataset(Dataset):
    def __init__(self, path: str | Path, n_t: int, stats: InferenceStats, use_sst: bool = True) -> None:
        self.path = Path(path).expanduser()
        self.data = np.load(self.path, mmap_mode="r")
        self.n_t = n_t
        self.stats = stats
        self.use_sst = use_sst

    def __len__(self) -> int:
        return max(0, self.data.shape[0] - self.n_t)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.data[index : index + self.n_t].copy()
        ssh = normalize_nonzero_np(window[:, :, :, 1], self.stats.mean_ssh, self.stats.std_ssh)
        if self.use_sst:
            sst = normalize_nonzero_np(window[:, :, :, 0], self.stats.mean_sst, self.stats.std_sst, invalid_below=273.0)
            features = np.stack((ssh, sst), axis=1)
        else:
            features = np.expand_dims(ssh, axis=1)
        dummy_target = np.zeros((400, 3), dtype=np.float32)
        return torch.from_numpy(features.astype(np.float32)), torch.from_numpy(dummy_target)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if state_dict and next(iter(state_dict.keys())).startswith("module."):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_all_regions(model: torch.nn.Module, region_files: list[str | Path], output_dir: str | Path, experiment_name: str, n_t: int, batch_size: int, stats: InferenceStats, time_index: int, use_sst: bool = True, num_workers: int = 0) -> None:
    output_root = ensure_dir(output_dir)
    device = next(model.parameters()).device
    for region_file in region_files:
        region_path = Path(region_file).expanduser()
        region_name = region_path.stem.replace("input_data_", "")
        dataset = RegionSequenceDataset(region_path, n_t=n_t, stats=stats, use_sst=use_sst)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        predictions = np.zeros((len(dataset), 128, 128), dtype=np.float32)
        offset = 0
        with torch.no_grad():
            for inputs, _ in loader:
                outputs = model(inputs.to(device)).detach().cpu().numpy()
                batch_predictions = outputs[:, time_index, 0, :, :] * stats.std_ssh + stats.mean_ssh
                predictions[offset : offset + batch_predictions.shape[0]] = batch_predictions
                offset += batch_predictions.shape[0]
        save_path = output_root / f"{experiment_name}{region_name}.npy"
        np.save(save_path, predictions)
        LOGGER.info("Saved %s", save_path)


def refactor_predictions_by_day(region_prediction_paths: list[str | Path], output_dir: str | Path, experiment_name: str, start_date: str, num_days: int) -> None:
    output_root = ensure_dir(output_dir)
    region_arrays = [np.load(Path(path).expanduser(), mmap_mode="r") for path in region_prediction_paths]
    start = np.datetime64(start_date)
    for day in range(num_days):
        stacked = np.stack([array[day].copy() for array in region_arrays], axis=0)
        date_str = str(start + np.timedelta64(day, "D"))
        save_path = output_root / f"{experiment_name}_pred_{date_str}.npy"
        np.save(save_path, stacked)
        LOGGER.info("Saved %s", save_path)
