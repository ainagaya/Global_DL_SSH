from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

from src.config_utils import ensure_dir, get_split_config


DEFAULT_GLOBAL_BBOX = (-180.0, 180.0, -60.0, 60.0)


def _import_oceantaco():
    try:
        from ocean_taco.dataset import OceanTACODataset, collate_ocean_samples
        from ocean_taco.dataset.queries import PatchSize, QueryGenerator
        from ocean_taco.dataset.retrieve import HF_DEFAULT_URL
        from ocean_taco.dataset import dataset as ocean_dataset_module
    except ImportError as exc:
        raise ImportError(
            "ocean-taco is required for this workflow. Install it with `pip install ocean-taco`."
        ) from exc

    return OceanTACODataset, collate_ocean_samples, PatchSize, QueryGenerator, HF_DEFAULT_URL, ocean_dataset_module


def _is_mergeable_grid(data) -> bool:
    return getattr(data, "ndim", 0) >= 2 and all(dim > 0 for dim in data.shape[-2:])


def _build_patched_dataset_class(base_cls, ocean_dataset_module):
    class PatchedOceanTACODataset(base_cls):
        """Compatibility wrapper around OceanTACO's dataset loader.

        OceanTACO can occasionally return 1D fragments for sources that are usually
        gridded, which makes the upstream GridMerger crash when it assumes HxW data.
        We skip those malformed fragments and keep the valid 2D tiles.
        """

        def _load_variable(self, var, file_df, bbox):
            if var.startswith("glorys_"):
                var_df = file_df[file_df["data_source"] == "glorys"]
            else:
                var_df = file_df[file_df["data_source"] == var]

            if var_df.empty:
                return None

            nc_var = ocean_dataset_module.VAR_NAMES[var]
            resolution = float(var_df["res_deg_lat"].iloc[0])

            use_merger = len(var_df) > 1
            merger = ocean_dataset_module.GridMerger(bbox, resolution) if use_merger else None
            data_list = []
            lats_out, lons_out = None, None

            for _, row in var_df.iterrows():
                vsi_path = row[ocean_dataset_module.COL_VSI]
                result = ocean_dataset_module.load_netcdf_var(vsi_path, nc_var, bbox)
                if not result:
                    continue
                data, lats, lons = result

                if getattr(data, "size", 0) == 0:
                    continue

                if merger and _is_mergeable_grid(data):
                    try:
                        merger.add(data, lons, lats)
                    except (IndexError, ValueError):
                        continue
                elif _is_mergeable_grid(data):
                    data_list.append(data)
                    if lats_out is None:
                        lats_out, lons_out = lats, lons

            if merger and getattr(merger, "count", None) is not None and merger.count.sum() > 0:
                data, lats_out, lons_out = merger.result()
            elif data_list:
                data = self._aggregate_temporal(data_list)
            else:
                return None

            if var == "l4_sst":
                data = data - 273.15

            if var not in ocean_dataset_module.POINT_SOURCES and self.default_patch_size is not None:
                target_size = self.patch_sizes.get(var, self.default_patch_size)
                if data.shape[-2:] != target_size:
                    data = ocean_dataset_module._interpolate_to_patch(data, target_size)
                lon_min, lon_max, lat_min, lat_max = bbox
                h, w = target_size
                lats_out = ocean_dataset_module.np.linspace(lat_min, lat_max, h, dtype=ocean_dataset_module.np.float32)
                lons_out = ocean_dataset_module.np.linspace(lon_min, lon_max, w, dtype=ocean_dataset_module.np.float32)

            data = ocean_dataset_module.np.nan_to_num(data, nan=0.0)
            if data.ndim > 2 and data.shape[0] == 1:
                data = data.squeeze(0)

            return {
                "data": ocean_dataset_module.torch.from_numpy(data.astype(ocean_dataset_module.np.float32)),
                "lats": ocean_dataset_module.torch.from_numpy(lats_out.astype(ocean_dataset_module.np.float32))
                if lats_out is not None
                else None,
                "lons": ocean_dataset_module.torch.from_numpy(lons_out.astype(ocean_dataset_module.np.float32))
                if lons_out is not None
                else None,
            }

    return PatchedOceanTACODataset


def _resolve_bbox(region_spec: Any, config: Dict[str, Any]) -> tuple[float, float, float, float]:
    if region_spec in (None, "global", "GLOBAL"):
        return DEFAULT_GLOBAL_BBOX

    if isinstance(region_spec, str):
        presets = config.get("regions", {})
        if region_spec not in presets:
            raise KeyError(f"Unknown region preset '{region_spec}'.")
        region_spec = presets[region_spec]

    if isinstance(region_spec, dict):
        bbox = region_spec.get("bbox")
    else:
        bbox = region_spec

    if bbox is None or len(bbox) != 4:
        raise ValueError(f"Invalid region bbox: {region_spec}")

    return tuple(float(value) for value in bbox)


def _split_region_bboxes(split_cfg: Dict[str, Any], config: Dict[str, Any]) -> List[tuple[float, float, float, float]]:
    regions = split_cfg.get("regions", "global")
    if regions in (None, "global", "GLOBAL"):
        return [DEFAULT_GLOBAL_BBOX]
    if not isinstance(regions, list):
        regions = [regions]
    return [_resolve_bbox(region_spec, config) for region_spec in regions]


def _build_patch_size(config: Dict[str, Any], PatchSize):
    patch_cfg = config["queries"]["patch_size"]
    return PatchSize(float(patch_cfg["value"]), patch_cfg["unit"])


def build_queries(config: Dict[str, Any], split_name: str):
    _, _, PatchSize, QueryGenerator, _, _ = _import_oceantaco()
    split_cfg = get_split_config(config, split_name)
    query_cfg = config["queries"]

    generator = QueryGenerator(land_mask_path=query_cfg.get("land_mask_path"))
    patch_size = _build_patch_size(config, PatchSize)
    date_range = (str(split_cfg["start_date"]), str(split_cfg["end_date"]))
    time_window_days = int(split_cfg.get("time_window_days", config["data"]["sequence_length"]))
    bboxes = _split_region_bboxes(split_cfg, config)

    queries = []
    if split_cfg["strategy"] == "training":
        total_queries = int(split_cfg["n_queries"])
        base_seed = int(config["training"]["seed"])
        counts = [total_queries // len(bboxes)] * len(bboxes)
        for idx in range(total_queries % len(bboxes)):
            counts[idx] += 1

        for idx, (bbox, query_count) in enumerate(zip(bboxes, counts)):
            if query_count == 0:
                continue
            queries.extend(
                generator.generate_training_queries(
                    n_queries=query_count,
                    patch_size=patch_size,
                    date_range=date_range,
                    bbox_constraint=bbox,
                    time_window_days=time_window_days,
                    max_land_fraction=float(query_cfg["max_land_fraction"]),
                    seed=base_seed + idx,
                    oversample_factor=float(query_cfg["oversample_factor"]),
                    verbose=bool(query_cfg["verbose"]),
                    max_spatial_overlap=float(query_cfg["max_spatial_overlap"]),
                )
            )
    elif split_cfg["strategy"] == "evaluation":
        for bbox in bboxes:
            queries.extend(
                generator.generate_eval_queries(
                    bbox=bbox,
                    patch_size=patch_size,
                    date_range=date_range,
                    spatial_overlap=float(split_cfg.get("spatial_overlap", query_cfg["spatial_overlap"])),
                    temporal_stride_days=int(split_cfg.get("temporal_stride_days", query_cfg["temporal_stride_days"])),
                    time_window_days=time_window_days,
                    max_land_fraction=float(query_cfg["max_land_fraction"]),
                    verbose=bool(query_cfg["verbose"]),
                )
            )
    else:
        raise ValueError(f"Unsupported split strategy '{split_cfg['strategy']}'")

    queries_dir = config["paths"].get("queries_dir")
    if queries_dir:
        _, _, _, QueryGenerator, _, _ = _import_oceantaco()
        query_path = ensure_dir(queries_dir, config) / split_name
        QueryGenerator.save_queries(
            queries,
            query_path,
            metadata={
                "split": split_name,
                "strategy": split_cfg["strategy"],
                "regions": split_cfg.get("regions", "global"),
            },
        )

    return queries


def build_dataset(config: Dict[str, Any], split_name: str):
    OceanTACODataset, _, _, _, HF_DEFAULT_URL, ocean_dataset_module = _import_oceantaco()
    data_cfg = config["data"]
    grid_cfg = data_cfg["grid"]

    taco_path = config["oceantaco"].get("taco_path", HF_DEFAULT_URL)
    queries = build_queries(config, split_name)
    dataset_cls = _build_patched_dataset_class(OceanTACODataset, ocean_dataset_module)
    return dataset_cls(
        taco_path=taco_path,
        queries=queries,
        input_variables=[source_cfg["key"] for source_cfg in data_cfg["inputs"]],
        target_variables=[source_cfg["key"] for source_cfg in data_cfg["targets"]],
        target_resolution=data_cfg.get("target_resolution"),
        temporal_agg=data_cfg.get("temporal_agg", "stack"),
        default_patch_size=(int(grid_cfg["height"]), int(grid_cfg["width"])),
    )


def get_collate_fn():
    _, collate_ocean_samples, _, _, _, _ = _import_oceantaco()
    return collate_ocean_samples


def _normalise_tensor(tensor: torch.Tensor, source_cfg: Dict[str, Any]) -> torch.Tensor:
    norm_cfg = source_cfg.get("normalize")
    if not norm_cfg:
        return tensor

    output = tensor.clone()
    min_valid = norm_cfg.get("min_valid")
    if min_valid is not None:
        output[output < float(min_valid)] = 0.0

    mean = norm_cfg.get("mean")
    std = norm_cfg.get("std")
    if mean is None or std in (None, 0):
        return output

    if norm_cfg.get("mask_zeros", True):
        mask = output != 0
    else:
        mask = torch.isfinite(output)

    output[mask] = (output[mask] - float(mean)) / float(std)
    return output


def _prepare_variable_tensor(
    tensor: torch.Tensor | None,
    batch_size: int,
    sequence_length: int,
    height: int,
    width: int,
    source_cfg: Dict[str, Any],
) -> torch.Tensor:
    if tensor is None:
        return torch.zeros((batch_size, sequence_length, height, width), dtype=torch.float32)

    output = tensor.float()
    if output.ndim == 3:
        output = output.unsqueeze(1)
    elif output.ndim != 4:
        raise ValueError(f"Unsupported tensor shape for variable '{source_cfg['key']}': {tuple(output.shape)}")

    current_length = int(output.shape[1])
    if current_length < sequence_length:
        if current_length == 0:
            output = torch.zeros((batch_size, sequence_length, height, width), dtype=torch.float32)
        else:
            pad_source = output[:, -1:, :, :]
            pad = pad_source.repeat(1, sequence_length - current_length, 1, 1)
            output = torch.cat([output, pad], dim=1)
    elif current_length > sequence_length:
        output = output[:, :sequence_length, :, :]

    return _normalise_tensor(output, source_cfg)


def batch_to_model_tensors(batch: Dict[str, Any], config: Dict[str, Any]):
    grid_cfg = config["data"]["grid"]
    sequence_length = int(config["data"]["sequence_length"])
    height = int(grid_cfg["height"])
    width = int(grid_cfg["width"])

    input_map = batch["inputs"]
    first_input = next((tensor for tensor in input_map.values() if tensor is not None), None)
    if first_input is None:
        return None, None
    batch_size = int(first_input.shape[0])

    input_tensors = [
        _prepare_variable_tensor(input_map.get(source_cfg["key"]), batch_size, sequence_length, height, width, source_cfg)
        for source_cfg in config["data"]["inputs"]
    ]
    target_tensors = [
        _prepare_variable_tensor(batch["targets"].get(source_cfg["key"]), batch_size, sequence_length, height, width, source_cfg)
        for source_cfg in config["data"]["targets"]
    ]

    inputs = torch.stack(input_tensors, dim=2)
    targets = torch.stack(target_tensors, dim=2)
    return inputs, targets


def prediction_records(batch: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_index = int(config["data"]["target_index"])
    records = []

    for bbox, time_range in zip(batch["metadata"]["bboxes"], batch["metadata"]["time_ranges"]):
        start_date = pd.Timestamp(time_range[0])
        end_date = pd.Timestamp(time_range[1])
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        target_date = dates[min(target_index, len(dates) - 1)].strftime("%Y-%m-%d")
        records.append(
            {
                "bbox": tuple(float(value) for value in bbox),
                "time_range": (str(start_date.date()), str(end_date.date())),
                "target_date": target_date,
            }
        )

    return records
