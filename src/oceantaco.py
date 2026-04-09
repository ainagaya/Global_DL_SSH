from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List

import pandas as pd
import torch

from src.config_utils import ensure_dir, get_split_config


DEFAULT_GLOBAL_BBOX = (-180.0, 180.0, -60.0, 60.0)
LOGGER = logging.getLogger(__name__)

DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_RETRY_BACKOFF_MULTIPLIER = 2.0
DEFAULT_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)


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


def _extract_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            return int(status_code)
    status_code = getattr(exc, "status_code", None)
    if status_code is not None:
        return int(status_code)
    return None


def _iter_exception_chain(exc: Exception):
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)


def _is_transient_data_error(exc: Exception, retry_status_codes: tuple[int, ...]) -> bool:
    for current_exc in _iter_exception_chain(exc):
        status_code = _extract_status_code(current_exc)
        if status_code is not None and status_code in retry_status_codes:
            return True

    message = " | ".join(str(current_exc).lower() for current_exc in _iter_exception_chain(exc))
    exception_type_names = {type(current_exc).__name__.lower() for current_exc in _iter_exception_chain(exc)}
    transient_markers = (
        "timed out",
        "timeout",
        "temporary failure",
        "temporary unavailable",
        "connection reset",
        "connection aborted",
        "connection refused",
        "connection broken",
        "server disconnected",
        "remote end closed connection",
        "incompleteread",
        "chunkedencodingerror",
        "protocolerror",
        "connection broken:",
        "broken pipe",
    )
    transient_exception_names = {
        "chunkedencodingerror",
        "protocolerror",
        "incompleteread",
        "readtimeout",
        "connecttimeout",
        "connectionerror",
    }
    return any(marker in message for marker in transient_markers) or bool(
        exception_type_names & transient_exception_names
    )


def _build_patched_dataset_class(base_cls, ocean_dataset_module):
    class PatchedOceanTACODataset(base_cls):
        """Compatibility wrapper around OceanTACO's dataset loader.

        OceanTACO's upstream `_load_variable()` should be preferred when it
        succeeds because it knows the intended handling for the dataset.

        We only fall back to the defensive implementation below when upstream
        crashes on malformed fragments during merging. This preserves normal
        loading behavior while still protecting training/inference from the
        `IndexError: tuple index out of range` issue we saw earlier.
        """

        def __init__(
            self,
            *args,
            retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
            retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
            retry_backoff_multiplier: float = DEFAULT_RETRY_BACKOFF_MULTIPLIER,
            retry_status_codes: tuple[int, ...] = DEFAULT_RETRY_STATUS_CODES,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.retry_attempts = max(1, int(retry_attempts))
            self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
            self.retry_backoff_multiplier = max(1.0, float(retry_backoff_multiplier))
            self.retry_status_codes = tuple(int(code) for code in retry_status_codes)

        def _sleep_before_retry(self, attempt: int) -> None:
            if self.retry_backoff_seconds <= 0:
                return
            delay = self.retry_backoff_seconds * (self.retry_backoff_multiplier ** max(0, attempt - 1))
            time.sleep(delay)

        def _load_with_retry(self, loader_name, loader, *args):
            for attempt in range(1, self.retry_attempts + 1):
                try:
                    return loader(*args)
                except Exception as exc:
                    if not _is_transient_data_error(exc, self.retry_status_codes):
                        raise
                    if attempt >= self.retry_attempts:
                        LOGGER.error(
                            "Skipping transiently unavailable OceanTACO %s after %s attempts: %s",
                            loader_name,
                            attempt,
                            exc,
                        )
                        return None
                    LOGGER.warning(
                        "Retrying OceanTACO %s after transient failure (%s/%s): %s",
                        loader_name,
                        attempt,
                        self.retry_attempts,
                        exc,
                    )
                    self._sleep_before_retry(attempt)

        def _postprocess_loaded_variable(self, var, result):
            if result is None:
                return None

            data = result.get("data")
            if data is None:
                return result

            updated = dict(result)
            updated["data"] = data
            return updated

        def _load_variable(self, var, file_df, bbox):
            try:
                result = self._load_with_retry("variable load", super()._load_variable, var, file_df, bbox)
                if result is None:
                    return None
                return self._postprocess_loaded_variable(var, result)
            except (IndexError, ValueError) as exc:
                LOGGER.warning(
                    "Falling back to defensive _load_variable for var=%s bbox=%s due to %s",
                    var,
                    bbox,
                    exc,
                )

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
                result = self._load_with_retry(
                    f"fragment load for {var}",
                    ocean_dataset_module.load_netcdf_var,
                    vsi_path,
                    nc_var,
                    bbox,
                )
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

            return self._postprocess_loaded_variable(var, {
                "data": ocean_dataset_module.torch.from_numpy(data.astype(ocean_dataset_module.np.float32)),
                "lats": ocean_dataset_module.torch.from_numpy(lats_out.astype(ocean_dataset_module.np.float32))
                if lats_out is not None
                else None,
                "lons": ocean_dataset_module.torch.from_numpy(lons_out.astype(ocean_dataset_module.np.float32))
                if lons_out is not None
                else None,
            })

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
    LOGGER.info(
        "Building %s queries with strategy=%s, date_range=%s..%s, regions=%s",
        split_name,
        split_cfg["strategy"],
        split_cfg["start_date"],
        split_cfg["end_date"],
        bboxes,
    )

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
        LOGGER.info("Saved %s queries for split=%s to %s", len(queries), split_name, query_path)

    LOGGER.info("Built %s queries for split=%s", len(queries), split_name)

    return queries


def build_dataset(config: Dict[str, Any], split_name: str):
    OceanTACODataset, _, _, _, HF_DEFAULT_URL, ocean_dataset_module = _import_oceantaco()
    data_cfg = config["data"]
    grid_cfg = data_cfg["grid"]

    taco_path = config.get("oceantaco", {}).get("taco_path")
    if not taco_path:
        taco_path = HF_DEFAULT_URL
    elif isinstance(taco_path, str) and "huggingface.co/datasets/" in taco_path and "/resolve/main/" in taco_path:
        LOGGER.warning(
            "Configured taco_path=%s looks like a raw Hugging Face resolve URL. "
            "Falling back to OceanTACO HF_DEFAULT_URL because the package expects a dataset root handle.",
            taco_path,
        )
        taco_path = HF_DEFAULT_URL
    queries = build_queries(config, split_name)
    LOGGER.info(
        "Creating OceanTACO dataset for split=%s with %s queries, %s inputs, %s targets, grid=%sx%s",
        split_name,
        len(queries),
        len(data_cfg["inputs"]),
        len(data_cfg["targets"]),
        int(grid_cfg["height"]),
        int(grid_cfg["width"]),
    )
    dataset_cls = _build_patched_dataset_class(OceanTACODataset, ocean_dataset_module)
    retry_cfg = config.get("oceantaco", {}).get("retry", {})
    dataset = dataset_cls(
        taco_path=taco_path,
        queries=queries,
        input_variables=[source_cfg["key"] for source_cfg in data_cfg["inputs"]],
        target_variables=[source_cfg["key"] for source_cfg in data_cfg["targets"]],
        target_resolution=data_cfg.get("target_resolution"),
        temporal_agg=data_cfg.get("temporal_agg", "stack"),
        default_patch_size=(int(grid_cfg["height"]), int(grid_cfg["width"])),
        retry_attempts=retry_cfg.get("attempts", DEFAULT_RETRY_ATTEMPTS),
        retry_backoff_seconds=retry_cfg.get("backoff_seconds", DEFAULT_RETRY_BACKOFF_SECONDS),
        retry_backoff_multiplier=retry_cfg.get("backoff_multiplier", DEFAULT_RETRY_BACKOFF_MULTIPLIER),
        retry_status_codes=tuple(retry_cfg.get("status_codes", DEFAULT_RETRY_STATUS_CODES)),
    )
    LOGGER.info("Dataset ready for split=%s with %s samples", split_name, len(dataset))
    return dataset


def get_collate_fn():
    def canonicalize_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        if tensor.ndim == 3:
            return tensor
        return tensor

    def pad_tensor_to_shape(tensor: torch.Tensor, target_shape: list[int]) -> torch.Tensor:
        if list(tensor.shape) == target_shape:
            return tensor

        pad = []
        for dim_index in range(tensor.ndim - 1, -1, -1):
            pad.extend([0, target_shape[dim_index] - tensor.shape[dim_index]])
        return torch.nn.functional.pad(tensor, pad, value=0.0)

    def collate_group(batch: list[dict[str, Any]], group_name: str) -> dict[str, torch.Tensor | None]:
        variable_names = []
        seen = set()
        for sample in batch:
            for variable_name in sample.get(group_name, {}).keys():
                if variable_name not in seen:
                    seen.add(variable_name)
                    variable_names.append(variable_name)

        collated = {}
        for variable_name in variable_names:
            tensors = [
                canonicalize_tensor(
                    None if sample[group_name].get(variable_name) is None else sample[group_name][variable_name].get("data")
                )
                for sample in batch
            ]
            present_tensors = [tensor for tensor in tensors if tensor is not None]
            if not present_tensors:
                collated[variable_name] = None
                continue

            reference = present_tensors[0]
            target_shape = [max(tensor.shape[dim] for tensor in present_tensors) for dim in range(reference.ndim)]
            padded_tensors = []
            for tensor in tensors:
                if tensor is None:
                    padded_tensors.append(torch.zeros(target_shape, dtype=reference.dtype))
                else:
                    padded_tensors.append(pad_tensor_to_shape(tensor, target_shape))
            collated[variable_name] = torch.stack(padded_tensors, dim=0)
        return collated

    def collate_preserving_alignment(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            return {}

        return {
            "inputs": collate_group(batch, "inputs"),
            "targets": collate_group(batch, "targets"),
            "coords": batch[0]["coords"],
            "metadata": {
                "bboxes": [sample["metadata"]["bbox"] for sample in batch],
                "time_ranges": [sample["metadata"]["time_range"] for sample in batch],
            },
        }

    return collate_preserving_alignment


def _convert_sst_to_celsius_if_needed(var: str, tensor: torch.Tensor) -> torch.Tensor:
    if var != "l4_sst":
        return tensor

    finite = tensor[torch.isfinite(tensor)]
    if finite.numel() == 0:
        return tensor

    # OceanTACO SST may arrive in Kelvin or Celsius depending on the loader path.
    # Use a conservative threshold so typical Celsius fields are left untouched.
    if float(finite.max().item()) > 200.0:
        return tensor - 273.15
    return tensor


def _preprocess_tensor(tensor: torch.Tensor, source_cfg: Dict[str, Any]) -> torch.Tensor:
    output = torch.nan_to_num(tensor.clone(), nan=0.0)
    output = _convert_sst_to_celsius_if_needed(source_cfg["key"], output)

    norm_cfg = source_cfg.get("normalize")
    if not norm_cfg:
        return output

    min_valid = norm_cfg.get("min_valid")
    if min_valid is not None:
        output[output < float(min_valid)] = 0.0
    return output


def _preprocess_tensor_for_plot(
    tensor: torch.Tensor,
    source_cfg: Dict[str, Any],
) -> torch.Tensor:
    output = tensor.clone()
    finite_mask = torch.isfinite(output)
    output = torch.where(finite_mask, output, torch.full_like(output, torch.nan))
    output = _convert_sst_to_celsius_if_needed(source_cfg["key"], output)

    norm_cfg = source_cfg.get("normalize")
    if not norm_cfg:
        return output

    min_valid = norm_cfg.get("min_valid")
    if min_valid is not None:
        output = torch.where(output < float(min_valid), torch.full_like(output, torch.nan), output)
    return output


def _normalise_tensor(tensor: torch.Tensor, source_cfg: Dict[str, Any]) -> torch.Tensor:
    output = _preprocess_tensor(tensor, source_cfg)
    norm_cfg = source_cfg.get("normalize")
    if not norm_cfg:
        return output

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
    normalise: bool = True,
    preserve_missing: bool = False,
) -> torch.Tensor:
    if tensor is None:
        fill_value = torch.nan if preserve_missing else 0.0
        return torch.full((batch_size, sequence_length, height, width), fill_value=fill_value, dtype=torch.float32)

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

    if normalise:
        return _normalise_tensor(output, source_cfg)
    if preserve_missing:
        return _preprocess_tensor_for_plot(output, source_cfg)
    return _preprocess_tensor(output, source_cfg)


def batch_input_fields(
    batch: Dict[str, Any],
    config: Dict[str, Any],
    normalise: bool = True,
    preserve_missing: bool = False,
) -> Dict[str, torch.Tensor] | None:
    grid_cfg = config["data"]["grid"]
    sequence_length = int(config["data"]["sequence_length"])
    height = int(grid_cfg["height"])
    width = int(grid_cfg["width"])

    input_map = batch["inputs"]
    first_input = next((tensor for tensor in input_map.values() if tensor is not None), None)
    first_target = next((tensor for tensor in batch["targets"].values() if tensor is not None), None)
    batch_size = None
    if batch.get("metadata", {}).get("bboxes"):
        batch_size = len(batch["metadata"]["bboxes"])
    elif first_input is not None:
        batch_size = int(first_input.shape[0])
    elif first_target is not None:
        batch_size = int(first_target.shape[0])

    if batch_size is None:
        return None

    return {
        source_cfg["key"]: _prepare_variable_tensor(
            input_map.get(source_cfg["key"]),
            batch_size,
            sequence_length,
            height,
            width,
            source_cfg,
            normalise=normalise,
            preserve_missing=preserve_missing,
        )
        for source_cfg in config["data"]["inputs"]
    }


def batch_to_model_tensors(batch: Dict[str, Any], config: Dict[str, Any], allow_empty_inputs: bool = False):
    grid_cfg = config["data"]["grid"]
    sequence_length = int(config["data"]["sequence_length"])
    height = int(grid_cfg["height"])
    width = int(grid_cfg["width"])

    input_map = batch["inputs"]
    first_input = next((tensor for tensor in input_map.values() if tensor is not None), None)
    first_target = next((tensor for tensor in batch["targets"].values() if tensor is not None), None)
    batch_size = None
    if batch.get("metadata", {}).get("bboxes"):
        batch_size = len(batch["metadata"]["bboxes"])
    elif first_input is not None:
        batch_size = int(first_input.shape[0])
    elif first_target is not None:
        batch_size = int(first_target.shape[0])

    if batch_size is None:
        return None, None
    if first_input is None and not allow_empty_inputs:
        return None, None

    prepared_inputs = batch_input_fields(batch, config, normalise=True)
    if prepared_inputs is None:
        return None, None
    input_tensors = [prepared_inputs[source_cfg["key"]] for source_cfg in config["data"]["inputs"]]
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
