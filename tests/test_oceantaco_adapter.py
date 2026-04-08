from __future__ import annotations

import sys
import types

import pytest
import torch

from src.oceantaco import (
    _build_patched_dataset_class,
    _resolve_bbox,
    _split_region_bboxes,
    batch_to_model_tensors,
    build_queries,
    prediction_records,
)


def test_resolve_bbox_supports_named_preset(base_config):
    bbox = _resolve_bbox("north_pacific_east", base_config)

    assert bbox == (90.0, 95.0, -5.0, 5.0)


def test_split_region_bboxes_supports_global(base_config):
    split_cfg = {"regions": "global"}

    bboxes = _split_region_bboxes(split_cfg, base_config)

    assert bboxes == [(-180.0, 180.0, -60.0, 60.0)]


def test_batch_to_model_tensors_pads_singleton_time_dimension(base_config):
    batch = {
        "inputs": {
            "l3_ssh": torch.ones(2, 128, 128),
            "l4_sst": torch.full((2, 128, 128), 25.0),
        },
        "targets": {
            "l3_swot": torch.ones(2, 128, 128),
        },
        "metadata": {"bboxes": [], "time_ranges": []},
    }

    inputs, targets = batch_to_model_tensors(batch, base_config)

    assert inputs.shape == (2, 5, 2, 128, 128)
    assert targets.shape == (2, 5, 1, 128, 128)
    assert torch.allclose(inputs[:, 0], inputs[:, -1])
    assert torch.allclose(targets[:, 0], targets[:, -1])


def test_batch_to_model_tensors_trims_long_time_dimension(base_config):
    batch = {
        "inputs": {
            "l3_ssh": torch.ones(1, 7, 128, 128),
            "l4_sst": torch.ones(1, 7, 128, 128),
        },
        "targets": {
            "l3_swot": torch.ones(1, 7, 128, 128),
        },
        "metadata": {"bboxes": [], "time_ranges": []},
    }

    inputs, targets = batch_to_model_tensors(batch, base_config)

    assert inputs.shape[1] == 5
    assert targets.shape[1] == 5


def test_batch_to_model_tensors_can_zero_fill_empty_inputs_for_prediction(base_config):
    batch = {
        "inputs": {
            "l3_ssh": None,
            "l4_sst": None,
        },
        "targets": {
            "l3_swot": torch.ones(1, 5, 128, 128),
        },
        "metadata": {"bboxes": [(90.0, 95.0, -5.0, 5.0)], "time_ranges": [("2023-05-01", "2023-05-05")]},
    }

    inputs, targets = batch_to_model_tensors(batch, base_config, allow_empty_inputs=True)

    assert inputs.shape == (1, 5, 2, 128, 128)
    assert torch.count_nonzero(inputs) == 0
    assert targets.shape == (1, 5, 1, 128, 128)


def test_prediction_records_uses_target_index(base_config):
    batch = {
        "metadata": {
            "bboxes": [(90.0, 95.0, -5.0, 5.0)],
            "time_ranges": [("2023-05-01", "2023-05-05")],
        }
    }

    records = prediction_records(batch, base_config)

    assert records[0]["target_date"] == "2023-05-03"
    assert records[0]["bbox"] == (90.0, 95.0, -5.0, 5.0)


def test_build_queries_uses_oceantaco_query_generator(base_config, monkeypatch):
    calls = []

    class FakePatchSize:
        def __init__(self, value, unit):
            self.value = value
            self.unit = unit

    class FakeQueryGenerator:
        def __init__(self, land_mask_path=None):
            self.land_mask_path = land_mask_path

        def generate_training_queries(self, **kwargs):
            calls.append(("train", kwargs))
            return [{"kind": "train", "bbox": kwargs["bbox_constraint"]}]

        def generate_eval_queries(self, **kwargs):
            calls.append(("eval", kwargs))
            return [{"kind": "eval", "bbox": kwargs["bbox"]}]

        @staticmethod
        def save_queries(queries, path, metadata=None):
            calls.append(("save", {"queries": queries, "path": path, "metadata": metadata}))

    fake_dataset_module = types.ModuleType("ocean_taco.dataset")
    fake_dataset_module.OceanTACODataset = object
    fake_dataset_module.collate_ocean_samples = lambda batch: batch
    fake_dataset_impl_module = types.ModuleType("ocean_taco.dataset.dataset")
    fake_dataset_module.dataset = fake_dataset_impl_module

    fake_queries_module = types.ModuleType("ocean_taco.dataset.queries")
    fake_queries_module.PatchSize = FakePatchSize
    fake_queries_module.QueryGenerator = FakeQueryGenerator

    fake_retrieve_module = types.ModuleType("ocean_taco.dataset.retrieve")
    fake_retrieve_module.HF_DEFAULT_URL = "hf://fake"

    fake_root_module = types.ModuleType("ocean_taco")

    monkeypatch.setitem(sys.modules, "ocean_taco", fake_root_module)
    monkeypatch.setitem(sys.modules, "ocean_taco.dataset", fake_dataset_module)
    monkeypatch.setitem(sys.modules, "ocean_taco.dataset.dataset", fake_dataset_impl_module)
    monkeypatch.setitem(sys.modules, "ocean_taco.dataset.queries", fake_queries_module)
    monkeypatch.setitem(sys.modules, "ocean_taco.dataset.retrieve", fake_retrieve_module)

    queries = build_queries(base_config, "train")

    assert queries == [{"kind": "train", "bbox": (90.0, 95.0, -5.0, 5.0)}]
    assert calls[0][0] == "train"
    assert calls[0][1]["time_window_days"] == 5
    assert calls[0][1]["bbox_constraint"] == (90.0, 95.0, -5.0, 5.0)


def test_patched_dataset_retries_transient_load_failures():
    class FakeHTTPError(Exception):
        def __init__(self, status_code):
            self.response = types.SimpleNamespace(status_code=status_code)
            super().__init__(f"{status_code} Server Error")

    class FakeBaseDataset:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def _load_variable(self, var, file_df, bbox):
            self.calls += 1
            if self.calls < 3:
                raise FakeHTTPError(504)
            return {"data": "ok", "lats": None, "lons": None}

    fake_dataset_module = types.SimpleNamespace(
        VAR_NAMES={},
        COL_VSI="vsi",
        POINT_SOURCES=set(),
        GridMerger=None,
        load_netcdf_var=None,
        _interpolate_to_patch=None,
        np=None,
        torch=None,
    )
    dataset_cls = _build_patched_dataset_class(FakeBaseDataset, fake_dataset_module)
    dataset = dataset_cls(retry_attempts=3, retry_backoff_seconds=0.0)

    result = dataset._load_variable("l4_sst", None, (0.0, 1.0, 2.0, 3.0))

    assert result == {"data": "ok", "lats": None, "lons": None}
    assert dataset.calls == 3
