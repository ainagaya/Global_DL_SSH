from __future__ import annotations

import download_oceantaco_local as download_script


def test_determine_download_splits_defaults_to_all_configured_splits(base_config):
    splits = download_script.determine_download_splits(base_config, requested_splits=None)

    assert splits == ["train", "validation", "test"]


def test_determine_download_splits_rejects_unknown_split(base_config):
    try:
        download_script.determine_download_splits(base_config, requested_splits=["train", "unknown"])
    except KeyError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("Expected unknown split selection to raise KeyError")


def test_extract_repo_relative_path_handles_vsicurl_resolve_urls():
    repo_path = download_script.extract_repo_relative_path(
        "/vsicurl/https://huggingface.co/datasets/nilsleh/OceanTACO/resolve/main/DATA/2025_02_18/SOUTH_ATLANTIC/l4_sst.nc"
    )

    assert repo_path == "DATA/2025_02_18/SOUTH_ATLANTIC/l4_sst.nc"


def test_extract_repo_relative_path_handles_vsisubfile_urls():
    repo_path = download_script.extract_repo_relative_path(
        "/vsisubfile/0_123,/vsicurl/https://huggingface.co/datasets/nilsleh/OceanTACO/resolve/main/DATA/2025_02_18/SOUTH_ATLANTIC/l4_sst.nc"
    )

    assert repo_path == "DATA/2025_02_18/SOUTH_ATLANTIC/l4_sst.nc"


def test_partition_repo_files_by_local_presence_skips_existing_files(tmp_path):
    existing_path = tmp_path / "DATA/2025_02_18/SOUTH_ATLANTIC/l4_sst.nc"
    existing_path.parent.mkdir(parents=True)
    existing_path.write_text("already local", encoding="utf-8")

    existing, missing = download_script.partition_repo_files_by_local_presence(
        [
            "DATA/2025_02_18/SOUTH_ATLANTIC/l4_sst.nc",
            "DATA/2025_02_18/SOUTH_ATLANTIC/l3_ssh.nc",
        ],
        tmp_path,
        force=False,
    )

    assert existing == ["DATA/2025_02_18/SOUTH_ATLANTIC/l4_sst.nc"]
    assert missing == ["DATA/2025_02_18/SOUTH_ATLANTIC/l3_ssh.nc"]


def test_partition_repo_files_by_local_presence_downloads_all_when_forced(tmp_path):
    existing_path = tmp_path / "COLLECTION.json"
    existing_path.write_text("already local", encoding="utf-8")

    existing, missing = download_script.partition_repo_files_by_local_presence(
        ["COLLECTION.json"],
        tmp_path,
        force=True,
    )

    assert existing == []
    assert missing == ["COLLECTION.json"]
