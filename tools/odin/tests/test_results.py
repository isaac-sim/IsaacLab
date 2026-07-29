# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The publish/fetch seam and bundle validation."""

import json
from pathlib import Path

from tools.odin.results import fetch_results, publish_command, results_uri_for, validate_bundle


def _bundle(directory: Path, *, schema_version: str = "1.2", task: str = "Isaac-Ant") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    # Upstream's SchemaBundleFile writes "<output_prefix>_<timestamp>.json" into
    # --output_path, where output_prefix is "benchmark_<workflow>_<task>".
    path = directory / f"benchmark_training_{task}_2026-07-28_12-00-00.json"
    path.write_text(json.dumps({"schema_version": schema_version, "run": {"task": task}}))
    return path


def test_uri_is_namespaced_by_dispatch_and_row() -> None:
    uri = results_uri_for("s3://isaac-odin/results", "20260728-120000", "rsl_rl_physx_Ant_seed42")
    assert uri == "s3://isaac-odin/results/20260728-120000/rsl_rl_physx_Ant_seed42"


def test_trailing_slash_in_base_uri_does_not_double_up() -> None:
    uri = results_uri_for("s3://isaac-odin/results/", "20260728-120000", "row")
    assert "//20260728" not in uri.removeprefix("s3://")


def test_publish_command_uploads_the_output_directory() -> None:
    cmd = publish_command(base_uri="s3://b/results", dispatch_id="20260728-120000", row_key="row")
    assert "osmo data upload" in cmd
    assert "s3://b/results/20260728-120000/row" in cmd
    assert '"$OUT"' in cmd


def test_publish_command_does_not_abort_the_task_on_upload_failure() -> None:
    # The benchmark exit code is the task's verdict; a failed upload must be
    # visible in the logs without masking it.
    cmd = publish_command(base_uri="s3://b/results", dispatch_id="d", row_key="row")
    assert "||" in cmd


def test_valid_bundle_passes(tmp_path: Path) -> None:
    _bundle(tmp_path / "row")
    assert validate_bundle(tmp_path / "row") is True


def test_missing_directory_fails(tmp_path: Path) -> None:
    assert validate_bundle(tmp_path / "absent") is False


def test_directory_without_a_bundle_fails(tmp_path: Path) -> None:
    (tmp_path / "row").mkdir()
    (tmp_path / "row" / "training.stdout.log").write_text("noise")
    assert validate_bundle(tmp_path / "row") is False


def test_malformed_json_fails(tmp_path: Path) -> None:
    (tmp_path / "row").mkdir()
    (tmp_path / "row" / "benchmark_training_x_2026-07-28_12-00-00.json").write_text("{not json")
    assert validate_bundle(tmp_path / "row") is False


def test_bundle_without_schema_version_fails(tmp_path: Path) -> None:
    (tmp_path / "row").mkdir()
    (tmp_path / "row" / "benchmark_training_x_2026-07-28_12-00-00.json").write_text('{"run": {}}')
    assert validate_bundle(tmp_path / "row") is False


def test_one_good_bundle_among_broken_ones_passes(tmp_path: Path) -> None:
    row = tmp_path / "row"
    row.mkdir()
    (row / "benchmark_training_a_2026-07-28_11-00-00.json").write_text("{broken")
    _bundle(row)
    assert validate_bundle(row) is True


def test_fetch_downloads_into_the_row_directory(tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    class _Client:
        def data_download(self, remote_uri: str, dest_dir: Path) -> None:
            calls.append((remote_uri, dest_dir))
            _bundle(dest_dir)

    dest = fetch_results(
        client=_Client(),
        base_uri="s3://b/results",
        dispatch_id="20260728-120000",
        row_key="row",
        dest_dir=tmp_path,
    )

    assert dest == tmp_path / "row"
    assert calls[0][0] == "s3://b/results/20260728-120000/row"
    assert validate_bundle(dest) is True


def test_fetch_is_idempotent_when_a_valid_bundle_exists(tmp_path: Path) -> None:
    _bundle(tmp_path / "row")

    class _Client:
        def data_download(self, remote_uri: str, dest_dir: Path) -> None:
            raise AssertionError("should not re-download a valid bundle")

    assert fetch_results(client=_Client(), base_uri="s3://b", dispatch_id="d", row_key="row", dest_dir=tmp_path) == (
        tmp_path / "row"
    )


def test_fetch_returns_the_row_dir_even_when_the_bundle_is_bad(tmp_path: Path) -> None:
    # The caller reclassifies this as malformed_bundle; fetch itself must not
    # raise, or one bad row would abort the whole dispatch.
    class _Client:
        def data_download(self, remote_uri: str, dest_dir: Path) -> None:
            dest_dir.mkdir(parents=True, exist_ok=True)

    dest = fetch_results(client=_Client(), base_uri="s3://b", dispatch_id="d", row_key="row", dest_dir=tmp_path)
    assert dest == tmp_path / "row"
    assert validate_bundle(dest) is False
