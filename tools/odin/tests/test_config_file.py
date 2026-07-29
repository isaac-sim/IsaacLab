# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validation behaviour of the ``odin.yaml`` loader."""

from pathlib import Path

import pytest

from tools.odin.config_file import OdinConfigError, load_odin_config

_VALID = """
osmo_profile: default
pool: isaac-dev-l40s-04
priority: NORMAL
image:
  registry: nvcr.io/nvidian
  repository: antoiner-isaac-lab
  pull_credential: ngc_read_only
resources:
  cpu: 8
  gpu: 1
  memory: 64Gi
  storage: 64Gi
  platform: ovx-l40s
exec_timeout_s: 86400
queue_timeout_s: 172800
chunk_size: 25
results_uri: s3://isaac-odin/results
retry:
  reschedule_codes: "3001-3006"
  restart_codes: ""
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "odin.yaml"
    path.write_text(text)
    return path


def test_valid_config_loads_all_fields(tmp_path: Path) -> None:
    cfg = load_odin_config(_write(tmp_path, _VALID))

    assert cfg.osmo_profile == "default"
    assert cfg.pool == "isaac-dev-l40s-04"
    assert cfg.priority == "NORMAL"
    assert cfg.image.registry == "nvcr.io/nvidian"
    assert cfg.image.repository == "antoiner-isaac-lab"
    assert cfg.image.pull_credential == "ngc_read_only"
    assert cfg.resources.cpu == 8
    assert cfg.resources.platform == "ovx-l40s"
    assert cfg.exec_timeout_s == 86400
    assert cfg.queue_timeout_s == 172800
    assert cfg.chunk_size == 25
    assert cfg.results_uri == "s3://isaac-odin/results"
    assert cfg.retry.reschedule_codes == "3001-3006"


def test_pull_credential_is_optional(tmp_path: Path) -> None:
    text = _VALID.replace("  pull_credential: ngc_read_only\n", "")
    assert load_odin_config(_write(tmp_path, text)).image.pull_credential is None


def test_missing_required_field_names_the_key_path(tmp_path: Path) -> None:
    text = _VALID.replace("  platform: ovx-l40s\n", "")
    with pytest.raises(OdinConfigError, match=r"resources\.platform"):
        load_odin_config(_write(tmp_path, text))


def test_bad_priority_is_rejected(tmp_path: Path) -> None:
    text = _VALID.replace("priority: NORMAL", "priority: URGENT")
    with pytest.raises(OdinConfigError, match="priority"):
        load_odin_config(_write(tmp_path, text))


def test_non_integer_cpu_is_rejected(tmp_path: Path) -> None:
    text = _VALID.replace("  cpu: 8", "  cpu: many")
    with pytest.raises(OdinConfigError, match=r"resources\.cpu"):
        load_odin_config(_write(tmp_path, text))


def test_bool_is_not_accepted_as_integer(tmp_path: Path) -> None:
    # YAML would otherwise read "gpu: true" as a request for one GPU.
    text = _VALID.replace("  gpu: 1", "  gpu: true")
    with pytest.raises(OdinConfigError, match=r"resources\.gpu"):
        load_odin_config(_write(tmp_path, text))


def test_chunk_size_below_one_is_rejected(tmp_path: Path) -> None:
    # Caught at load time: chunking happens after dispatch.json is written, so
    # rejecting it there would strand an orphan dispatch.
    text = _VALID.replace("chunk_size: 25", "chunk_size: 0")
    with pytest.raises(OdinConfigError, match="chunk_size"):
        load_odin_config(_write(tmp_path, text))


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(OdinConfigError, match="mapping"):
        load_odin_config(_write(tmp_path, "- just\n- a\n- list\n"))


def test_unreadable_path_is_reported(tmp_path: Path) -> None:
    with pytest.raises(OdinConfigError, match="could not read"):
        load_odin_config(tmp_path / "absent.yaml")


def test_shipped_config_is_valid() -> None:
    shipped = Path(__file__).resolve().parents[1] / "config" / "odin.yaml"
    assert load_odin_config(shipped).pool


def test_registry_host_strips_the_namespace(tmp_path: Path) -> None:
    assert load_odin_config(_write(tmp_path, _VALID)).image.registry_host == "nvcr.io"
