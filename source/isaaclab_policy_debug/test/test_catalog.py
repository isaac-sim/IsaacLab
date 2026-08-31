# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import suppress
from pathlib import Path

import torch
from isaaclab_policy_debug.catalog import CheckpointCatalog, CheckpointLoader


def test_catalog_waits_for_stable_direct_children_and_tracks_live_changes(tmp_path: Path):
    checkpoint = tmp_path / "model_10.pt"
    checkpoint.write_bytes(b"partial")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "model_99.pt").write_bytes(b"ignored")

    catalog = CheckpointCatalog(tmp_path)
    assert not catalog.scan()[0].ready
    assert catalog.scan()[0].ready
    assert [entry.path.name for entry in catalog.entries] == ["model_10.pt"]

    checkpoint.write_bytes(b"still writing")
    assert not catalog.scan()[0].ready
    assert catalog.scan()[0].ready

    added = tmp_path / "model_20.pt"
    added.write_bytes(b"new")
    assert len(catalog.scan()) == 2
    assert not next(entry for entry in catalog.entries if entry.path == added).ready
    catalog.scan()
    assert next(entry for entry in catalog.entries if entry.path == added).ready

    checkpoint.unlink()
    assert [entry.path for entry in catalog.scan()] == [added]


def test_catalog_orders_loaded_iteration_then_filename_then_mtime(tmp_path: Path):
    for name in ("model_2.pt", "model_10.pt", "final.pt"):
        torch.save({"model_state_dict": {"weight": torch.zeros(1)}}, tmp_path / name)
    catalog = CheckpointCatalog(tmp_path)
    catalog.scan()
    catalog.scan()
    assert [entry.path.name for entry in catalog.entries[:2]] == ["model_10.pt", "model_2.pt"]

    final = next(entry for entry in catalog.entries if entry.path.name == "final.pt")
    final.iteration = 100
    assert catalog.entries[0] is final


def test_corrupt_checkpoint_is_retryable_after_file_changes(tmp_path: Path):
    path = tmp_path / "model_1.pt"
    path.write_bytes(b"broken")
    catalog = CheckpointCatalog(tmp_path)
    catalog.scan()
    entry = catalog.scan()[0]
    loader = CheckpointLoader()
    with suppress(ValueError):
        loader.load(entry)
    assert entry.status == "error"

    torch.save(
        {
            "actor_state_dict": {"mlp.weight": torch.zeros(4, 3)},
            "critic_state_dict": {"mlp.weight": torch.zeros(1, 3)},
            "iter": 7,
            "infos": {"production_contract": "test-contract"},
        },
        path,
    )
    catalog.scan()
    assert entry.error is None and not entry.ready
    catalog.scan()
    loaded = loader.load(entry)
    assert loaded.iteration == 7
    assert loaded.metadata == {"production_contract": "test-contract"}
    assert loaded.parameter_shapes == {
        "actor.mlp.weight": (4, 3),
        "critic.mlp.weight": (1, 3),
    }
