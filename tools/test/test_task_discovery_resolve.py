# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task discovery against real configs.

Kept apart from ``test_task_discovery.py`` so that file stays importable with nothing
but pytest. Everything here needs Isaac Lab and resolves real task configs, which costs
a few seconds of warm-up and roughly 0.1s per combination.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _bootstrap_paths() -> None:
    """Prepend ``tools/`` and the editable ``source/*`` packages."""
    repo_root = Path(__file__).resolve().parents[2]
    prepend = [str(repo_root / "tools")]
    for package_dir in sorted((repo_root / "source").iterdir()):
        if (package_dir / package_dir.name).is_dir():
            prepend.append(str(package_dir))
    for path in reversed(prepend):
        if path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_paths()

pytest.importorskip("isaaclab_tasks", reason="task discovery resolution needs Isaac Lab")

import task_discovery  # noqa: E402
from task_discovery import DiscoveryError, _mode_resolves  # noqa: E402


def _raise(exc: BaseException):
    """Return a ``resolve_task_config`` stand-in that raises *exc*."""

    def fake(*args, **kwargs):
        raise exc

    return fake


def test_a_kit_backed_physics_and_a_kitless_renderer_are_rejected() -> None:
    """The pairing that justifies resolving at all must actually come back rejected.

    OVRTX is kitless and cannot share a process with Isaac Sim PhysX. If the validator
    ever starts accepting this, every other test here still passes.
    """
    assert _mode_resolves("Isaac-Cartpole-Camera", "isaacsim_physx", "ovrtx", None) is None
    assert _mode_resolves("Isaac-Cartpole-Camera", "ovphysx", "isaacsim_rtx", None) is None
    assert _mode_resolves("Isaac-Cartpole-Camera", "isaacsim_physx", "isaacsim_rtx", None) is not None
    assert _mode_resolves("Isaac-Cartpole-Camera", "ovphysx", "ovrtx", None) is not None


def test_distinct_backends_get_distinct_fingerprints() -> None:
    """Guards the collapse against a config serialization that stops discriminating.

    ``to_dict`` erases class identity, so backends differ only by the values it keeps.
    Should that discriminator ever be dropped upstream, two backends would silently
    merge into one mode and a dispatcher would stop scheduling one of them.
    """
    resolutions = [_mode_resolves("Isaac-Cartpole", name, None, None) for name in ("newton_mjwarp", "newton_kamino")]
    assert all(r is not None for r in resolutions)
    fingerprints = {r[0] for r in resolutions}
    backends = {r[1] for r in resolutions}
    assert len(backends) == 2, backends
    assert len(fingerprints) == len(backends)


def test_an_alias_collapses_onto_the_backend_it_resolves_to() -> None:
    """``physics=physx`` is an alias, so it must share a fingerprint with its target."""
    alias = _mode_resolves("Isaac-Cartpole", "physx", None, None)
    concrete = _mode_resolves("Isaac-Cartpole", "ovphysx", None, None)

    assert alias is not None and concrete is not None
    assert alias == concrete


def test_the_same_combination_fingerprints_the_same_way_twice() -> None:
    """An unstable fingerprint would split one run across several modes, silently."""
    assert _mode_resolves("Isaac-Cartpole", None, None, None) == _mode_resolves("Isaac-Cartpole", None, None, None)


@pytest.mark.parametrize(
    ("raised", "expectation"),
    [
        # A config needing an uninstalled extra cannot run here -- same answer as a
        # rejection, which is what keeps discovery usable from a partial install.
        (ModuleNotFoundError("No module named 'isaaclab_absent_extra'"), "rejected"),
        # A module that imports but no longer exports what it should is API drift.
        (ImportError("cannot import name 'gone'"), "raises"),
        (AttributeError("'NoneType' object has no attribute 'solver_cfg'"), "raises"),
        # Anything else means the combination cannot run.
        (ValueError("Invalid backend combination"), "rejected"),
    ],
)
def test_failures_are_sorted_into_rejection_or_api_drift(monkeypatch, raised, expectation) -> None:
    import isaaclab_tasks.utils as utils

    monkeypatch.setattr(utils, "resolve_task_config", _raise(raised))

    if expectation == "raises":
        with pytest.raises(DiscoveryError):
            _mode_resolves("Isaac-Cartpole", None, None, None)
    else:
        assert _mode_resolves("Isaac-Cartpole", None, None, None) is None


def test_sys_argv_is_restored_even_when_resolution_fails(monkeypatch) -> None:
    """``discover_tasks`` runs in-process from a CLI tool, so a leak would corrupt it."""
    import isaaclab_tasks.utils as utils

    monkeypatch.setattr(utils, "resolve_task_config", _raise(ValueError("nope")))
    before = list(sys.argv)

    assert _mode_resolves("Isaac-Cartpole", "ovphysx", None, "rgb") is None

    assert sys.argv == before


def test_an_unloadable_config_reports_unknown_and_no_modes(monkeypatch) -> None:
    """``declared is None`` must not sit beside a mode claiming the task runs."""
    import gymnasium as gym

    import isaaclab_tasks.utils.preset_cli as preset_cli

    monkeypatch.setattr(preset_cli, "enumerate_task_presets", lambda task_name: None)
    spec = next(spec for spec in gym.registry.values() if spec.id == "Isaac-Cartpole")

    task = task_discovery.discover_tasks([spec], resolve=False)[0]

    assert task.declared is None
    assert task.modes == ()
    assert task.default is None
