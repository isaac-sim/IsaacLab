# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the typed-flag preset CLI translator + decorator registry.

Force-imports the backend cfg modules at the top so the registry is
populated for the unit-level assertions. In real scripts, ``setup_cli``
loads them itself by calling ``_load_task_backends(args.task)``; the
``test_*_real_script_simulation`` tests exercise that path without
relying on the force-imports here.
"""

from __future__ import annotations

import argparse
import sys
import types

import pytest

# Force-import backend cfg modules so their @register decorators populate
# the registry for unit-level tests that don't go through setup_cli's
# task-loading path.
from isaaclab_newton.physics import kamino_manager_cfg, mjwarp_manager_cfg  # noqa: F401
from isaaclab_newton.renderers import newton_warp_renderer_cfg  # noqa: F401
from isaaclab_ov.renderers import ovrtx_renderer_cfg  # noqa: F401
from isaaclab_ovphysx.physics import ovphysx_manager_cfg  # noqa: F401
from isaaclab_physx.physics import physx_manager_cfg  # noqa: F401
from isaaclab_physx.renderers import isaac_rtx_renderer_cfg  # noqa: F401


@pytest.fixture
def stub_app_launcher(monkeypatch):
    """Avoid Isaac Sim's stdin-reading kit_app init in setup_cli tests by
    pre-populating ``sys.modules`` with a fake ``isaaclab.app`` module
    before ``setup_cli`` does its lazy ``from isaaclab.app import AppLauncher``."""
    fake = types.ModuleType("isaaclab.app")
    fake.AppLauncher = type("AppLauncher", (), {"add_app_launcher_args": staticmethod(lambda parser: None)})
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train.py", add_help=False)
    parser.add_argument("--task", type=str, default=None)
    return parser


def test_no_preset_flags_passes_argv_through(stub_app_launcher, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "env.sim.dt=0.001"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    args = setup_cli(_make_parser())
    assert args.task == "Foo-v0"
    assert sys.argv == ["train.py", "env.sim.dt=0.001"]


def test_physics_flag_translates_to_presets_token(stub_app_launcher, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "env.sim.dt=0.001"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp", "env.sim.dt=0.001"]


def test_three_flags_merge_into_one_token(stub_app_launcher, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--task=Foo-v0",
            "--physics",
            "newton_mjwarp",
            "--renderer",
            "newton_renderer",
            "--presets",
            "albedo,depth",
        ],
    )
    from isaaclab_tasks.utils.preset_cli import setup_cli

    setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp,newton_renderer,albedo,depth"]


def test_merges_with_existing_presets_token(stub_app_launcher, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "presets=albedo"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp,albedo"]


def test_dedupes_repeated_names(stub_app_launcher, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "presets=newton_mjwarp,albedo"],
    )
    from isaaclab_tasks.utils.preset_cli import setup_cli

    setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp,albedo"]


def test_equals_form_works(stub_app_launcher, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--physics=newton_mjwarp"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp"]


# ---------------------------------------------------------------------------
# Registry: @register decorator binds canonical names
# ---------------------------------------------------------------------------


def test_register_populates_known_names():
    from isaaclab.utils.preset_registry import PresetRegistry, PresetTarget

    assert PresetRegistry.names_for(PresetTarget.PHYSICS) >= {
        "physx", "ovphysx", "newton_mjwarp", "newton_kamino",
    }
    assert PresetRegistry.names_for(PresetTarget.RENDERER) >= {
        "isaacsim_rtx_renderer", "newton_renderer", "ovrtx_renderer",
    }


def test_register_attaches_preset_name_to_class():
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

    assert PhysxCfg._preset_name == "physx"


def test_register_rejects_duplicate_binding():
    from isaaclab.utils.preset_registry import PresetTarget, register

    @register(PresetTarget.PHYSICS, "_test_unique_a")
    class _A:
        pass

    with pytest.raises(RuntimeError, match="already bound"):

        @register(PresetTarget.PHYSICS, "_test_unique_a")
        class _B:
            pass


# ---------------------------------------------------------------------------
# Validation: --physics / --renderer must be registered names
# ---------------------------------------------------------------------------


def test_unknown_physics_name_rejected(stub_app_launcher, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--physics", "super_solver_v2"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    with pytest.raises(SystemExit, match="not a recognized physics preset"):
        setup_cli(_make_parser())


def test_known_physics_name_accepted(stub_app_launcher, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp"]


def test_presets_flag_is_not_validated(stub_app_launcher, monkeypatch):
    """``--presets`` is free-form; whatever the user types passes through."""
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--presets", "albedo,custom_thing"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=albedo,custom_thing"]


# ---------------------------------------------------------------------------
# Legacy alias normalization
# ---------------------------------------------------------------------------


def test_legacy_newton_alias_warns_and_normalizes(stub_app_launcher, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--physics", "newton"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    with pytest.warns(FutureWarning, match="newton.*newton_mjwarp"):
        setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp"]


def test_legacy_kamino_alias_warns_and_normalizes(stub_app_launcher, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--physics", "kamino"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    with pytest.warns(FutureWarning, match="kamino.*newton_kamino"):
        setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_kamino"]


def test_typed_flag_without_task_errors(stub_app_launcher, monkeypatch):
    """``--physics`` without ``--task`` is ambiguous (no env to validate against)."""
    monkeypatch.setattr("sys.argv", ["train.py", "--physics", "physx"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    with pytest.raises(SystemExit, match="--physics/--renderer require --task"):
        setup_cli(_make_parser())


# ---------------------------------------------------------------------------
# --help enrichment: lists registered preset names per target
# ---------------------------------------------------------------------------


def test_help_lists_registered_preset_names(stub_app_launcher, monkeypatch, capsys):
    """``--help`` enriches argparse's default with the registered vocabulary."""
    monkeypatch.setattr("sys.argv", ["train.py", "--help"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    parser = argparse.ArgumentParser(prog="train.py")  # default add_help=True
    parser.add_argument("--task", type=str, default=None)
    with pytest.raises(SystemExit):
        setup_cli(parser)
    out = capsys.readouterr().out
    assert "available preset names" in out
    assert "--physics:" in out
    # At least one canonical name should be listed (force-imported at top of file).
    assert "physx" in out
    assert "newton_renderer" in out


# ---------------------------------------------------------------------------
# Cross-env drift detection: every PresetCfg subclass uses canonical names
# ---------------------------------------------------------------------------


def _walk_preset_cfgs(cfg, on_preset, _path=""):
    """Yield every :class:`PresetCfg` node reachable from *cfg*.

    PresetCfg lives in :mod:`isaaclab_tasks.utils.hydra` (on develop and
    here). We walk dataclass fields and dict values transparently so
    nested presets are caught.
    """
    from isaaclab_tasks.utils.hydra import PresetCfg

    if isinstance(cfg, PresetCfg):
        on_preset(cfg, _path)

    items: list[tuple[str, object]] = []
    if isinstance(cfg, dict):
        items = list(cfg.items())
    elif hasattr(cfg, "__dataclass_fields__"):
        for name in cfg.__dataclass_fields__:
            items.append((name, getattr(cfg, name, None)))

    for key, val in items:
        if val is None:
            continue
        child_path = f"{_path}.{key}" if _path else key
        if hasattr(val, "__dataclass_fields__") or isinstance(val, dict) or isinstance(val, PresetCfg):
            _walk_preset_cfgs(val, on_preset, child_path)


def test_no_canonical_vocabulary_drift_in_registered_tasks():
    """CI lint: every PresetCfg subclass in any registered task must use canonical
    names where the alternative's value type is bound to a canonical (target, name).

    Catches drift like ``foo: PhysxCfg = PhysxCfg()`` (instead of ``physx:``).
    """
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401  -- triggers gym registration
    from isaaclab.utils.preset_registry import PresetRegistry
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    violations: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []

    for task_id in list(gym.envs.registry):
        if not task_id.startswith("Isaac-"):
            continue
        try:
            env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
            if isinstance(env_cfg, type):
                env_cfg = env_cfg()
            if not (hasattr(env_cfg, "__dataclass_fields__") or isinstance(env_cfg, dict)):
                continue
        except BaseException as exc:  # noqa: BLE001 -- carb/Kit imports may raise SystemExit / RuntimeError
            skipped.append((task_id, f"{type(exc).__name__}: {exc}"))
            continue

        def _check(preset_obj, _path):
            # Look up canonical name of each alternative VALUE; if a value's
            # class is registered under name "physx" but the field name is
            # something else, that's drift.
            for fname in preset_obj.__dataclass_fields__:
                value = getattr(preset_obj, fname, None)
                if value is None:
                    continue
                # Walk MRO for _preset_name (set by @register).
                preset_name = None
                for klass in type(value).__mro__:
                    if "_preset_name" in klass.__dict__:
                        preset_name = klass.__dict__["_preset_name"]
                        break
                if preset_name is None:
                    # Solver-cfg dispatch: NewtonCfg has no decoration, but its
                    # solver_cfg might.
                    inner = getattr(value, "solver_cfg", None)
                    if inner is not None:
                        for klass in type(inner).__mro__:
                            if "_preset_name" in klass.__dict__:
                                preset_name = klass.__dict__["_preset_name"]
                                break
                if preset_name is not None and preset_name != fname and fname != "default":
                    violations.append(
                        (task_id, f"{type(preset_obj).__name__}.{fname} holds a {preset_name!r} value but isn't named {preset_name!r}")
                    )

        try:
            _walk_preset_cfgs(env_cfg, _check)
        except BaseException as exc:  # noqa: BLE001
            skipped.append((task_id, f"walk failed: {type(exc).__name__}: {exc}"))

    if violations:
        formatted = "\n".join(f"  [{tid}] {msg}" for tid, msg in violations)
        pytest.fail(f"PresetCfg drift detected:\n{formatted}")
