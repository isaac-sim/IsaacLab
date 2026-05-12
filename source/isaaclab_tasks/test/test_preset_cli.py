# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the typed-flag preset CLI translator.

The CLI is a pure translator -- it folds typed flags into ``presets=<csv>``
and registers help text. It does no validation. Name validation, alias
rewriting, and resolution all live in :mod:`isaaclab_tasks.utils.hydra`
and have their own tests in ``test_hydra.py``; this file does not
re-cover them.
"""

from __future__ import annotations

import argparse
import sys
import types

import pytest

# Force-import backend cfg modules so their @register decorators populate the
# registry for unit-level assertions. In real scripts the same import chain
# fires when the env config is loaded (transitively imports backend cfgs).
from isaaclab_newton.physics import kamino_manager_cfg, mjwarp_manager_cfg  # noqa: F401
from isaaclab_newton.renderers import newton_warp_renderer_cfg  # noqa: F401
from isaaclab_ov.renderers import ovrtx_renderer_cfg  # noqa: F401
from isaaclab_ovphysx.physics import ovphysx_manager_cfg  # noqa: F401
from isaaclab_physx.physics import physx_manager_cfg  # noqa: F401
from isaaclab_physx.renderers import isaac_rtx_renderer_cfg  # noqa: F401


@pytest.fixture
def stub_app_launcher(monkeypatch):
    """Avoid Isaac Sim's stdin-reading kit_app init by stubbing the lazy import."""
    fake = types.ModuleType("isaaclab.app")
    fake.AppLauncher = type("AppLauncher", (), {"add_app_launcher_args": staticmethod(lambda parser: None)})
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train.py", add_help=False)
    parser.add_argument("--task", type=str, default=None)
    return parser


# ---------------------------------------------------------------------------
# Registry: @register decorator binds canonical names
# ---------------------------------------------------------------------------


def test_register_populates_known_names():
    from isaaclab.utils.preset_registry import PresetRegistry, PresetTarget

    assert PresetRegistry.names_for(PresetTarget.PHYSICS) >= {
        "physx",
        "ovphysx",
        "newton_mjwarp",
        "newton_kamino",
    }
    assert PresetRegistry.names_for(PresetTarget.RENDERER) >= {
        "isaacsim_rtx_renderer",
        "newton_renderer",
        "ovrtx_renderer",
    }


def test_register_rejects_duplicate_binding():
    from isaaclab.utils.preset_registry import PresetTarget, register

    @register(PresetTarget.PHYSICS, "_test_unique_a")
    class _A:
        pass

    with pytest.raises(RuntimeError, match="already bound"):

        @register(PresetTarget.PHYSICS, "_test_unique_a")
        class _B:
            pass


def test_all_legacy_aliases_aggregates_per_target_tables():
    from isaaclab.utils.preset_registry import PresetTarget

    flat = PresetTarget.all_legacy_aliases()
    assert flat["newton"] == "newton_mjwarp"
    assert flat["kamino"] == "newton_kamino"


# ---------------------------------------------------------------------------
# setup_cli: typed flags fold into a single presets=<csv> token
# ---------------------------------------------------------------------------


def test_no_preset_flags_passes_argv_through(stub_app_launcher, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "env.sim.dt=0.001"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    args = setup_cli(_make_parser())
    assert args.task == "Foo-v0"
    assert sys.argv == ["train.py", "env.sim.dt=0.001"]


def test_physics_flag_translates_to_presets_token(stub_app_launcher, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "env.sim.dt=0.001"],
    )
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
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "presets=albedo"],
    )
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
# Pure-passthrough behavior: no validation, no warning
# ---------------------------------------------------------------------------


def test_unknown_physics_name_passes_through_silently(stub_app_launcher, monkeypatch, capsys):
    """A name not in the registry is passed through verbatim with no warning.

    At CLI parse time we can't tell a typo apart from a legitimate task-local
    preset name; the resolver has the loaded task's full vocabulary and
    produces the rich error at resolve time if the name truly doesn't exist.
    """
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "--physics", "newton_mujoco"],
    )
    from isaaclab_tasks.utils.preset_cli import setup_cli

    setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mujoco"]
    err = capsys.readouterr().err
    assert err == ""


def test_custom_task_preset_via_typed_flag_passes_through(stub_app_launcher, monkeypatch, capsys):
    """A task-local custom preset name (e.g. Dexsuite's ``cube``) is accepted via
    the typed flag with no fuss -- the registry is a hint, not a gate."""
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--presets", "cube,peg_insert_4mm,mayank_solver"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=cube,peg_insert_4mm,mayank_solver"]
    err = capsys.readouterr().err
    assert err == ""


# ---------------------------------------------------------------------------
# --help: typed-flag help mentions registered canonicals as a hint
# ---------------------------------------------------------------------------


def test_help_lists_registered_canonicals_as_hint(stub_app_launcher, monkeypatch, capsys):
    """``--help`` lists registered canonical names in the typed-flag help text.

    The hint is purely advisory: the wording says "other names are accepted",
    so users know they can pass custom task-local presets too.
    """
    monkeypatch.setattr("sys.argv", ["train.py", "--help"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    parser = argparse.ArgumentParser(prog="train.py")  # default add_help=True
    parser.add_argument("--task", type=str, default=None)
    with pytest.raises(SystemExit):
        setup_cli(parser)
    out = capsys.readouterr().out
    # Registered canonicals appear in the help output.
    assert "physx" in out
    assert "newton_mjwarp" in out
    assert "newton_renderer" in out
    # The wording makes clear other names are accepted too.
    assert "Other names are accepted" in out
