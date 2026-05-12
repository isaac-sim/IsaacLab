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
# setup_preset_cli: typed flags fold into a single presets=<csv> token
# ---------------------------------------------------------------------------


def test_no_preset_flags_passes_argv_through(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "env.sim.dt=0.001"])
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    args = setup_preset_cli(_make_parser())
    assert args.task == "Foo-v0"
    assert sys.argv == ["train.py", "env.sim.dt=0.001"]


def test_physics_flag_translates_to_presets_token(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "env.sim.dt=0.001"],
    )
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    setup_preset_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp", "env.sim.dt=0.001"]


def test_three_flags_merge_into_one_token(monkeypatch):
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
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    setup_preset_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp,newton_renderer,albedo,depth"]


def test_merges_with_existing_presets_token(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "presets=albedo"],
    )
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    setup_preset_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp,albedo"]


def test_dedupes_repeated_names(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "presets=newton_mjwarp,albedo"],
    )
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    setup_preset_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp,albedo"]


def test_equals_form_works(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--physics=newton_mjwarp"])
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    setup_preset_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mjwarp"]


# ---------------------------------------------------------------------------
# Pure-passthrough behavior: no validation, no warning
# ---------------------------------------------------------------------------


def test_unknown_physics_name_passes_through_silently(monkeypatch, capsys):
    """A name not in the registry is passed through verbatim with no warning.

    At CLI parse time we can't tell a typo apart from a legitimate task-local
    preset name; the resolver has the loaded task's full vocabulary and
    produces the rich error at resolve time if the name truly doesn't exist.
    """
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "--physics", "newton_mujoco"],
    )
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    setup_preset_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_mujoco"]
    err = capsys.readouterr().err
    assert err == ""


def test_custom_task_preset_via_typed_flag_passes_through(monkeypatch, capsys):
    """A task-local custom preset name (e.g. Dexsuite's ``cube``) is accepted via
    the typed flag with no fuss -- the registry is a hint, not a gate."""
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--presets", "cube,peg_insert_4mm,mayank_solver"])
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    setup_preset_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=cube,peg_insert_4mm,mayank_solver"]
    err = capsys.readouterr().err
    assert err == ""


# ---------------------------------------------------------------------------
# Helpers: _peek_task and _bucket_variants_by_target
# ---------------------------------------------------------------------------


def test_peek_task_finds_equals_form():
    from isaaclab_tasks.utils.preset_cli import _peek_task

    assert _peek_task(["train.py", "--task=Foo-v0"]) == "Foo-v0"


def test_peek_task_finds_separated_form():
    from isaaclab_tasks.utils.preset_cli import _peek_task

    assert _peek_task(["train.py", "--task", "Foo-v0"]) == "Foo-v0"


def test_peek_task_missing_returns_none():
    from isaaclab_tasks.utils.preset_cli import _peek_task

    assert _peek_task(["train.py", "--physics", "newton_mjwarp"]) is None


def test_bucket_variants_buckets_by_registered_target():
    from isaaclab.utils.preset_registry import PresetTarget

    from isaaclab_tasks.utils.preset_cli import _bucket_variants_by_target

    walked = {
        "physics": {"default": None, "physx": None, "newton_mjwarp": None},
        "renderer": {"default": None, "newton_renderer": None},
        "weight": {"default": None, "light": None, "heavy": None},  # custom names, not registered
    }
    result = _bucket_variants_by_target(walked)
    assert {"physx", "newton_mjwarp"} <= result[PresetTarget.PHYSICS]
    assert "newton_renderer" in result[PresetTarget.RENDERER]
    # Custom names with no registry entry fall into DOMAIN.
    assert {"light", "heavy"} <= result[PresetTarget.DOMAIN]
    # 'default' is filtered out everywhere -- it's the fallback, not a selectable name.
    for bucket in result.values():
        assert "default" not in bucket


# ---------------------------------------------------------------------------
# --help: task-aware variant listing
# ---------------------------------------------------------------------------


def test_help_without_task_says_pass_task(monkeypatch, capsys):
    """``--help`` without ``--task`` tells the user to pass ``--task=X``."""
    monkeypatch.setattr("sys.argv", ["train.py", "--help"])
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    parser = argparse.ArgumentParser(prog="train.py")  # default add_help=True
    parser.add_argument("--task", type=str, default=None)
    with pytest.raises(SystemExit):
        setup_preset_cli(parser)
    out = capsys.readouterr().out
    assert "Pass `--task=X`" in out


def test_help_with_task_shows_actual_variants(monkeypatch, capsys):
    """``--task=X --help`` shows variants from X's env_cfg, bucketed by target.

    Typed flags (``--physics``) list only registered names of that target.
    The DOMAIN catch-all (``--presets``) lists every variant in the task.
    """
    from isaaclab.utils import configclass

    from isaaclab_tasks.utils.hydra import preset

    @configclass
    class _FakeCfg:
        physics: object = preset(default=None, physx="px", newton_mjwarp="nmj")
        weight: object = preset(default=1.0, light=0.5, heavy=2.0)

    import isaaclab_tasks.utils.parse_cfg as parse_cfg

    monkeypatch.setattr(parse_cfg, "load_cfg_from_registry", lambda *_a, **_kw: _FakeCfg())
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Fake-v0", "--help"])
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--task", type=str, default=None)
    with pytest.raises(SystemExit):
        setup_preset_cli(parser)
    out = capsys.readouterr().out

    # Registered physics names actually in this cfg appear in --physics help.
    assert "physx" in out
    assert "newton_mjwarp" in out
    # Custom names appear in the DOMAIN catch-all (--presets) help.
    assert "light" in out
    assert "heavy" in out


def test_help_reports_env_cfg_load_failure(monkeypatch, capsys):
    """If env_cfg load raises, ``--help`` still prints, with the error shown in help text."""
    import isaaclab_tasks.utils.parse_cfg as parse_cfg

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated import error")

    monkeypatch.setattr(parse_cfg, "load_cfg_from_registry", _boom)
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Bad-Task", "--help"])
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--task", type=str, default=None)
    with pytest.raises(SystemExit):
        setup_preset_cli(parser)
    out = capsys.readouterr().out
    assert "failed to load env_cfg" in out
    assert "Bad-Task" in out
