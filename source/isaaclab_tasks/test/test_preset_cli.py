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


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train.py", add_help=False)
    parser.add_argument("--task", type=str, default=None)
    return parser


# ---------------------------------------------------------------------------
# PresetTarget: per-target metadata on the enum
# ---------------------------------------------------------------------------


def test_all_legacy_aliases_aggregates_per_target_tables():
    from isaaclab_tasks.utils.preset_target import PresetTarget

    flat = PresetTarget.all_legacy_aliases()
    assert flat["newton"] == "newton_mjwarp"
    assert flat["kamino"] == "newton_kamino"


def test_preset_target_carries_base_classes():
    """Typed targets carry the cfg base classes whose subclass instances
    should bucket to them. DOMAIN carries no base classes (it's the
    catch-all)."""
    from isaaclab.physics import PhysicsCfg
    from isaaclab.renderers.renderer_cfg import RendererCfg

    from isaaclab_tasks.utils.preset_target import PresetTarget

    assert PresetTarget.PHYSICS.base_classes == (PhysicsCfg,)
    assert PresetTarget.RENDERER.base_classes == (RendererCfg,)
    assert PresetTarget.DOMAIN.base_classes == ()


# ---------------------------------------------------------------------------
# setup_preset_cli: typed flags fold into a single presets=<csv> token
# ---------------------------------------------------------------------------


def test_no_preset_flags_returns_remainder_only(monkeypatch):
    original = ["train.py", "--task=Foo-v0", "env.sim.dt=0.001"]
    monkeypatch.setattr("sys.argv", original)
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    args, hydra_argv = setup_preset_cli(_make_parser())
    assert args.task == "Foo-v0"
    assert hydra_argv == ["env.sim.dt=0.001"]
    # setup_preset_cli must NOT mutate sys.argv -- the caller controls when to assign.
    assert sys.argv == original


def test_physics_flag_translates_to_presets_token(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "env.sim.dt=0.001"],
    )
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    _, hydra_argv = setup_preset_cli(_make_parser())
    assert hydra_argv == ["presets=newton_mjwarp", "env.sim.dt=0.001"]


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

    _, hydra_argv = setup_preset_cli(_make_parser())
    assert hydra_argv == ["presets=newton_mjwarp,newton_renderer,albedo,depth"]


def test_merges_with_existing_presets_token(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "presets=albedo"],
    )
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    _, hydra_argv = setup_preset_cli(_make_parser())
    assert hydra_argv == ["presets=newton_mjwarp,albedo"]


def test_dedupes_repeated_names(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "presets=newton_mjwarp,albedo"],
    )
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    _, hydra_argv = setup_preset_cli(_make_parser())
    assert hydra_argv == ["presets=newton_mjwarp,albedo"]


def test_equals_form_works(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--physics=newton_mjwarp"])
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    _, hydra_argv = setup_preset_cli(_make_parser())
    assert hydra_argv == ["presets=newton_mjwarp"]


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

    _, hydra_argv = setup_preset_cli(_make_parser())
    assert hydra_argv == ["presets=newton_mujoco"]
    err = capsys.readouterr().err
    assert err == ""


def test_custom_task_preset_via_typed_flag_passes_through(monkeypatch, capsys):
    """A task-local custom preset name (e.g. Dexsuite's ``cube``) is accepted via
    the typed flag with no fuss -- the registry is a hint, not a gate."""
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--presets", "cube,peg_insert_4mm,mayank_solver"])
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    _, hydra_argv = setup_preset_cli(_make_parser())
    assert hydra_argv == ["presets=cube,peg_insert_4mm,mayank_solver"]
    err = capsys.readouterr().err
    assert err == ""


# ---------------------------------------------------------------------------
# Helpers: _peek_task and _bucket_variants_by_target
# ---------------------------------------------------------------------------


def test_argv_helper_finds_task_equals_form():
    from isaaclab_tasks.utils.preset_cli import _ArgvHelper

    argv = _ArgvHelper(["train.py", "--task=Foo-v0"])
    assert argv.task_name == "Foo-v0"
    assert argv.help_requested is False


def test_argv_helper_finds_task_separated_form():
    from isaaclab_tasks.utils.preset_cli import _ArgvHelper

    argv = _ArgvHelper(["train.py", "--task", "Foo-v0"])
    assert argv.task_name == "Foo-v0"


def test_argv_helper_task_missing_returns_none():
    from isaaclab_tasks.utils.preset_cli import _ArgvHelper

    argv = _ArgvHelper(["train.py", "--physics", "newton_mjwarp"])
    assert argv.task_name is None
    assert argv.help_requested is False


def test_argv_helper_detects_help_flag():
    """``--help`` and ``-h`` both flip ``help_requested``."""
    from isaaclab_tasks.utils.preset_cli import _ArgvHelper

    assert _ArgvHelper(["train.py", "--help"]).help_requested is True
    assert _ArgvHelper(["train.py", "-h"]).help_requested is True
    assert _ArgvHelper(["train.py", "--task=Foo", "--help"]).help_requested is True
    assert _ArgvHelper(["train.py", "env.sim.dt=0.001"]).help_requested is False


def test_bucket_variants_routes_by_base_class_isinstance():
    """Variants bucket by ``isinstance`` against ``PresetTarget.base_classes``.

    PhysicsCfg subclass instances route to PHYSICS, RendererCfg subclass
    instances route to RENDERER, and everything else falls into DOMAIN.
    This also covers the wrapper-class case that motivated the design:
    a wrapper cfg that subclasses ``PhysicsCfg`` (e.g., ``NewtonCfg``) is
    routed correctly even when its inner solver class is something else.
    """
    from isaaclab.physics import PhysicsCfg
    from isaaclab.renderers.renderer_cfg import RendererCfg
    from isaaclab.utils import configclass

    from isaaclab_tasks.utils.preset_cli import _bucket_variants_by_target
    from isaaclab_tasks.utils.preset_target import PresetTarget

    @configclass
    class _PhysVariant(PhysicsCfg):
        class_type: str = "mock"

    @configclass
    class _PhysWrapper(PhysicsCfg):
        # Mirrors NewtonCfg's "wrapper holds an inner solver" shape: still
        # subclasses PhysicsCfg, so the base-class isinstance check still
        # buckets it correctly regardless of any nested member type.
        class_type: str = "mock_wrapper"
        inner: object = None

    @configclass
    class _RendVariant(RendererCfg):
        pass

    walked = {
        "physics": {
            "default": _PhysVariant(),
            "physx": _PhysVariant(),
            "newton_mjwarp": _PhysWrapper(inner=_PhysVariant()),
            "newton_kamino": _PhysWrapper(inner=_PhysVariant()),
        },
        "renderer": {
            "default": _RendVariant(),
            "newton_renderer": _RendVariant(),
        },
        "weight": {  # cfgs whose type is not a typed-target base subclass -> DOMAIN
            "default": 1.0,
            "light": 0.5,
            "heavy": 2.0,
        },
    }
    result = _bucket_variants_by_target(walked)
    # All four physics variants bucket to PHYSICS (including the wrapper-shaped ones).
    assert {"physx", "newton_mjwarp", "newton_kamino"} <= result[PresetTarget.PHYSICS]
    assert "newton_renderer" in result[PresetTarget.RENDERER]
    # Primitive-typed variants land in DOMAIN.
    assert {"light", "heavy"} <= result[PresetTarget.DOMAIN]
    # 'default' is filtered out everywhere -- it's the fallback, not a selectable name.
    for bucket in result.values():
        assert "default" not in bucket


# ---------------------------------------------------------------------------
# --help: task-aware variant listing
# ---------------------------------------------------------------------------


def test_help_without_task_says_pass_task(monkeypatch, capsys):
    """``--help`` without ``--task`` tells the user to pass ``--task=X``,
    once on the section description rather than repeated per-flag.
    """
    monkeypatch.setattr("sys.argv", ["train.py", "--help"])
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    parser = argparse.ArgumentParser(prog="train.py")  # default add_help=True
    parser.add_argument("--task", type=str, default=None)
    with pytest.raises(SystemExit):
        setup_preset_cli(parser)
    out = capsys.readouterr().out
    assert out.count("Pass `--task=X`") == 1


@pytest.mark.parametrize(
    "build_key, expected_phrases",
    [
        pytest.param(
            "empty",
            [
                "Physics preset name. No physics preset variants in this task.",
                "Renderer preset name. No renderer preset variants in this task.",
                "Comma-separated preset names. No preset variants in this task.",
            ],
            id="zero_variants_everywhere",
        ),
        pytest.param(
            "physics_only",
            [
                "Physics preset name. Available: alpha, beta.",
                "Renderer preset name. No renderer preset variants in this task.",
                "Comma-separated preset names (broadcast to every matching PresetCfg). Available: alpha, beta.",
            ],
            id="typed_populated_other_typed_empty",
        ),
        pytest.param(
            "domain_only",
            [
                "Physics preset name. No physics preset variants in this task.",
                "Renderer preset name. No renderer preset variants in this task.",
                "Comma-separated preset names (broadcast to every matching PresetCfg). Available: heavy, light.",
            ],
            id="domain_bucket_only",
        ),
        pytest.param(
            "mixed",
            [
                "Physics preset name. Available: my_phys.",
                "Renderer preset name. Available: my_rend.",
                (
                    "Comma-separated preset names (broadcast to every matching PresetCfg)."
                    " Available: heavy, light, my_phys, my_rend."
                ),
            ],
            id="all_three_buckets_populated",
        ),
    ],
)
def test_help_text_branch_strings(monkeypatch, capsys, build_key, expected_phrases):
    """Each branch of :func:`_help_text` renders the documented string for
    its variant shape. ``PhysicsCfg`` subclass instances land in
    ``--physics``; ``RendererCfg`` subclass instances land in ``--renderer``;
    primitives and task-local cfg classes fall into the ``--presets``
    DOMAIN catch-all. The parametrize id captures which branch each case
    locks; argparse line-wrapping is normalized away before substring
    assertions so wording changes are deliberate.
    """
    from isaaclab.physics import PhysicsCfg
    from isaaclab.renderers.renderer_cfg import RendererCfg
    from isaaclab.utils import configclass

    from isaaclab_tasks.utils.hydra import preset

    @configclass
    class _HelpPhysCfg(PhysicsCfg):
        class_type: str = "mock"

    @configclass
    class _HelpRendCfg(RendererCfg):
        pass

    @configclass
    class _EmptyCfg:
        pass

    @configclass
    class _PhysOnlyCfg:
        physics: object = preset(default=_HelpPhysCfg(), alpha=_HelpPhysCfg(), beta=_HelpPhysCfg())

    @configclass
    class _DomainOnlyCfg:
        weight: object = preset(default=1.0, light=0.5, heavy=2.0)

    @configclass
    class _MixedCfg:
        physics: object = preset(default=_HelpPhysCfg(), my_phys=_HelpPhysCfg())
        renderer: object = preset(default=_HelpRendCfg(), my_rend=_HelpRendCfg())
        weight: object = preset(default=1.0, light=0.5, heavy=2.0)

    builders = {
        "empty": _EmptyCfg,
        "physics_only": _PhysOnlyCfg,
        "domain_only": _DomainOnlyCfg,
        "mixed": _MixedCfg,
    }

    import isaaclab_tasks.utils.parse_cfg as parse_cfg

    monkeypatch.setattr(parse_cfg, "load_cfg_from_registry", lambda *_a, **_kw: builders[build_key]())
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Fake-v0", "--help"])
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--task", type=str, default=None)
    with pytest.raises(SystemExit):
        setup_preset_cli(parser)
    # Collapse argparse line-wrapping so substring checks survive width changes.
    flat = " ".join(capsys.readouterr().out.split())

    for phrase in expected_phrases:
        assert phrase in flat, f"Missing phrase: {phrase!r}"


# ---------------------------------------------------------------------------
# Contract: setup_preset_cli does NOT mutate sys.argv
# ---------------------------------------------------------------------------


def test_does_not_mutate_sys_argv(monkeypatch):
    """``setup_preset_cli`` must not mutate ``sys.argv`` -- mutation is the
    caller's responsibility. Locks the contract that ``rsl_rl/{train,play}.py``
    rely on so an ``--external_callback`` hook invoked after ``setup_preset_cli``
    can still read the user's original command line. If mutation happened
    here, the callback would see the folded ``presets=...`` token instead of
    ``--physics newton_mjwarp`` and fail to recognize the user's intent."""
    original = ["train.py", "--task=Foo-v0", "--physics", "newton_mjwarp", "env.sim.dt=0.001"]
    monkeypatch.setattr("sys.argv", original)
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    args, hydra_argv = setup_preset_cli(_make_parser())
    # sys.argv must remain exactly what the user typed.
    assert sys.argv == original
    # The folded form is exposed via the second return value.
    assert hydra_argv == ["presets=newton_mjwarp", "env.sim.dt=0.001"]
    # Parsed namespace still carries the typed values.
    assert args.physics == "newton_mjwarp"


def test_hydra_argv_keeps_presets_token_for_telemetry(monkeypatch):
    """Benchmarks capture ``hydra_argv`` for ``get_preset_string`` telemetry.
    Verify ``hydra_argv[0]`` is the folded ``presets=...`` token whenever any
    preset flag was given, so ``get_preset_string`` keeps reporting the
    active preset selection."""
    monkeypatch.setattr("sys.argv", ["bench.py", "--task=Foo-v0", "--physics=newton_mjwarp"])
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    _, hydra_argv = setup_preset_cli(_make_parser())
    assert hydra_argv[0] == "presets=newton_mjwarp"


# ---------------------------------------------------------------------------
# _peek_task: argparse-compatible semantics for repeated / malformed --task
# ---------------------------------------------------------------------------


def test_argv_helper_task_returns_last_value():
    """argparse's ``store`` action uses the last ``--task``; the scanner
    must match so ``--help`` shows variants for the task argparse will
    actually use."""
    from isaaclab_tasks.utils.preset_cli import _ArgvHelper

    assert _ArgvHelper(["train.py", "--task=Old", "--task=New"]).task_name == "New"
    assert _ArgvHelper(["train.py", "--task", "Old", "--task", "New"]).task_name == "New"
    assert _ArgvHelper(["train.py", "--task=Old", "--task", "New"]).task_name == "New"
