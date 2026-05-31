# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the typed-preset CLI translator.

Two functions cover the translator:

* :func:`setup_preset_cli` -- register help description and parse argv.
  Returns the raw pre-fold remainder; no folding happens inside.
* :func:`fold_preset_tokens` -- fold typed selectors (``physics=``,
  ``renderer=``) and free-form ``presets=`` into a single
  ``presets=<csv>`` token consumed by Hydra's resolver.

Splitting parse from fold lets callers (notably ``rsl_rl/{train,play}.py``)
intersect the pre-fold remainder with an ``--external_callback`` return list
in matching vocabulary before folding once at the end. Tests below cover both
functions individually plus the bug-fix scenario they were split for.

Name validation, alias rewriting, and resolution all live in
:mod:`isaaclab_tasks.utils.hydra` and have their own tests in
``test_hydra.py``; this file does not re-cover them.
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
# setup_preset_cli: parse-only, returns the pre-fold remainder verbatim
# ---------------------------------------------------------------------------


def test_setup_preset_cli_returns_remainder_only(monkeypatch):
    """Without any preset tokens, the remainder is just the un-touched
    non-argparse tokens (Hydra path overrides, etc.)."""
    original = ["train.py", "--task=Foo-v0", "env.sim.dt=0.001"]
    monkeypatch.setattr("sys.argv", original)
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    args, remaining = setup_preset_cli(_make_parser())
    assert args.task == "Foo-v0"
    assert remaining == ["env.sim.dt=0.001"]
    # setup_preset_cli must NOT mutate sys.argv -- the caller controls when to assign.
    assert sys.argv == original


def test_setup_preset_cli_passes_typed_tokens_verbatim(monkeypatch):
    """``setup_preset_cli`` no longer folds; preset tokens come back in
    their original ``physics=`` / ``renderer=`` / ``presets=`` form so callers
    can intersect with callback returns in matching vocabulary before folding."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--task=Foo-v0",
            "physics=newton_mjwarp",
            "renderer=newton_renderer",
            "presets=albedo,depth",
            "env.sim.dt=0.001",
        ],
    )
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    _, remaining = setup_preset_cli(_make_parser())
    assert remaining == [
        "physics=newton_mjwarp",
        "renderer=newton_renderer",
        "presets=albedo,depth",
        "env.sim.dt=0.001",
    ]


def test_setup_preset_cli_does_not_mutate_sys_argv(monkeypatch):
    """``setup_preset_cli`` must not mutate ``sys.argv`` -- mutation is the
    caller's responsibility. Locks the contract that ``rsl_rl/{train,play}.py``
    rely on so an ``--external_callback`` hook invoked after ``setup_preset_cli``
    can still read the user's original command line and return pre-fold tokens
    that the caller intersects against the pre-fold remainder."""
    original = ["train.py", "--task=Foo-v0", "physics=newton_mjwarp", "env.sim.dt=0.001"]
    monkeypatch.setattr("sys.argv", original)
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    _, remaining = setup_preset_cli(_make_parser())
    assert sys.argv == original
    # Remainder is pre-fold (typed selector unchanged).
    assert remaining == ["physics=newton_mjwarp", "env.sim.dt=0.001"]


def test_setup_preset_cli_namespace_carries_no_preset_attributes(monkeypatch):
    """Preset tokens are never registered with argparse, so the parsed
    Namespace gains no ``physics`` / ``renderer`` / ``presets`` attribute.

    This is the bug-class-level guarantee against AppLauncher's name-based
    forwarding (``set(_SIM_APP_CFG_TYPES) & set(vars(args))``,
    ``app_launcher.py:681``): an attribute that doesn't exist can't collide.
    """
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "physics=newton_mjwarp", "renderer=newton_renderer", "presets=albedo"],
    )
    from isaaclab_tasks.utils.preset_cli import setup_preset_cli

    args, _ = setup_preset_cli(_make_parser())
    for attr in ("physics", "renderer", "presets"):
        assert not hasattr(args, attr), (
            f"setup_preset_cli wrote ``args.{attr}`` to the namespace -- AppLauncher's name-based"
            " forwarding can then push it into SimulationApp config. Drop the argparse registration"
            " for preset selectors and use Hydra-style tokens instead."
        )


def test_setup_preset_cli_does_not_leak_into_app_launcher_sim_app_intersection(monkeypatch):
    """Mirrors the literal intersection :class:`~isaaclab.app.AppLauncher`
    computes (``set(_SIM_APP_CFG_TYPES) & set(vars(args))``,
    ``app_launcher.py:681``). After ``setup_preset_cli`` runs with all three
    preset selectors, no preset name can be in that intersection -- the only
    keys present are those AppLauncher itself registered on the parser
    (``headless``, ``experience``, ...).
    """
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "physics=newton_mjwarp", "renderer=newton_renderer", "presets=albedo"],
    )
    from isaaclab.app import AppLauncher

    from isaaclab_tasks.utils.preset_cli import setup_preset_cli
    from isaaclab_tasks.utils.preset_target import PresetTarget

    args, _ = setup_preset_cli(_make_parser())
    intersection = set(AppLauncher._SIM_APP_CFG_TYPES.keys()) & set(vars(args).keys())
    leaked = {t.value for t in PresetTarget} & intersection
    assert not leaked, (
        f"setup_preset_cli leaked preset value(s) {sorted(leaked)} into the AppLauncher"
        " SimulationApp forwarding set -- they would land in SimulationApp.config and crash"
        " Kit (``None.lower()`` for ``renderer``). The hydra-style grammar keeps the namespace"
        " clean of preset attributes; this test guards against accidentally re-introducing them."
    )


# ---------------------------------------------------------------------------
# fold_preset_tokens: typed + broadcast tokens fold into one presets=<csv> token
# ---------------------------------------------------------------------------


def test_fold_returns_empty_input_unchanged():
    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    assert fold_preset_tokens([]) == []


def test_fold_no_preset_tokens_returns_input_unchanged():
    """Path-targeted overrides and unknown ``--flag``s pass through verbatim."""
    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    assert fold_preset_tokens(["env.sim.dt=0.001"]) == ["env.sim.dt=0.001"]
    assert fold_preset_tokens(["--my_flag=42", "agent.lr=3e-4"]) == ["--my_flag=42", "agent.lr=3e-4"]


def test_fold_physics_token_to_presets_token():
    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    assert fold_preset_tokens(["physics=newton_mjwarp", "env.sim.dt=0.001"]) == [
        "presets=newton_mjwarp",
        "env.sim.dt=0.001",
    ]


def test_fold_three_selectors_merge_into_one_token():
    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    assert fold_preset_tokens(
        [
            "physics=newton_mjwarp",
            "renderer=newton_renderer",
            "presets=albedo,depth",
        ]
    ) == ["presets=newton_mjwarp,newton_renderer,albedo,depth"]


def test_fold_dedupes_repeated_names():
    """A name appearing in both a typed selector and the broadcast list
    survives once in the folded token."""
    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    assert fold_preset_tokens(["physics=newton_mjwarp", "presets=newton_mjwarp,albedo"]) == [
        "presets=newton_mjwarp,albedo"
    ]


def test_fold_path_targeted_overrides_pass_through():
    """``env.sim.physics=NAME`` is a Hydra path-targeted override (dotted key)
    not a typed preset selector (bare ``physics``); it must pass through the
    fold untouched and reach the resolver in its original form."""
    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    assert fold_preset_tokens(["physics=newton_mjwarp", "env.sim.physics=newton_mjwarp", "env.lr=3e-4"]) == [
        "presets=newton_mjwarp",
        "env.sim.physics=newton_mjwarp",
        "env.lr=3e-4",
    ]


def test_fold_unknown_argparse_flag_passes_through():
    """Anything starting with ``--`` is not a preset token; the fold leaves
    callback-owned flags in place so the caller's intersection step can drop
    them via the callback's claim."""
    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    assert fold_preset_tokens(["--my_callback_flag=42", "physics=newton_mjwarp"]) == [
        "presets=newton_mjwarp",
        "--my_callback_flag=42",
    ]


def test_fold_unknown_name_passes_through_silently(capsys):
    """A name unknown to the registry is passed through verbatim with no
    warning. The resolver has the loaded task's full vocabulary and produces
    the rich error at resolve time if the name truly doesn't exist."""
    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    assert fold_preset_tokens(["physics=newton_mujoco"]) == ["presets=newton_mujoco"]
    assert capsys.readouterr().err == ""


def test_fold_custom_task_preset_via_broadcast_passes_through(capsys):
    """A task-local custom preset name (e.g. Dexsuite's ``cube``) is accepted
    via the broadcast selector with no fuss -- the registry is a hint, not a gate."""
    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    assert fold_preset_tokens(["presets=cube,peg_insert_4mm,mayank_solver"]) == [
        "presets=cube,peg_insert_4mm,mayank_solver"
    ]
    assert capsys.readouterr().err == ""


def test_fold_keeps_relative_order_of_non_preset_tokens():
    """Non-preset tokens retain their relative order; the folded
    ``presets=<csv>`` token is prepended."""
    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    assert fold_preset_tokens(["env.a=1", "physics=newton_mjwarp", "env.b=2", "env.c=3"]) == [
        "presets=newton_mjwarp",
        "env.a=1",
        "env.b=2",
        "env.c=3",
    ]


def test_fold_drops_empty_typed_value():
    """An empty typed-selector value (``physics=``) is skipped, not folded
    as an empty name."""
    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    assert fold_preset_tokens(["physics=", "env.sim.dt=0.001"]) == ["env.sim.dt=0.001"]


# ---------------------------------------------------------------------------
# Bug fix regression: intersection-then-fold preserves typed preset selections
#
# Reproduces the rsl_rl/{train,play}.py + --external_callback failure mode
# (PR #5587 review): a callback that reads the user's pre-fold sys.argv and
# returns pre-fold tokens must be intersected before folding so vocabularies
# match. Folding first would put ``presets=NAME`` on one side and
# ``physics=NAME`` on the other, dropping the preset by string mismatch.
# ---------------------------------------------------------------------------


def test_intersection_then_fold_preserves_typed_selection():
    """The bug-fix order: list_intersection on pre-fold tokens, then fold once.

    Models the rsl_rl callback path. With this order, a typed selector
    (``physics=newton_mjwarp``) appearing in both the main remainder and the
    callback's pre-fold return survives the intersection and folds correctly.
    """
    from isaaclab.utils.string import list_intersection

    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    main_remainder_pre_fold = [
        "physics=newton_mjwarp",
        "--my_callback_flag=42",  # main parser doesn't know this; callback owns it
        "env.lr=3e-4",
    ]
    # Callback reads (untouched) sys.argv, consumes its --my_callback_flag, returns the rest.
    callback_remainder_pre_fold = ["physics=newton_mjwarp", "env.lr=3e-4"]

    intersected = list_intersection(main_remainder_pre_fold, callback_remainder_pre_fold)
    folded = fold_preset_tokens(intersected)

    # Preset selection survives; callback-owned flag is correctly dropped.
    assert folded == ["presets=newton_mjwarp", "env.lr=3e-4"]


def test_fold_then_intersection_would_lose_typed_selection():
    """Document the wrong order. If the caller folded first and intersected
    second, the post-fold ``presets=newton_mjwarp`` would not match the
    callback's pre-fold ``physics=newton_mjwarp`` and the preset would be
    silently dropped. This test pins the bug shape so a future caller doesn't
    accidentally re-introduce it.
    """
    from isaaclab.utils.string import list_intersection

    from isaaclab_tasks.utils.preset_cli import fold_preset_tokens

    main_remainder_pre_fold = ["physics=newton_mjwarp", "--my_callback_flag=42", "env.lr=3e-4"]
    callback_remainder_pre_fold = ["physics=newton_mjwarp", "env.lr=3e-4"]

    # Wrong order: fold main first, then intersect against pre-fold callback.
    folded_first = fold_preset_tokens(main_remainder_pre_fold)
    intersected = list_intersection(folded_first, callback_remainder_pre_fold)

    # Preset is gone -- this is exactly the bug to avoid in rsl_rl scripts.
    assert intersected == ["env.lr=3e-4"]
    assert "presets=newton_mjwarp" not in intersected


# ---------------------------------------------------------------------------
# Helpers: _ArgvHelper and _bucket_variants_by_target
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

    argv = _ArgvHelper(["train.py", "physics=newton_mjwarp"])
    assert argv.task_name is None
    assert argv.help_requested is False


def test_argv_helper_detects_help_flag():
    """``--help`` and ``-h`` both flip ``help_requested``."""
    from isaaclab_tasks.utils.preset_cli import _ArgvHelper

    assert _ArgvHelper(["train.py", "--help"]).help_requested is True
    assert _ArgvHelper(["train.py", "-h"]).help_requested is True
    assert _ArgvHelper(["train.py", "--task=Foo", "--help"]).help_requested is True
    assert _ArgvHelper(["train.py", "env.sim.dt=0.001"]).help_requested is False


def test_argv_helper_task_returns_last_value():
    """argparse's ``store`` action uses the last ``--task``; the scanner
    must match so ``--help`` shows variants for the task argparse will
    actually use."""
    from isaaclab_tasks.utils.preset_cli import _ArgvHelper

    assert _ArgvHelper(["train.py", "--task=Old", "--task=New"]).task_name == "New"
    assert _ArgvHelper(["train.py", "--task", "Old", "--task", "New"]).task_name == "New"
    assert _ArgvHelper(["train.py", "--task=Old", "--task", "New"]).task_name == "New"


def test_bucket_variants_routes_by_base_class_isinstance():
    """Variants bucket by ``isinstance`` against ``PresetTarget.base_classes``.

    PhysicsCfg subclass instances route to PHYSICS, RendererCfg subclass
    instances route to RENDERER, and everything else falls into DOMAIN.
    """
    from isaaclab.physics import PhysicsCfg
    from isaaclab.renderers.renderer_cfg import RendererCfg
    from isaaclab.utils.configclass import configclass

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
    # All physics variants bucket to PHYSICS (including the wrapper-shaped ones).
    assert {"physx", "newton_mjwarp", "newton_kamino"} <= result[PresetTarget.PHYSICS]
    assert "newton_renderer" in result[PresetTarget.RENDERER]
    # Primitive-typed variants land in DOMAIN.
    assert {"light", "heavy"} <= result[PresetTarget.DOMAIN]
    # 'default' is filtered out everywhere -- it's the fallback, not a selectable name.
    for bucket in result.values():
        assert "default" not in bucket


# ---------------------------------------------------------------------------
# --help: section description renders the variant listing
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
                "physics=NAME (typed) selects a PhysicsCfg variant. Available: (none)",
                "renderer=NAME (typed) selects a RendererCfg variant. Available: (none)",
                "presets=NAME[,NAME,...] broadcast: applied to every matching PresetCfg. Available: (none)",
            ],
            id="zero_variants_everywhere",
        ),
        pytest.param(
            "physics_only",
            [
                "physics=NAME (typed) selects a PhysicsCfg variant. Available: - alpha - beta",
                "renderer=NAME (typed) selects a RendererCfg variant. Available: (none)",
                "presets=NAME[,NAME,...] broadcast: applied to every matching PresetCfg. Available: (none)",
            ],
            id="typed_populated_other_typed_empty",
        ),
        pytest.param(
            "domain_only",
            [
                "physics=NAME (typed) selects a PhysicsCfg variant. Available: (none)",
                "renderer=NAME (typed) selects a RendererCfg variant. Available: (none)",
                "presets=NAME[,NAME,...] broadcast: applied to every matching PresetCfg. Available: - heavy - light",
            ],
            id="domain_bucket_only",
        ),
        pytest.param(
            "mixed",
            [
                "physics=NAME (typed) selects a PhysicsCfg variant. Available: - my_phys",
                "renderer=NAME (typed) selects a RendererCfg variant. Available: - my_rend",
                "presets=NAME[,NAME,...] broadcast: applied to every matching PresetCfg. Available: - heavy - light",
            ],
            id="all_three_buckets_populated",
        ),
    ],
)
def test_help_text_branch_strings(monkeypatch, capsys, build_key, expected_phrases):
    """Each branch of the description builder renders the documented strings
    for its variant shape. Typed-bucketed names (PhysicsCfg/RendererCfg subclass
    instances) appear only under their typed section; the DOMAIN bucket
    (``presets:``) lists only variants that fell into the catch-all. The
    parametrize id captures which branch each case locks; argparse line-
    wrapping is normalized away before substring assertions so wording changes
    are deliberate.
    """
    from isaaclab.physics import PhysicsCfg
    from isaaclab.renderers.renderer_cfg import RendererCfg
    from isaaclab.utils.configclass import configclass

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
