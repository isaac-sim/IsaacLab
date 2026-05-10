# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the typed-flag preset CLI translator + decorator registry.

Force-imports the backend cfg modules at the top so the registry is
populated for the unit-level assertions. In real scripts, ``setup_cli``
loads them itself by calling ``_load_task_env_cfg(args.task)``; the
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


def test_register_subclass_gets_own_canonical_name():
    """A decorated subclass gets its OWN ``_preset_name``: subclass
    ``__dict__`` doesn't yet contain the attribute (the parent's value is
    only reachable via MRO), so the per-class stamp succeeds. Decorating a
    subclass with a different name is a deliberate "new preset" declaration
    and shadows the parent's canonical at the subclass level.
    """
    from isaaclab.utils.preset_registry import PresetRegistry, PresetTarget, register

    @register(PresetTarget.PHYSICS, "_test_sub_parent")
    class _Parent:
        pass

    @register(PresetTarget.PHYSICS, "_test_sub_child")
    class _Child(_Parent):
        pass

    assert PresetRegistry.names_for(PresetTarget.PHYSICS) >= {"_test_sub_parent", "_test_sub_child"}
    assert _Parent._preset_name == "_test_sub_parent"
    assert _Child._preset_name == "_test_sub_child"


def test_register_first_wins_on_chained_decoration():
    """Re-decorating the SAME class with a different name keeps the FIRST
    binding's canonical attributes. Decorators apply bottom-up, so the inner
    ``@register`` runs first and stamps ``_preset_name``. The outer
    ``@register`` then sees ``__dict__`` already populated and skips the
    stamp; its name is still added to the registry so ``names_for`` returns
    both, but the class-level attribute reflects the inner (first) name.
    """
    from isaaclab.utils.preset_registry import PresetRegistry, PresetTarget, register

    @register(PresetTarget.PHYSICS, "_test_chain_outer")
    @register(PresetTarget.PHYSICS, "_test_chain_inner")
    class _Chained:
        pass

    assert PresetRegistry.names_for(PresetTarget.PHYSICS) >= {"_test_chain_inner", "_test_chain_outer"}
    # Inner decorator ran first and stamped; outer skipped.
    assert _Chained._preset_name == "_test_chain_inner"
    assert _Chained._preset_target is PresetTarget.PHYSICS


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
# Task variants: a field name on the selected task's PresetCfg is accepted
# even if it isn't @register'd as a canonical name. Lets users define
# alternative configurations of the same backend without re-decorating.
# ---------------------------------------------------------------------------


def _patch_load_with(monkeypatch, env_cfg_value):
    """Make ``load_cfg_from_registry`` return *env_cfg_value* for any task name."""
    monkeypatch.setattr(
        "isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry",
        lambda *args, **kwargs: env_cfg_value,
    )


def test_variant_field_name_accepted(stub_app_launcher, monkeypatch):
    """A field name in the task's PresetCfg is accepted as a variant.

    The PresetCfg has both a canonical-named field (``newton_renderer``)
    and a variant (``newton_renderer_strict``). Selecting the variant
    via ``--renderer newton_renderer_strict`` should pass validation
    and end up in the ``presets=`` token.
    """
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    from isaaclab_tasks.utils.hydra import preset
    from isaaclab_tasks.utils.preset_cli import setup_cli

    renderer_preset = preset(
        default=NewtonWarpRendererCfg(),
        newton_renderer=NewtonWarpRendererCfg(),
        newton_renderer_strict=NewtonWarpRendererCfg(enable_shadows=True),
    )
    _patch_load_with(monkeypatch, {"renderer": renderer_preset})
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--renderer", "newton_renderer_strict"])

    setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_renderer_strict"]


def test_unknown_variant_rejected(stub_app_launcher, monkeypatch):
    """A name that is neither registered nor a task variant is rejected, with
    the error message listing what *was* available (variants + registered)."""
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    from isaaclab_tasks.utils.hydra import preset
    from isaaclab_tasks.utils.preset_cli import setup_cli

    renderer_preset = preset(
        default=NewtonWarpRendererCfg(),
        newton_renderer=NewtonWarpRendererCfg(),
        newton_renderer_strict=NewtonWarpRendererCfg(),
    )
    _patch_load_with(monkeypatch, {"renderer": renderer_preset})
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--renderer", "gobbledygook"])

    with pytest.raises(SystemExit, match="not a recognized renderer preset"):
        setup_cli(_make_parser())


def test_variant_picks_up_class_level_override(stub_app_launcher, monkeypatch):
    """The Hydra resolver reads class-level field values over instance values
    so robot-specific cfg modules can reassign fields after instantiation.
    The variant walker must agree -- otherwise the typed-flag layer would
    reject names the resolver would have applied.
    """
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    from isaaclab_tasks.utils.hydra import preset
    from isaaclab_tasks.utils.preset_cli import setup_cli

    renderer_preset = preset(
        default=NewtonWarpRendererCfg(),
        newton_renderer=NewtonWarpRendererCfg(),
    )
    # Reassign at class level after the instance was built (the IsaacLab
    # robot-cfg pattern). The walker must see this and treat
    # 'newton_renderer_strict' as a valid variant.
    type(renderer_preset).newton_renderer_strict = NewtonWarpRendererCfg(enable_shadows=True)
    _patch_load_with(monkeypatch, {"renderer": renderer_preset})
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--renderer", "newton_renderer_strict"])

    setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton_renderer_strict"]


def test_variant_shadowing_legacy_alias_preserved(stub_app_launcher, monkeypatch):
    """A task-local field named after a deprecated alias (``newton``) must
    NOT be rewritten to the alias's canonical (``newton_mjwarp``); the
    user wrote the field deliberately and the variant should win.
    """
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    from isaaclab_tasks.utils.hydra import preset
    from isaaclab_tasks.utils.preset_cli import setup_cli

    # 'newton' is deprecated for --physics (rewrites to 'newton_mjwarp').
    # Here we use --renderer instead so the alias semantics on PHYSICS don't
    # interfere; the parallel renderer canonical anchor is 'newton_renderer'.
    # Build a PresetCfg whose anchor is 'newton_renderer' and whose variant
    # is named exactly 'newton' (which would also be a PHYSICS alias). The
    # CLI sees --renderer newton, which must pass through unchanged.
    renderer_preset = preset(
        default=NewtonWarpRendererCfg(),
        newton_renderer=NewtonWarpRendererCfg(),
        newton=NewtonWarpRendererCfg(enable_shadows=True),  # variant named after alias
    )
    _patch_load_with(monkeypatch, {"renderer": renderer_preset})
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--renderer", "newton"])

    # No FutureWarning should fire and the value must pass through as-is.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)  # any FutureWarning would fail the test
        setup_cli(_make_parser())
    assert sys.argv == ["train.py", "presets=newton"]


def test_variant_without_canonical_anchor_rejected(stub_app_launcher, monkeypatch):
    """Strict-anchor rule: a PresetCfg with only variants (no canonical-named
    sibling) does NOT make those variants selectable via ``--renderer``.

    The user wrote ``my_variant_a: NewtonWarpRendererCfg`` and
    ``my_variant_b: NewtonWarpRendererCfg`` without ever using the canonical
    field name ``newton_renderer``. Treating ``my_variant_a`` as valid would
    legitimize a drifted PresetCfg at the CLI surface; the cross-env drift
    lint already flags the same task on a separate run, so the two
    messages would diverge. The CLI mirrors the lint and rejects.
    """
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    from isaaclab_tasks.utils.hydra import preset
    from isaaclab_tasks.utils.preset_cli import setup_cli

    renderer_preset = preset(
        default=NewtonWarpRendererCfg(),
        my_variant_a=NewtonWarpRendererCfg(),
        my_variant_b=NewtonWarpRendererCfg(enable_shadows=True),
    )
    _patch_load_with(monkeypatch, {"renderer": renderer_preset})
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Foo-v0", "--renderer", "my_variant_a"])

    with pytest.raises(SystemExit, match="not a recognized renderer preset"):
        setup_cli(_make_parser())


# ---------------------------------------------------------------------------
# --help enrichment: argparse help= strings list valid preset names
# ---------------------------------------------------------------------------


def test_help_lists_registered_preset_names(stub_app_launcher, monkeypatch, capsys):
    """``--help`` shows registered names per target via argparse ``help=`` strings.

    No custom HelpAction; the names are baked into ``add_argument(..., help=...)``
    so ``--help`` emits standard argparse output with the listing inline.
    """
    monkeypatch.setattr("sys.argv", ["train.py", "--help"])
    from isaaclab_tasks.utils.preset_cli import setup_cli

    parser = argparse.ArgumentParser(prog="train.py")  # default add_help=True
    parser.add_argument("--task", type=str, default=None)
    with pytest.raises(SystemExit):
        setup_cli(parser)
    out = capsys.readouterr().out
    # Names must be visible in the help output (force-imported at top of file).
    assert "physx" in out
    assert "newton_renderer" in out


# ---------------------------------------------------------------------------
# Cross-env drift detection: every PresetCfg subclass uses canonical names
# ---------------------------------------------------------------------------


def _walk_preset_cfgs(cfg, on_preset, _path=""):
    """Yield every :class:`PresetCfg` node reachable from *cfg*."""
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
        if hasattr(val, "__dataclass_fields__") or isinstance(val, (dict, PresetCfg)):
            _walk_preset_cfgs(val, on_preset, child_path)


def _canonical_for(value: object) -> str | None:
    """Walk *value*'s class MRO for ``_preset_name`` (set by :func:`register`).

    Falls back to ``value.solver_cfg``'s MRO when *value* itself isn't decorated
    but holds a registered solver-cfg (e.g., ``NewtonCfg`` wraps ``MJWarpSolverCfg``).
    Returns ``None`` if neither is decorated.
    """
    for klass in type(value).__mro__:
        if "_preset_name" in klass.__dict__:
            return klass.__dict__["_preset_name"]
    inner = getattr(value, "solver_cfg", None)
    if inner is not None:
        for klass in type(inner).__mro__:
            if "_preset_name" in klass.__dict__:
                return klass.__dict__["_preset_name"]
    return None


def _drift_violations(preset_obj) -> list[str]:
    """Drift checks for one :class:`PresetCfg` instance.

    Two rules:

    1. **Name/type match.** A field named after a registered canonical
       (e.g. ``physx``) must hold a value of the class registered under
       that name. ``physx: OvrtxRendererCfg`` is drift because ``physx``
       is reserved for the ``PhysxCfg``-class anchor.
    2. **Anchor present.** Fields holding values of the same registered
       class are grouped by that canonical name; each group must include
       at least one field named after the canonical. Other fields in the
       group are accepted as variants.

    Returns a list of human-readable violation messages (empty on pass).
    Field ``"default"`` is excluded because it holds the active selection,
    not an alternative.
    """
    from isaaclab.utils.preset_registry import PresetRegistry, PresetTarget

    # All canonical names across all targets. Used to detect when a field
    # name claims a canonical identity its value class doesn't earn.
    all_registered_names = {name for target in PresetTarget for name in PresetRegistry.names_for(target)}

    by_canonical: dict[str, list[str]] = {}
    violations: list[str] = []
    cls_name = type(preset_obj).__name__

    for fname in preset_obj.__dataclass_fields__:
        if fname == "default":
            continue
        value = getattr(preset_obj, fname, None)
        if value is None:
            continue
        canonical = _canonical_for(value)

        # Rule 1: when a field name is itself a registered canonical AND its
        # value's class is registered under a DIFFERENT canonical, both sides
        # claim canonical identities that disagree -- drift. Only fires when
        # both are registered: the IsaacLab PresetCfg pattern legitimately
        # uses canonical names ('physx', 'newton_mjwarp') as preset keys
        # dispatching arbitrary unrelated types (ArticulationCfg, etc.) per
        # active backend, where the value isn't itself a registered preset.
        if fname in all_registered_names and canonical is not None and canonical != fname:
            violations.append(
                f"{cls_name}.{fname} holds {type(value).__name__} (canonical={canonical!r}) "
                f"but field name {fname!r} is reserved for the class registered under that name; "
                f"name and value class must match."
            )
            continue

        if canonical is None:
            continue
        by_canonical.setdefault(canonical, []).append(fname)

    # Rule 2: each group needs an anchor.
    for canonical, fnames in by_canonical.items():
        if canonical not in fnames:
            violations.append(
                f"{cls_name} has alternative(s) {fnames!r} of canonical {canonical!r} "
                f"but no field named {canonical!r} (variants need at least one canonical anchor)"
            )
    return violations


def test_drift_lint_rejects_only_variants():
    """Unit test for the loose drift logic: a PresetCfg with only variants
    (no canonical-named field) is flagged."""
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    from isaaclab_tasks.utils.hydra import preset

    bad_preset = preset(
        default=NewtonWarpRendererCfg(),
        my_variant_a=NewtonWarpRendererCfg(),
        my_variant_b=NewtonWarpRendererCfg(),
    )
    violations = _drift_violations(bad_preset)
    assert len(violations) == 1
    assert "newton_renderer" in violations[0]


def test_drift_lint_catches_name_type_mismatch():
    """A field named after a registered canonical must hold a value of that
    canonical's class. Naming a field ``physx`` while assigning a renderer
    cfg there shadows the real anchor and is reported as drift.
    """
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg
    from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

    from isaaclab_tasks.utils.hydra import preset

    # 'physx' is the registered PHYSICS canonical; here we (wrongly) assign
    # a renderer cfg, so the value's canonical resolves to 'newton_renderer'.
    bad_preset = preset(
        default=PhysxCfg(),
        physx=NewtonWarpRendererCfg(),
    )
    violations = _drift_violations(bad_preset)
    assert any("name and value class must match" in v for v in violations), violations


def test_drift_lint_allows_canonical_name_with_unregistered_value():
    """The IsaacLab PresetCfg dispatch pattern uses canonical names like
    ``physx`` / ``newton_mjwarp`` as preset KEYS that hold arbitrary
    backend-tuned values (e.g. an ``ArticulationCfg`` whose tuning is
    physx-flavored). The value's class isn't itself a registered preset;
    the field name is just selecting which value to use when ``physx`` is
    the active preset. This is legitimate and must not trigger drift.
    """

    from isaaclab_tasks.utils.hydra import preset

    class _PlainArticulationCfg:
        pass

    ok_preset = preset(
        default=_PlainArticulationCfg(),
        physx=_PlainArticulationCfg(),  # canonical name + unregistered value: dispatch pattern
        newton_mjwarp=_PlainArticulationCfg(),
    )
    assert _drift_violations(ok_preset) == []


def test_drift_lint_accepts_canonical_plus_variants():
    """Unit test for the loose drift logic: a canonical-named field with
    additional variants of the same class is fine."""
    from isaaclab_newton.renderers.newton_warp_renderer_cfg import NewtonWarpRendererCfg

    from isaaclab_tasks.utils.hydra import preset

    good_preset = preset(
        default=NewtonWarpRendererCfg(),
        newton_renderer=NewtonWarpRendererCfg(),
        newton_renderer_strict=NewtonWarpRendererCfg(),
    )
    assert _drift_violations(good_preset) == []


def test_no_canonical_vocabulary_drift_in_registered_tasks():
    """CI lint: every PresetCfg subclass in any registered task must include
    at least one canonical-named field per backend-class group. Variants are
    allowed alongside the canonical name; standalone variants (no canonical)
    fail because they make the CLI surface ambiguous.

    Skipped tasks (typically ones whose env-cfg load raises before we can
    inspect them) are reported on stderr so CI catches surprises.
    """
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401  -- triggers gym registration
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

        def _record(preset_obj, _path):
            for msg in _drift_violations(preset_obj):
                violations.append((task_id, msg))

        try:
            _walk_preset_cfgs(env_cfg, _record)
        except BaseException as exc:  # noqa: BLE001
            skipped.append((task_id, f"walk failed: {type(exc).__name__}: {exc}"))

    if violations:
        formatted = "\n".join(f"  [{tid}] {msg}" for tid, msg in violations)
        pytest.fail(f"PresetCfg drift detected:\n{formatted}")

    # Tolerated skip reasons: environment-level imports that aren't available
    # in headless / CI runs. A broken task config that doesn't match these
    # patterns still hard-fails the lint so a drift can't hide behind a
    # bogus skip.
    tolerated_skip_substrings = (
        "No module named 'carb'",  # Isaac Sim runtime missing
    )
    unexpected_skips = [
        (tid, reason) for tid, reason in skipped
        if not any(substr in reason for substr in tolerated_skip_substrings)
    ]
    if unexpected_skips:
        formatted = "\n".join(f"  [{tid}] {reason}" for tid, reason in unexpected_skips)
        pytest.fail(
            f"Drift lint could not load {len(unexpected_skips)} task(s); coverage is incomplete.\n"
            f"{formatted}\n"
            "Either fix the import or extend ``tolerated_skip_substrings`` in this test."
        )
