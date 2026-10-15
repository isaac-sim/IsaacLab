# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task preset discovery, CLI help, and isolation from launcher arguments.

Preset-name validation and task composition are covered in ``test_hydra.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="train.py", add_help=False)
    parser.add_argument("--task", type=str, default=None)
    return parser


# ---------------------------------------------------------------------------
# setup_preset_cli: pass task selectors through without changing launcher settings
# ---------------------------------------------------------------------------


def test_setup_preset_cli_returns_remainder_only(monkeypatch):
    """Without any preset tokens, the remainder is just the un-touched
    non-argparse tokens (Hydra path overrides, etc.)."""
    original = ["train.py", "--task=Foo-v0", "env.sim.dt=0.001"]
    monkeypatch.setattr("sys.argv", original)
    from isaaclab_tasks.utils.hydra import setup_preset_cli

    args, remaining = setup_preset_cli(_make_parser())
    assert args.task == "Foo-v0"
    assert remaining == ["env.sim.dt=0.001"]
    # setup_preset_cli must NOT mutate sys.argv -- the caller controls when to assign.
    assert sys.argv == original


def test_setup_preset_cli_passes_typed_tokens_verbatim(monkeypatch):
    """Preset tokens come back in their original ``physics=`` / ``renderer=`` /
    ``presets=`` form so hydra can parse them directly and callers can intersect
    with callback returns in matching vocabulary."""
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
    from isaaclab_tasks.utils.hydra import setup_preset_cli

    _, remaining = setup_preset_cli(_make_parser())
    assert remaining == [
        "physics=newton_mjwarp",
        "renderer=newton_renderer",
        "presets=albedo,depth",
        "env.sim.dt=0.001",
    ]


def test_setup_preset_cli_namespace_carries_no_preset_attributes(monkeypatch):
    """Preset tokens are never registered with argparse, so the parsed
    Namespace gains no ``physics`` / ``renderer`` / ``presets`` attribute.

    This is the bug-class-level guarantee against the Kit launcher's name-based
    forwarding (``_SIM_APP_CONFIG_KEYS & launcher_args.keys()``,
    ``kit_launcher.py``): an attribute that doesn't exist can't collide.
    """
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--task=Foo-v0", "physics=newton_mjwarp", "renderer=newton_renderer", "presets=albedo"],
    )
    from isaaclab_tasks.utils.hydra import setup_preset_cli

    args, _ = setup_preset_cli(_make_parser())
    for attr in ("physics", "renderer", "presets"):
        assert not hasattr(args, attr), (
            f"setup_preset_cli wrote ``args.{attr}`` to the namespace -- the Kit launcher's name-based"
            " forwarding can then push it into SimulationApp config. Drop the argparse registration"
            " for preset selectors and use Hydra-style tokens instead."
        )


def test_intersection_preserves_typed_selection():
    """A typed selector present in both the remainder and the callback return
    survives ``list_intersection`` verbatim; a callback-owned flag is dropped."""
    from isaaclab.utils.string import list_intersection

    main_remainder = ["physics=newton_mjwarp", "--my_callback_flag=42", "env.lr=3e-4"]
    callback_remainder = ["physics=newton_mjwarp", "env.lr=3e-4"]

    intersected = list_intersection(main_remainder, callback_remainder)

    assert intersected == ["physics=newton_mjwarp", "env.lr=3e-4"]


def test_enumerate_task_presets_routes_by_target_match(monkeypatch):
    """Variants bucket through :meth:`PresetTarget.matches`.

    PhysicsCfg subclass instances route to PHYSICS, RendererCfg subclass
    instances route to RENDERER, PhysicsCfg-containing SimulationCfg bundles
    route to PHYSICS, and values matching no target fall into DOMAIN.
    """
    from isaaclab.physics import PhysicsCfg
    from isaaclab.renderers.renderer_cfg import RendererCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass

    from isaaclab_tasks.utils.hydra import PresetTarget, enumerate_task_presets, preset

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
        "sim": {
            "default": SimulationCfg(dt=1 / 60, physics=_PhysVariant()),
            "simulation_physx": SimulationCfg(dt=1 / 60, physics=_PhysVariant()),
        },
        "sim_no_backend": {
            "default": SimulationCfg(dt=1 / 240),
            "plain": SimulationCfg(dt=1 / 240),
        },
        "weight": {  # cfgs whose type is not a typed-target base subclass -> DOMAIN
            "default": 1.0,
            "light": 0.5,
            "heavy": 2.0,
        },
    }
    monkeypatch.setattr(
        "isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry",
        lambda *_: {path: preset(**fields) for path, fields in walked.items()},
    )
    result = enumerate_task_presets("Test")
    # All physics variants bucket to PHYSICS (including the wrapper-shaped ones).
    assert {"physx", "newton_mjwarp", "newton_kamino", "simulation_physx"} <= set(result[PresetTarget.PHYSICS])
    assert "newton_renderer" in result[PresetTarget.RENDERER]
    # Primitive-typed variants land in DOMAIN.
    assert {"plain", "light", "heavy"} <= set(result[PresetTarget.DOMAIN])
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
    from isaaclab_tasks.utils import hydra, setup_preset_cli

    # Task preset parsing, categories, and resolution share one module.
    assert not any(Path(hydra.__file__).with_name(module).exists() for module in ("preset_cli.py", "preset_target.py"))
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
                "physics=NAME (typed) selects a physics backend. Available: (none)",
                "renderer=NAME (typed) selects a renderer backend. Available: (none)",
                "presets=NAME[,NAME,...] broadcast: applied to every matching PresetCfg. Available: (none)",
            ],
            id="zero_variants_everywhere",
        ),
        pytest.param(
            "mixed",
            [
                "physics=NAME (typed) selects a physics backend. Available: - my_phys",
                "renderer=NAME (typed) selects a renderer backend. Available: - my_rend",
                "presets=NAME[,NAME,...] broadcast: applied to every matching PresetCfg. Available: - heavy - light",
            ],
            id="all_three_buckets_populated",
        ),
    ],
)
def test_help_text_branch_strings(monkeypatch, capsys, build_key, expected_phrases):
    """Help uses the supplied argv's last task and lists variants in their typed sections."""
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
    class _MixedCfg:
        physics: object = preset(default=_HelpPhysCfg(), my_phys=_HelpPhysCfg())
        renderer: object = preset(default=_HelpRendCfg(), my_rend=_HelpRendCfg())
        weight: object = preset(default=1.0, light=0.5, heavy=2.0)

    builders = {
        "empty": _EmptyCfg,
        "mixed": _MixedCfg,
    }

    import isaaclab_tasks.utils.parse_cfg as parse_cfg

    def load(task, entry):
        assert task == "Fake-v0"
        return builders[build_key]()

    monkeypatch.setattr(parse_cfg, "load_cfg_from_registry", load)
    monkeypatch.setattr("sys.argv", ["train.py", "--task=Unrelated-v0", "--help"])
    argv = (
        ["--task=Old-v0", "--task", "Fake-v0", "-h"]
        if build_key == "empty"
        else ["--task", "Old-v0", "--task=Fake-v0", "--help"]
    )
    from isaaclab_tasks.utils.hydra import setup_preset_cli

    parser = argparse.ArgumentParser(prog="train.py")
    parser.add_argument("--task", type=str, default=None)
    with pytest.raises(SystemExit):
        setup_preset_cli(parser, argv)
    # Collapse argparse line-wrapping so substring checks survive width changes.
    flat = " ".join(capsys.readouterr().out.split())

    for phrase in expected_phrases:
        assert phrase in flat, f"Missing phrase: {phrase!r}"
