# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cartpole perception task consolidation under the preset CLI.

The retired per-data-type and per-backbone Cartpole task IDs each route their
``env_cfg_entry_point`` through a deprecation factory that emits a
``DeprecationWarning`` and returns the equivalent preset variant of the new
consolidated cfg. These tests lock that contract:

* Every retired task ID emits a ``DeprecationWarning`` exactly once that
  names the new consolidated task ID and the corresponding ``presets=`` name.
* The new ``Isaac-Cartpole-Camera-Direct-v0`` and ``Isaac-Cartpole-Camera-v0``
  task IDs load cleanly with no deprecation warning of our own.

The cfg returned by the deprecation factory is not opened/instantiated here --
that path goes through configclass __post_init__, which surfaces unrelated
pre-existing deprecations (e.g. ``RigidBodyMaterialCfg``) that aren't part of
this contract.
"""

from __future__ import annotations

import warnings

import pytest

# Tasks must be registered.
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# (deprecated task id, expected migration suffix appearing in the warning body)
_DIRECT_DEPRECATIONS = [
    ("Isaac-Cartpole-Camera-Presets-Direct-v0", "--task=Isaac-Cartpole-Camera-Direct-v0"),
    ("Isaac-Cartpole-RGB-Camera-Direct-v0", "--task=Isaac-Cartpole-Camera-Direct-v0 presets=rgb"),
    ("Isaac-Cartpole-Depth-Camera-Direct-v0", "--task=Isaac-Cartpole-Camera-Direct-v0 presets=depth"),
    ("Isaac-Cartpole-Albedo-Camera-Direct-v0", "--task=Isaac-Cartpole-Camera-Direct-v0 presets=albedo"),
    (
        "Isaac-Cartpole-SimpleShading-Constant-Camera-Direct-v0",
        "--task=Isaac-Cartpole-Camera-Direct-v0 presets=simple_shading_constant_diffuse",
    ),
    (
        "Isaac-Cartpole-SimpleShading-Diffuse-Camera-Direct-v0",
        "--task=Isaac-Cartpole-Camera-Direct-v0 presets=simple_shading_diffuse_mdl",
    ),
    (
        "Isaac-Cartpole-SimpleShading-Full-Camera-Direct-v0",
        "--task=Isaac-Cartpole-Camera-Direct-v0 presets=simple_shading_full_mdl",
    ),
]

_MANAGER_DEPRECATIONS = [
    ("Isaac-Cartpole-RGB-v0", "--task=Isaac-Cartpole-Camera-v0 presets=rgb"),
    ("Isaac-Cartpole-Depth-v0", "--task=Isaac-Cartpole-Camera-v0 presets=depth"),
    (
        "Isaac-Cartpole-RGB-ResNet18-v0",
        "--task=Isaac-Cartpole-Camera-v0 --agent=rl_games_feature_cfg_entry_point presets=resnet18",
    ),
    (
        "Isaac-Cartpole-RGB-TheiaTiny-v0",
        "--task=Isaac-Cartpole-Camera-v0 --agent=rl_games_feature_cfg_entry_point presets=theia_tiny",
    ),
]


def _showcase_migration(new_id: str, preset: str) -> str:
    """Expected migration command for a retired showcase task ID.

    box_box matches the consolidated task's default skrl yaml so no --agent
    is needed; every other variant needs an explicit
    --agent=skrl_<obs>_<action>_cfg_entry_point. Convention: --task first,
    --agent next when present, presets= selector last.
    """
    tokens = [f"--task={new_id}"]
    if preset != "box_box":
        tokens.append(f"--agent=skrl_{preset}_cfg_entry_point")
    tokens.append(f"presets={preset}")
    return " ".join(tokens)


# Proprioceptive cartpole showcase: 15 (observation, action) shape combinations
# collapse into Isaac-Cartpole-Showcase-Direct-v0 via presets=<obs>_<action>.
_PROPRIO_SHOWCASE_DEPRECATIONS = [
    (
        f"Isaac-Cartpole-Showcase-{obs_label}-{act_label}-Direct-v0",
        _showcase_migration("Isaac-Cartpole-Showcase-Direct-v0", f"{obs}_{act}"),
    )
    for obs_label, obs in [
        ("Box", "box"),
        ("Discrete", "discrete"),
        ("MultiDiscrete", "multidiscrete"),
        ("Dict", "dict"),
        ("Tuple", "tuple"),
    ]
    for act_label, act in [("Box", "box"), ("Discrete", "discrete"), ("MultiDiscrete", "multidiscrete")]
]

# Camera-based cartpole showcase: 9 (observation, action) shape combinations
# collapse into Isaac-Cartpole-Camera-Showcase-Direct-v0 via
# presets=<obs>_<action>.
_CAMERA_SHOWCASE_DEPRECATIONS = [
    (
        f"Isaac-Cartpole-Camera-Showcase-{obs_label}-{act_label}-Direct-v0",
        _showcase_migration("Isaac-Cartpole-Camera-Showcase-Direct-v0", f"{obs}_{act}"),
    )
    for obs_label, obs in [("Box", "box"), ("Dict", "dict"), ("Tuple", "tuple")]
    for act_label, act in [("Box", "box"), ("Discrete", "discrete"), ("MultiDiscrete", "multidiscrete")]
]


def _load_capturing_task_deprecations(task_id: str) -> list[str]:
    """Return the task-deprecation warning messages emitted when loading *task_id*.

    Filters to warnings whose body starts with ``"Task '"`` so unrelated
    configclass-level deprecations don't pollute the assertion.
    """
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        load_cfg_from_registry(task_id, "env_cfg_entry_point")
    return [
        str(w.message)
        for w in captured
        if issubclass(w.category, DeprecationWarning) and str(w.message).startswith("Task '")
    ]


@pytest.mark.parametrize(
    "task_id, migration",
    _DIRECT_DEPRECATIONS + _MANAGER_DEPRECATIONS + _PROPRIO_SHOWCASE_DEPRECATIONS + _CAMERA_SHOWCASE_DEPRECATIONS,
)
def test_retired_task_id_emits_deprecation_pointing_at_new_task(task_id: str, migration: str) -> None:
    """Each retired cartpole perception task ID emits a DeprecationWarning
    with the canonical ``Task 'X' is deprecated ... Use '<migration>'.`` body."""
    messages = _load_capturing_task_deprecations(task_id)
    assert len(messages) == 1, f"{task_id}: expected 1 task-deprecation warning, got {len(messages)}: {messages}"
    expected = f"Task '{task_id}' is deprecated and will be removed in a future release. Use '{migration}'."
    assert messages[0] == expected, (
        f"warning body mismatch for {task_id}:\n  got:      {messages[0]!r}\n  expected: {expected!r}"
    )


@pytest.mark.parametrize(
    "task_id",
    [
        "Isaac-Cartpole-Camera-Direct-v0",
        "Isaac-Cartpole-Camera-v0",
        "Isaac-Cartpole-Showcase-Direct-v0",
        "Isaac-Cartpole-Camera-Showcase-Direct-v0",
    ],
)
def test_consolidated_task_id_loads_without_deprecation(task_id: str) -> None:
    """The new consolidated task IDs load with no task-deprecation warning
    from this refactor."""
    messages = _load_capturing_task_deprecations(task_id)
    assert messages == [], f"{task_id}: unexpected deprecation warning(s): {messages}"


# Each retired direct-camera task ID returns its historical per-variant
# subclass via cfg_factory, so the camera offset rotation matches develop's
# pre-deprecation value bit-for-bit. RGB and Depth subclasses use the
# identity rotation; the Albedo and SimpleShading subclasses ship a
# 180-degree flip that the consolidated cfg's matching variant does not
# preserve (a long-standing latent divergence on develop between the
# Style A per-variant subclasses and the Style B consolidated PresetCfg).
_IDENTITY_ROT = (0.0, 0.0, 0.0, 1.0)
_FLIPPED_ROT = (1.0, 0.0, 0.0, 0.0)

# (deprecated direct camera task id, expected data_types value,
#  expected obs channels, expected offset rotation).
# Verifies the shim returns the historical per-variant subclass with its
# observation_space, tiled_camera.data_types, and camera offset preserved.
_DIRECT_CAMERA_SHAPE_PINS = [
    ("Isaac-Cartpole-RGB-Camera-Direct-v0", ["rgb"], 3, _IDENTITY_ROT),
    ("Isaac-Cartpole-Depth-Camera-Direct-v0", ["depth"], 1, _IDENTITY_ROT),
    ("Isaac-Cartpole-Albedo-Camera-Direct-v0", ["albedo"], 3, _FLIPPED_ROT),
    ("Isaac-Cartpole-SimpleShading-Constant-Camera-Direct-v0", ["simple_shading_constant_diffuse"], 3, _FLIPPED_ROT),
    ("Isaac-Cartpole-SimpleShading-Diffuse-Camera-Direct-v0", ["simple_shading_diffuse_mdl"], 3, _FLIPPED_ROT),
    ("Isaac-Cartpole-SimpleShading-Full-Camera-Direct-v0", ["simple_shading_full_mdl"], 3, _FLIPPED_ROT),
]


@pytest.mark.parametrize("task_id, expected_data_types, expected_channels, expected_rot", _DIRECT_CAMERA_SHAPE_PINS)
def test_direct_camera_shim_returns_historical_per_variant_cfg(
    task_id: str,
    expected_data_types: list[str],
    expected_channels: int,
    expected_rot: tuple[float, float, float, float],
) -> None:
    """The direct-camera deprecation shim must return the historical
    per-variant subclass bit-for-bit (via ``cfg_factory=lambda: SubClass()``
    at the call site), preserving its observation_space, tiled_camera
    data_types, and camera offset rotation -- including the 180-degree flip
    that the Albedo / SimpleShading subclasses ship.
    """
    # Suppress warning emission noise -- the warning text is locked elsewhere.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    # observation_space matches the old per-variant class.
    assert cfg.observation_space[-1] == expected_channels, (
        f"{task_id}: observation_space[-1] = {cfg.observation_space[-1]!r}, expected {expected_channels}"
    )
    # tiled_camera was resolved -- not the wrapping MultiDataTypeCartpoleTiledCameraCfg.
    assert hasattr(cfg.tiled_camera, "data_types"), (
        f"{task_id}: tiled_camera is not resolved to a concrete CameraCfg "
        f"(got {type(cfg.tiled_camera).__name__}). The shim left the nested PresetCfg unresolved."
    )
    assert cfg.tiled_camera.data_types == expected_data_types, (
        f"{task_id}: tiled_camera.data_types = {cfg.tiled_camera.data_types!r}, expected {expected_data_types!r}"
    )
    assert cfg.tiled_camera.offset.rot == expected_rot, (
        f"{task_id}: tiled_camera.offset.rot = {cfg.tiled_camera.offset.rot!r}, expected {expected_rot!r}"
    )


# (deprecated manager-perception task id, expected env_cfg class name).
# Each retired ID returns its historical per-variant cfg subclass verbatim
# via the deprecation shim's ``cfg_factory``, so the class name is the
# pre-deprecation source of truth.
_MANAGER_CAMERA_CLASS_PINS = [
    ("Isaac-Cartpole-RGB-v0", "CartpoleRGBCameraEnvCfg"),
    ("Isaac-Cartpole-Depth-v0", "CartpoleDepthCameraEnvCfg"),
    ("Isaac-Cartpole-RGB-ResNet18-v0", "CartpoleResNet18CameraEnvCfg"),
    ("Isaac-Cartpole-RGB-TheiaTiny-v0", "CartpoleTheiaTinyCameraEnvCfg"),
]


# (deprecated showcase task id, expected env_cfg class name). The showcase
# shim picks a per-shape cfg class out of the consolidated PresetCfg; the
# class names match the (Obs, Action) labels in the retired task id.
_SHOWCASE_CLASS_PINS = [
    (f"Isaac-Cartpole-Showcase-{obs_label}-{act_label}-Direct-v0", f"{obs_label}{act_label}EnvCfg")
    for obs_label in ("Box", "Discrete", "MultiDiscrete", "Dict", "Tuple")
    for act_label in ("Box", "Discrete", "MultiDiscrete")
] + [
    (f"Isaac-Cartpole-Camera-Showcase-{obs_label}-{act_label}-Direct-v0", f"{obs_label}{act_label}EnvCfg")
    for obs_label in ("Box", "Dict", "Tuple")
    for act_label in ("Box", "Discrete", "MultiDiscrete")
]


@pytest.mark.parametrize("task_id, expected_cls_name", _MANAGER_CAMERA_CLASS_PINS + _SHOWCASE_CLASS_PINS)
def test_flat_shim_resolves_to_expected_cfg_class(task_id: str, expected_cls_name: str) -> None:
    """Each flat (non-nested) deprecation shim must return an instance of the
    historical per-variant cfg class -- not the consolidated PresetCfg
    wrapper, not the default variant. Locks the variant resolution beyond
    the warning text alone."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    assert type(cfg).__name__ == expected_cls_name, (
        f"{task_id}: shim returned {type(cfg).__name__}, expected {expected_cls_name}"
    )
