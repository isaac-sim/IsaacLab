# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

NUM_CAMERAS = 4
CAMERA_HEIGHT = 64
CAMERA_WIDTH = 64


def _make_camera_cfg(frame_stack: int = 1) -> CameraCfg:
    return CameraCfg(
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        frame_stack=frame_stack,
    )


def _populate_scene():
    """Add minimal prims to the scene."""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.SphereLightCfg()
    cfg.func("/World/Light", cfg, translation=(0.0, 0.0, 10.0))
    for i in range(NUM_CAMERAS):
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform")
        sim_utils.create_prim(
            f"/World/Origin_{i}/Cube",
            "Cube",
            translation=(0.0, 0.0, 1.0),
            scale=(0.5, 0.5, 0.5),
        )


@pytest.fixture(scope="function")
def setup_scene():
    """Set up a minimal simulation scene and tear it down after the test."""
    sim_utils.create_new_stage()
    dt = 0.01
    sim_cfg = sim_utils.SimulationCfg(dt=dt)
    sim = sim_utils.SimulationContext(sim_cfg)
    _populate_scene()
    sim_utils.update_stage()
    yield sim
    sim.clear_instance()


# -- Imports needed by _populate_scene but deferred until after AppLauncher --
from pxr import Gf, UsdGeom  # noqa: E402


@pytest.mark.isaacsim_ci
def test_frame_stack_1_preserves_shape(setup_scene):
    """frame_stack=1 should produce the standard (N, H, W, 3) output."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=1))
    sim.reset()
    camera.update(dt=0.01)

    rgb = camera.data.output["rgb"]
    assert rgb.shape == (NUM_CAMERAS, CAMERA_HEIGHT, CAMERA_WIDTH, 3), f"Expected (N, H, W, 3), got {rgb.shape}"


@pytest.mark.isaacsim_ci
def test_frame_stack_2_doubles_channels(setup_scene):
    """frame_stack=2 should produce (N, H, W, 6) output."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=2))
    sim.reset()
    # Step twice to fill the buffer
    camera.update(dt=0.01)
    camera.update(dt=0.01)

    rgb = camera.data.output["rgb"]
    assert rgb.shape == (NUM_CAMERAS, CAMERA_HEIGHT, CAMERA_WIDTH, 6), f"Expected (N, H, W, 6), got {rgb.shape}"


@pytest.mark.isaacsim_ci
def test_frame_stack_init_fills_history(setup_scene):
    """On first update, all history slots should be filled with the same frame."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=2))
    sim.reset()
    camera.update(dt=0.01)

    rgb = camera.data.output["rgb"]
    # With 2-frame stack, channels 0:3 and 3:6 should be identical (same frame copied)
    first_frame = rgb[..., :3]
    second_frame = rgb[..., 3:6]
    assert torch.equal(first_frame, second_frame), "After first update, all history slots should contain the same frame"


@pytest.mark.isaacsim_ci
def test_frame_stack_ring_buffer_shifts_correctly(setup_scene):
    """After a camera move, the ring buffer should shift: old newest becomes new oldest."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=2))
    sim.reset()
    # Pump one ``sim.step()`` before the first ``camera.update()`` so RTX has fully primed
    # its tiled annotators (mirrors the canonical pattern from ``test_first_frame_textured_rendering``).
    sim.step()
    camera.update(dt=0.01)

    # Capture the pre-move newest frame (channels 3:6)
    rgb_before = camera.data.output["rgb"].clone()
    pre_move_newest = rgb_before[..., 3:6].clone()

    # Move the camera to guarantee a different rendered frame
    for prim in camera._sensor_prims:
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 4.0))

    sim.step()
    camera.update(dt=0.01)

    rgb_after = camera.data.output["rgb"]
    post_move_oldest = rgb_after[..., :3]
    post_move_newest = rgb_after[..., 3:6]
    # Snapshot the renderer's single-frame buffer for the second update so we can verify
    # the ring buffer copied it verbatim into the newest slot (independent of pixel content,
    # which can be a zero buffer on rare RTX cold-start iterations).
    latest_single = camera._single_frame_output["rgb"].clone()

    # Ring-buffer invariant 1: the pre-move newest frame becomes the post-move oldest.
    assert torch.equal(pre_move_newest, post_move_oldest), (
        "Ring buffer should shift: previous newest frame becomes the new oldest"
    )
    # Ring-buffer invariant 2: the newest slot post-update equals the latest single-frame
    # renderer output (catches the case where ``history[idx].copy_(single)`` doesn't land
    # or where ``_stacked_output`` isn't refreshed). We compare against the single buffer
    # rather than asserting visual difference, because that conflates ring-buffer correctness
    # with renderer behavior.
    assert torch.equal(post_move_newest, latest_single), (
        "Ring buffer's newest slot should contain the latest renderer output"
    )


@pytest.mark.isaacsim_ci
def test_frame_stack_reset_clears_history(setup_scene):
    """After reset, history should be re-initialized with the current frame."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=2))
    sim.reset()

    # Step a few times to build up different history
    for _ in range(5):
        sim.step()
        camera.update(dt=0.01)

    # Reset specific envs
    env_ids = torch.tensor([0], dtype=torch.long)
    camera.reset(env_ids=env_ids)
    camera.update(dt=0.01)

    # After reset + update, env 0's history should be all identical (re-filled)
    rgb = camera.data.output["rgb"]
    first_frame_env0 = rgb[0, :, :, :3]
    second_frame_env0 = rgb[0, :, :, 3:6]
    assert torch.equal(first_frame_env0, second_frame_env0), (
        "After reset, env 0's history slots should contain the same frame"
    )


@pytest.mark.isaacsim_ci
def test_frame_stack_partial_reset_preserves_others(setup_scene):
    """Resetting env 0 should not affect env 1's history."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=2))
    sim.reset()

    # Step to build history
    for _ in range(3):
        sim.step()
        camera.update(dt=0.01)

    # Capture env 1's stacked output before reset
    rgb_before = camera.data.output["rgb"][1].clone()

    # Reset only env 0
    camera.reset(env_ids=torch.tensor([0], dtype=torch.long))
    camera.update(dt=0.01)

    # env 1 should still have progressed (new frame), not be reset
    rgb_after = camera.data.output["rgb"][1]
    # Shape should be preserved
    assert rgb_after.shape == rgb_before.shape


@pytest.mark.isaacsim_ci
def test_frame_stack_default_is_one():
    """CameraCfg should default to frame_stack=1."""
    cfg = CameraCfg()
    assert cfg.frame_stack == 1


# -- Renderer capability flag tests --
# These test :meth:`RendererCfg.provides_temporal_camera_data` directly across each subclass.
# RTX is dynamic on the active AA mode; Newton Warp is always False; the base class is always True.

from isaaclab.renderers.renderer_cfg import RendererCfg  # noqa: E402
from isaaclab.sim.simulation_cfg import RenderCfg  # noqa: E402
from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: E402
from isaaclab_physx.renderers import IsaacRtxRendererCfg  # noqa: E402


@pytest.mark.isaacsim_ci
def test_base_renderer_provides_temporal_default_true():
    """Base RendererCfg defaults to providing temporal data (True)."""
    cfg = RendererCfg()
    assert cfg.provides_temporal_camera_data(None) is True
    assert cfg.provides_temporal_camera_data(RenderCfg()) is True


@pytest.mark.isaacsim_ci
def test_newton_warp_renderer_provides_temporal_false():
    """NewtonWarpRendererCfg overrides to False (pure rasterization, no prior-frame blending)."""
    cfg = NewtonWarpRendererCfg()
    assert cfg.provides_temporal_camera_data(None) is False
    assert cfg.provides_temporal_camera_data(RenderCfg()) is False
    assert cfg.provides_temporal_camera_data(RenderCfg(antialiasing_mode="DLSS")) is False


@pytest.mark.isaacsim_ci
def test_rtx_renderer_temporal_with_no_render_cfg():
    """RTX with no sim_render_cfg falls back to upstream default (DLSS, temporal)."""
    cfg = IsaacRtxRendererCfg()
    assert cfg.provides_temporal_camera_data(None) is True


@pytest.mark.isaacsim_ci
def test_rtx_renderer_temporal_with_default_aa_mode():
    """RTX with antialiasing_mode=None uses upstream default (DLSS), which is temporal."""
    cfg = IsaacRtxRendererCfg()
    assert cfg.provides_temporal_camera_data(RenderCfg(antialiasing_mode=None)) is True


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("mode", ["DLSS", "DLAA", "TAA"])
def test_rtx_renderer_temporal_aa_modes(mode):
    """RTX with DLSS/DLAA/TAA reports temporal=True (these blend prior-frame data)."""
    cfg = IsaacRtxRendererCfg()
    assert cfg.provides_temporal_camera_data(RenderCfg(antialiasing_mode=mode)) is True


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("mode", ["FXAA", "Off"])
def test_rtx_renderer_no_temporal_aa_modes(mode):
    """RTX with FXAA or Off reports temporal=False (spatial-only AA, no prior-frame blending)."""
    cfg = IsaacRtxRendererCfg()
    assert cfg.provides_temporal_camera_data(RenderCfg(antialiasing_mode=mode)) is False


# -- Newton warning and preset resolution tests --
# These require isaaclab_tasks for launch_simulation and preset configs.

import logging

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_tasks.utils.sim_launcher import launch_simulation

_warning_capture: list[str] = []


@pytest.fixture(autouse=True, scope="session")
def _install_launcher_warning_capture():
    """Capture warnings from sim_launcher logger."""

    class _Handler(logging.Handler):
        def handle(self, record):
            _warning_capture.append(record.getMessage())
            return True

    handler = _Handler(level=logging.WARNING)
    logger = logging.getLogger("isaaclab_tasks.utils.sim_launcher")
    logger.addHandler(handler)
    yield
    logger.removeHandler(handler)


# -- Warning matrix: physics × renderer × frame_stack -----------------------------------------
#
# The warning fires only when ALL of:
#   - physics.requires_temporal_camera_data == True
#   - renderer.provides_temporal_camera_data(sim.render) == False
#   - any camera has frame_stack <= 1
#
# Matrix coverage (cartpole-camera-presets task):
#
#   physics | renderer       | AA mode | stack | warn? | covered by
#   --------|----------------|---------|-------|-------|------------
#   PhysX   | RTX            | default | 1     | NO    | test_no_warn_physx_rtx
#   PhysX   | Warp           | n/a     | 1     | NO    | test_no_warn_physx_warp
#   Newton  | RTX (DLSS)     | default | 1     | NO    | test_no_warn_newton_rtx_dlss
#   Newton  | RTX (FXAA)     | FXAA    | 1     | YES   | test_warn_newton_rtx_fxaa
#   Newton  | Warp           | n/a     | 1     | YES   | test_warn_newton_warp
#   Newton  | Warp           | n/a     | 2     | NO    | test_no_warn_newton_warp_stacked


def _resolve_with_presets(presets_arg: str):
    """Resolve the cartpole-camera task config with the given ``presets=...`` value."""
    import sys

    old_argv = sys.argv
    sys.argv = [sys.argv[0], f"presets={presets_arg}"]
    try:
        env_cfg, _ = resolve_task_config("Isaac-Cartpole-Camera-Presets-Direct-v0", "skrl_cfg_entry_point")
    finally:
        sys.argv = old_argv
    return env_cfg


def _run_launcher_and_capture(env_cfg) -> list[str]:
    """Run launch_simulation once and return captured warnings."""
    _warning_capture.clear()
    with launch_simulation(env_cfg, {"enable_cameras": True}):
        pass
    return list(_warning_capture)


@pytest.mark.isaacsim_ci
def test_no_warn_physx_rtx():
    """PhysX has implicit damping; RTX provides temporal. No warning regardless of frame_stack."""
    env_cfg = _resolve_with_presets("physx")
    env_cfg.tiled_camera.frame_stack = 1
    warnings = _run_launcher_and_capture(env_cfg)
    assert not any("frame_stack" in w for w in warnings), f"PhysX+RTX should not warn: {warnings}"


@pytest.mark.isaacsim_ci
def test_no_warn_physx_warp():
    """PhysX has implicit damping. No warning even with Warp renderer."""
    env_cfg = _resolve_with_presets("physx,newton_renderer")
    env_cfg.tiled_camera.frame_stack = 1
    warnings = _run_launcher_and_capture(env_cfg)
    assert not any("frame_stack" in w for w in warnings), f"PhysX+Warp should not warn: {warnings}"


@pytest.mark.isaacsim_ci
def test_no_warn_newton_rtx_dlss():
    """Newton needs temporal data, but RTX with default DLSS provides it. No warning."""
    env_cfg = _resolve_with_presets("newton")
    env_cfg.tiled_camera.frame_stack = 1  # override the preset's frame_stack=2 to test renderer-only path
    # Default antialiasing_mode is None which falls back to DLSS (temporal).
    warnings = _run_launcher_and_capture(env_cfg)
    assert not any("frame_stack" in w for w in warnings), (
        f"Newton+RTX(DLSS) should not warn (renderer provides temporal): {warnings}"
    )


@pytest.mark.isaacsim_ci
def test_warn_newton_rtx_fxaa():
    """Newton + RTX with user-overridden FXAA (spatial-only) should fire the warning dynamically."""
    env_cfg = _resolve_with_presets("newton")
    env_cfg.tiled_camera.frame_stack = 1
    env_cfg.sim.render.antialiasing_mode = "FXAA"
    warnings = _run_launcher_and_capture(env_cfg)
    assert any("frame_stack" in w for w in warnings), (
        f"Newton+RTX(FXAA) should warn (renderer no longer provides temporal): {warnings}"
    )


@pytest.mark.isaacsim_ci
def test_warn_newton_warp():
    """Newton + Warp + frame_stack=1: physics needs temporal, renderer doesn't provide. Warning fires."""
    env_cfg = _resolve_with_presets("newton,newton_renderer")
    env_cfg.tiled_camera.frame_stack = 1
    warnings = _run_launcher_and_capture(env_cfg)
    assert any("frame_stack" in w for w in warnings), f"Newton+Warp should warn: {warnings}"


@pytest.mark.isaacsim_ci
def test_no_warn_newton_warp_stacked():
    """Newton + Warp + frame_stack=2: explicit temporal data via stacking. Warning suppressed."""
    env_cfg = _resolve_with_presets("newton,newton_renderer")
    env_cfg.tiled_camera.frame_stack = 2
    warnings = _run_launcher_and_capture(env_cfg)
    assert not any("frame_stack" in w for w in warnings), (
        f"Newton+Warp+stack=2 should not warn (explicit temporal): {warnings}"
    )


# -- Frame stack policy: nested PresetCfg + runtime auto-apply -------------------------------
#
# Architecture (proxy pattern):
#   - MultiBackendCameraCfg.frame_stack: int = 0  (sentinel for "no user override; use policy")
#   - MultiBackendCameraCfg.frame_stack_policy: FrameStackPolicyCfg = ...
#       - Outer keys on physics preset name; default=0
#       - newton branch is _FrameStackPolicyBranch (regular configclass) holding the renderer-keyed
#         inner _FrameStackPolicyByRenderer (default=0, newton_renderer=2)
#   - At launch time, _apply_frame_stack_policies() walks env_cfg and propagates the resolved
#     policy onto frame_stack when frame_stack == 0. User-supplied values are respected.
#   - Camera reads frame_stack via max(1, ...) so the 0 sentinel is equivalent to 1 at runtime.

from isaaclab_tasks.utils.sim_launcher import _apply_frame_stack_policies, _resolve_frame_stack_policy  # noqa: E402


def _resolve_with_presets_and_override(presets_arg: str, frame_stack_override):
    """Resolve cfg with given presets, optionally apply a CLI-style frame_stack override,
    then run the launcher's auto-apply step. Returns the final cam config."""
    import sys

    argv_extra = []
    if presets_arg:
        argv_extra.append(f"presets={presets_arg}")
    if frame_stack_override is not None:
        argv_extra.append(f"env.tiled_camera.frame_stack={frame_stack_override}")

    old_argv = sys.argv
    sys.argv = [sys.argv[0]] + argv_extra
    try:
        env_cfg, _ = resolve_task_config("Isaac-Cartpole-Camera-Presets-Direct-v0", "skrl_cfg_entry_point")
    finally:
        sys.argv = old_argv

    _apply_frame_stack_policies(env_cfg)
    return env_cfg.tiled_camera


# -- Helper unit tests for _resolve_frame_stack_policy ----------------------------------------


@pytest.mark.isaacsim_ci
def test_resolve_policy_int_passthrough():
    """A plain int policy value is returned as-is."""
    assert _resolve_frame_stack_policy(0) == 0
    assert _resolve_frame_stack_policy(1) == 1
    assert _resolve_frame_stack_policy(2) == 2


@pytest.mark.isaacsim_ci
def test_resolve_policy_wrapper_with_int_inner():
    """A wrapper with a by_renderer int unwraps to the int."""

    class _W:
        by_renderer = 2

    assert _resolve_frame_stack_policy(_W()) == 2


@pytest.mark.isaacsim_ci
def test_resolve_policy_wrapper_with_zero_inner():
    """A wrapper with by_renderer=0 unwraps to 0 (sentinel preserved)."""

    class _W:
        by_renderer = 0

    assert _resolve_frame_stack_policy(_W()) == 0


@pytest.mark.isaacsim_ci
def test_resolve_policy_unknown_structure():
    """An object without by_renderer or int returns None."""

    class _Unknown:
        pass

    assert _resolve_frame_stack_policy(_Unknown()) is None


# -- Truth table: presets only (no CLI override) ----------------------------------------------


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    "presets_arg,expected_frame_stack",
    [
        # No presets at all
        ("", 0),  # Sentinel, treated as 1 at runtime
        # PhysX physics: default RTX, Warp, OVRTX — all stay at sentinel
        ("physx", 0),
        ("physx,newton_renderer", 0),
        # Newton physics: depends on renderer
        ("newton", 0),  # Newton + default RTX (DLSS provides temporal): sentinel
        ("newton,newton_renderer", 2),  # Newton + Warp: ONLY combo that needs stacking
        ("newton,ovrtx_renderer", 0),  # Newton + OVRTX (assumed temporal): sentinel
        # Renderer preset alone (no physics preset): sentinel
        ("newton_renderer", 0),
        ("ovrtx_renderer", 0),
    ],
)
def test_frame_stack_policy_truth_table(presets_arg, expected_frame_stack):
    """Verify the AND-condition: frame_stack=2 only for newton+newton_renderer combo."""
    cam = _resolve_with_presets_and_override(presets_arg, frame_stack_override=None)
    assert cam.frame_stack == expected_frame_stack, (
        f"presets={presets_arg!r}: expected frame_stack={expected_frame_stack}, got {cam.frame_stack}"
    )


# -- CLI overrides: user's value always wins ---------------------------------------------------


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    "presets_arg,override,expected_frame_stack",
    [
        # No presets, with various overrides
        ("", 1, 1),  # Explicit 1 respected
        ("", 2, 2),
        ("", 4, 4),
        # PhysX with overrides (policy wouldn't fire anyway, but verify override is preserved)
        ("physx", 1, 1),
        ("physx", 4, 4),
        ("physx,newton_renderer", 2, 2),
        # Newton + RTX with overrides (policy says 0, override wins)
        ("newton", 1, 1),
        ("newton", 4, 4),
        # Newton + Warp with overrides (policy WOULD set 2, but override wins)
        ("newton,newton_renderer", 1, 1),  # User wants no stacking despite Newton+Warp
        ("newton,newton_renderer", 4, 4),
        # Newton + OVRTX with override
        ("newton,ovrtx_renderer", 3, 3),
    ],
)
def test_frame_stack_cli_override_respected(presets_arg, override, expected_frame_stack):
    """A non-zero CLI override on env.tiled_camera.frame_stack is always respected."""
    cam = _resolve_with_presets_and_override(presets_arg, frame_stack_override=override)
    assert cam.frame_stack == expected_frame_stack, (
        f"presets={presets_arg!r}, override={override}: expected {expected_frame_stack}, got {cam.frame_stack}"
    )


# -- Direct policy tree resolution (without auto-apply) ----------------------------------------


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    "presets_arg,expected_resolved",
    [
        # No physics preset: outer's default=0 fires, policy is int 0
        ("", 0),
        ("physx", 0),
        ("physx,newton_renderer", 0),
        ("newton_renderer", 0),
        # Newton physics: outer picks _FrameStackPolicyBranch, inner resolves
        ("newton", 0),  # No renderer preset: inner's default=0
        ("newton,newton_renderer", 2),  # Inner's newton_renderer=2
        ("newton,ovrtx_renderer", 0),  # Inner has no ovrtx_renderer field
    ],
)
def test_frame_stack_policy_resolves_correctly(presets_arg, expected_resolved):
    """The frame_stack_policy field itself resolves correctly via the preset system,
    independent of the auto-apply step."""
    import sys

    argv_extra = [f"presets={presets_arg}"] if presets_arg else []
    old_argv = sys.argv
    sys.argv = [sys.argv[0]] + argv_extra
    try:
        env_cfg, _ = resolve_task_config("Isaac-Cartpole-Camera-Presets-Direct-v0", "skrl_cfg_entry_point")
    finally:
        sys.argv = old_argv

    policy = env_cfg.tiled_camera.frame_stack_policy
    resolved = _resolve_frame_stack_policy(policy)
    assert resolved == expected_resolved, (
        f"presets={presets_arg!r}: expected resolved policy={expected_resolved}, got {resolved}"
    )


# -- Sentinel default at construction time -----------------------------------------------------


@pytest.mark.isaacsim_ci
def test_multibackend_camera_default_frame_stack_is_sentinel():
    """A freshly-constructed MultiBackendCameraCfg has frame_stack=0 (sentinel)."""
    from isaaclab_tasks.utils.presets import MultiBackendCameraCfg

    cam = MultiBackendCameraCfg()
    assert cam.frame_stack == 0, f"Expected sentinel default 0, got {cam.frame_stack}"


@pytest.mark.isaacsim_ci
def test_plain_cameracfg_default_frame_stack_unchanged():
    """A plain CameraCfg keeps its frame_stack=1 default (no auto-policy mechanism)."""
    cfg = CameraCfg()
    assert cfg.frame_stack == 1, f"Plain CameraCfg should default to 1, got {cfg.frame_stack}"


# -- Auto-apply behavior: respects existing values, applies only when frame_stack == 0 ---------


@pytest.mark.isaacsim_ci
def test_auto_apply_does_not_overwrite_user_value():
    """If frame_stack is already > 0 (user-set or already-applied), policy is not re-applied."""
    from isaaclab_tasks.utils.presets import MultiBackendCameraCfg

    cam = MultiBackendCameraCfg()
    cam.frame_stack = 5  # simulate user override / earlier application
    cam.frame_stack_policy = 2  # stubbed policy that would suggest 2
    _apply_frame_stack_policies(cam)
    assert cam.frame_stack == 5, "Auto-apply must not overwrite an existing non-zero frame_stack"


@pytest.mark.isaacsim_ci
def test_auto_apply_skips_when_policy_is_zero():
    """A policy resolved to 0 (sentinel) does not bump frame_stack."""
    from isaaclab_tasks.utils.presets import MultiBackendCameraCfg

    cam = MultiBackendCameraCfg()
    cam.frame_stack = 0
    cam.frame_stack_policy = 0  # int directly
    _apply_frame_stack_policies(cam)
    assert cam.frame_stack == 0, "Policy=0 should leave frame_stack at sentinel 0"


@pytest.mark.isaacsim_ci
def test_auto_apply_bumps_when_sentinel_and_policy_positive():
    """Auto-apply sets frame_stack from policy when frame_stack is the sentinel 0."""
    from isaaclab_tasks.utils.presets import MultiBackendCameraCfg

    cam = MultiBackendCameraCfg()
    cam.frame_stack = 0
    cam.frame_stack_policy = 3
    _apply_frame_stack_policies(cam)
    assert cam.frame_stack == 3, f"Auto-apply should set frame_stack=3, got {cam.frame_stack}"


@pytest.mark.isaacsim_ci
def test_auto_apply_handles_wrapper_policy():
    """Auto-apply correctly unwraps a _FrameStackPolicyBranch-style wrapper."""
    from isaaclab_tasks.utils.presets import MultiBackendCameraCfg, _FrameStackPolicyBranch

    cam = MultiBackendCameraCfg()
    cam.frame_stack = 0
    # Post-resolution shape: wrapper.by_renderer has been collapsed to an int by Hydra.
    branch = _FrameStackPolicyBranch()
    branch.by_renderer = 2
    cam.frame_stack_policy = branch
    _apply_frame_stack_policies(cam)
    assert cam.frame_stack == 2, f"Auto-apply should unwrap wrapper to 2, got {cam.frame_stack}"


# -- End-to-end: full bridge from policy through auto-apply to actual Camera buffer ----------


def _make_e2e_cfg():
    """Build a MultiBackendCameraCfg pointing at the test scene with a concrete RTX renderer."""
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_tasks.utils.presets import MultiBackendCameraCfg

    return MultiBackendCameraCfg(
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        offset=MultiBackendCameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        prim_path="/World/Origin_.*/CameraSensor",
        update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        renderer_cfg=IsaacRtxRendererCfg(),
    )


def _make_newton_warp_policy(value: int):
    """Construct a resolved policy wrapper with the given by_renderer value."""
    from isaaclab_tasks.utils.presets import _FrameStackPolicyBranch

    branch = _FrameStackPolicyBranch()
    branch.by_renderer = value
    return branch


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    "scenario,policy,frame_stack_pre,expected_channels",
    [
        # Case 1: policy is an int=0 (e.g., physx, or newton+RTX). Auto-apply does nothing,
        # camera treats sentinel 0 as 1, output has 3 channels.
        ("policy_inactive", 0, 0, 3),
        # Case 2: policy is a wrapper resolved to 2 (newton+newton_renderer). Auto-apply bumps
        # frame_stack to 2, camera produces 6 channels.
        ("policy_active", _make_newton_warp_policy, 0, 6),
        # Case 3: user overrode frame_stack via CLI to 4. Auto-apply must skip; camera produces
        # 12 channels regardless of what the policy would say.
        ("user_override", _make_newton_warp_policy, 4, 12),
    ],
)
def test_e2e_auto_apply_propagates_to_camera_output(setup_scene, scenario, policy, frame_stack_pre, expected_channels):
    """End-to-end bridge: cfg state -> _apply_frame_stack_policies -> real Camera() -> rendered
    output shape. Verifies that the auto-apply step's effect actually reaches the Camera buffer
    and that user CLI overrides are respected (auto-apply skips when frame_stack != 0)."""
    sim = setup_scene
    cfg = _make_e2e_cfg()
    cfg.frame_stack_policy = policy(2) if callable(policy) else policy
    cfg.frame_stack = frame_stack_pre

    _apply_frame_stack_policies(cfg)

    camera = Camera(cfg)
    sim.reset()
    camera.update(dt=0.01)
    rgb = camera.data.output["rgb"]
    assert rgb.shape == (NUM_CAMERAS, CAMERA_HEIGHT, CAMERA_WIDTH, expected_channels), (
        f"[{scenario}] expected {expected_channels} channels, got shape {rgb.shape}"
    )


@pytest.mark.isaacsim_ci
def test_physx_no_regression_output_shape(setup_scene):
    """PhysX with default frame_stack=1 should produce standard (N, H, W, 3) output."""
    sim = setup_scene
    camera = Camera(_make_camera_cfg(frame_stack=1))
    sim.reset()
    camera.update(dt=0.01)

    rgb = camera.data.output["rgb"]
    assert rgb.shape == (NUM_CAMERAS, CAMERA_HEIGHT, CAMERA_WIDTH, 3), (
        f"PhysX default should produce (N, H, W, 3), got {rgb.shape}"
    )
