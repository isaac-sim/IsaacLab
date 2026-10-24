# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Task-owned action validation and backend preparation shared by live and replay."""


def prepare_teleop_action(env, action, conditioner, validator, state: dict):
    """Validate and condition one task-owned action without stepping physics."""
    issue = validator(action) if validator is not None else None
    if issue is None and conditioner is not None:
        action = conditioner(env, action, state)
        issue = validator(action) if validator is not None else None
    return action, issue


def prepare_teleop_backend(env_cfg, physics_cfg) -> None:
    """Apply task-owned preparation for the resolved physics configuration."""
    from isaaclab_ov.physics import OvPhysxCfg

    backend = None
    if isinstance(physics_cfg, OvPhysxCfg):
        backend = "ovphysx"
    elif getattr(getattr(physics_cfg, "solver_cfg", None), "solver_type", None) == "mujoco_warp":
        backend = "newton_mjwarp"
    if backend is not None:
        stabilize = getattr(env_cfg, f"stabilize_for_kitless_{backend}", None)
        if callable(stabilize):
            stabilize()


def initial_hand_tracking_issue(teleop_interface, initialized_sides: set[str], *, required: bool = True) -> str | None:
    """Wait for initial hand presence without changing later tracking-loss behavior."""
    required_sides = {"left", "right"}
    if not required or required_sides.issubset(initialized_sides):
        return None
    lifecycle = teleop_interface._session_lifecycle
    result = lifecycle.last_step_result
    if result is None:
        return "hand-tracking presence is unavailable"
    hands = (result.get(lifecycle.HAND_LEFT_KEY), result.get(lifecycle.HAND_RIGHT_KEY))
    if any(hand is None for hand in hands):
        return "hand-tracking presence is unavailable"
    initialized_sides.update(side for side, hand in zip(("left", "right"), hands, strict=True) if not hand.is_none)
    missing = sorted(required_sides - initialized_sides)
    return f"waiting for initial {'/'.join(missing)} hand tracking" if missing else None
