# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration compatibility and calibrated input behavior without an XR session."""

import math
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab.utils import to_dict, validate

from isaaclab_tasks.utils import resolve_task_config

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.mark.parametrize("controller,input_preset", [(None, None), ("diffik", "keyboard"), ("diffik_abs", "xr")])
def test_task_presets_preserve_original_controllers_and_rl_configuration(controller, input_preset):
    import isaaclab_teleop_franka  # noqa: F401

    base_overrides = ["physics=newton_mjwarp"]
    if controller is not None:
        base_overrides.append(f"presets={controller}")
    overrides = base_overrides + ([f"presets={input_preset}"] if input_preset is not None else [])
    original, original_agent = resolve_task_config(
        "Isaac-Reach-Franka", "rsl_rl_cfg_entry_point", overrides=base_overrides
    )
    example, example_agent = resolve_task_config(
        "Isaac-Reach-Franka-Teleop", "rsl_rl_cfg_entry_point", overrides=overrides
    )
    validate(example)

    assert type(example.actions.arm_action) is type(original.actions.arm_action)
    assert (example.actions.arm_action.input is not None) == (input_preset is not None)
    assert (example.observations.operator is not None) == (input_preset == "xr")
    example_dict = to_dict(example)
    example_dict["actions"]["arm_action"]["input"] = None
    example_dict["observations"].pop("operator")
    assert example_dict == to_dict(original)
    assert to_dict(example_agent) == to_dict(original_agent)
    assert gym.spec("Isaac-Reach-Franka-Teleop").entry_point == gym.spec("Isaac-Reach-Franka").entry_point


def test_discovery_preserves_original_registration_without_loading_capture():
    """Installing the example must not rewrite another task or load optional native runtimes."""
    script = textwrap.dedent(
        """
        import copy
        import sys
        import gymnasium as gym
        import isaaclab_tasks
        from isaaclab.cli import load_external_tasks
        from isaaclab_tasks.utils import resolve_task_config

        original = copy.deepcopy(vars(gym.spec("Isaac-Reach-Franka")))
        load_external_tasks()
        for presets in ([], ["presets=diffik,keyboard"], ["presets=diffik_abs,xr"]):
            resolve_task_config("Isaac-Reach-Franka-Teleop", None, overrides=presets)
        assert vars(gym.spec("Isaac-Reach-Franka")) == original
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in ("isaaccapture", "isaacteleop", "ovxr", "ovrtx", "ovstage")
        )
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_xr_presentation_requires_the_absolute_controller_input():
    import isaaclab_teleop_franka  # noqa: F401

    cfg, _ = resolve_task_config("Isaac-Reach-Franka-Teleop", None, overrides=["presets=diffik,xr"])
    with pytest.raises(ValueError, match="requires presets=diffik_abs,xr"):
        validate(cfg)


@pytest.fixture
def capture_pipeline():
    pytest.importorskip("isaaccapture", minversion="1.7.93a1")
    from isaaccapture.retargeting_engine.deviceio_source_nodes import ControllersSource
    from isaaccapture.retargeting_engine.interface import ComputeContext, ExecutionEvents, ExecutionState, TensorGroup
    from isaaccapture.schema import ControllerPose, ControllerSnapshot, Point, Pose, Quaternion
    from isaaclab_teleop_franka.xr import controller_pose_pipeline

    pipeline = controller_pose_pipeline()
    anchor = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float32)

    def compute(
        position=(0.1, 0.2, 0.3), orientation=(0, 0, 0, 1), *, valid=True, present=True, running=True, reset=False
    ):
        pose = ControllerPose(Pose(Point(*position), Quaternion(*orientation)), valid)
        snapshot = ControllerSnapshot(grip_pose=pose, aim_pose=pose) if present else None
        inputs = {}
        for leaf in pipeline.get_leaf_nodes():
            values = {name: TensorGroup(spec) for name, spec in leaf.input_spec().items()}
            if isinstance(leaf, ControllersSource):
                values["deviceio_controller_left"][0] = None
                values["deviceio_controller_right"][0] = snapshot
            else:
                values["value"][0] = anchor
            inputs[leaf.name] = values
        events = ExecutionEvents(
            reset=reset, execution_state=ExecutionState.RUNNING if running else ExecutionState.PAUSED
        )
        result = pipeline.execute_pipeline(inputs, ComputeContext(execution_events=events))
        return np.from_dlpack(result["action"][0]).copy()

    return compute


def test_capture_pipeline_transforms_valid_poses_and_preserves_recalibration_boundaries(capture_pipeline):
    """The actual Capture graph emits world poses and epochs even if consumers skip invalid frames."""
    compute = capture_pipeline
    first = compute(orientation=(0, 0, 0, 2))
    half_sqrt = math.sqrt(0.5)
    np.testing.assert_allclose(first[:8], [0.8, 2.1, 3.3, 0, 0, half_sqrt, half_sqrt, 1], atol=1e-6)
    assert compute()[8] == first[8]

    # Each condition reaches a distinct SDK or pose-validation branch.
    invalid_samples = (
        {"present": False},
        {"valid": False},
        {"orientation": (0, 0, 0, 0)},
        {"position": (math.nan, 0, 0)},
        {"running": False},
    )
    previous = first
    for kwargs in invalid_samples:
        invalid = compute(**kwargs)
        assert invalid[7] == 0 and np.isfinite(invalid).all()
        recovered = compute(position=(0.5, 0.2, 0.3))
        assert recovered[7] == 1
        assert recovered[8] != previous[8]
        assert compute()[8] == recovered[8]
        previous = recovered

    reset = compute(reset=True)
    assert reset[7] == 1 and reset[8] != previous[8]
    assert compute()[8] == reset[8]


def test_absolute_input_tracks_one_to_one_holds_targets_and_recalibrates():
    """World-frame motion maps into the original normalized action independently of robot lag."""
    pytest.importorskip("isaaccapture", minversion="1.7.93a1")
    from isaaclab_teleop.source import TeleopSample
    from isaaclab_teleop_franka.xr import CalibratedAbsolutePoseInput

    cfg, _ = resolve_task_config("Isaac-Reach-Franka-Teleop", None, overrides=["presets=diffik_abs,xr"])
    action_cfg = cfg.actions.arm_action
    half_sqrt = math.sqrt(0.5)
    # The robot base is translated and turned +90 degrees about world Z.
    root_quat = torch.tensor([[0, 0, half_sqrt, half_sqrt]])
    body_position = torch.tensor([[[1.0, 2.5, 0.3]]])
    asset = SimpleNamespace(
        find_bodies=lambda name: ([0], ["panda_hand"]),
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=torch.tensor([[1.0, 2.0, 0.0]])),
            root_quat_w=SimpleNamespace(torch=root_quat),
            body_pos_w=SimpleNamespace(torch=body_position),
            body_quat_w=SimpleNamespace(torch=root_quat[:, None]),
        ),
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={"robot": asset},
        action_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(cfg=action_cfg)),
    )
    term = CalibratedAbsolutePoseInput(action_cfg.input, env, action_dim=7)
    policy = torch.full((1, 7), 0.25)

    def resolve(position=(2.0, 3.0, 1.0), orientation=(0.0, 0.0, 0.0, 1.0), *, epoch=1, valid=True, active=True):
        term.set_sample(
            TeleopSample(torch.tensor([*position, *orientation, float(valid), float(epoch)]), active=active)
        )
        return term.resolve_actions(policy)

    # Engagement maps to the measured pose, not the absolute tracking origin.
    expected = torch.tensor([[0.0, 0.0, -1.0 / 7.0, 0.0, 0.0, 0.0, 1.0]])
    torch.testing.assert_close(resolve(), expected, atol=1e-6, rtol=1e-5)
    # +20 cm world Y is +20 cm base X; world X rotation is negative base Y.
    moved = resolve(position=(2.0, 3.2, 1.0), orientation=(half_sqrt, 0.0, 0.0, half_sqrt))
    expected[0] = torch.tensor([4.0 / 3.0, 0.0, -1.0 / 7.0, 0.0, -half_sqrt, 0.0, half_sqrt])
    torch.testing.assert_close(moved, expected, atol=1e-6, rtol=1e-5)
    body_position[0, 0, 1] = 2.6
    torch.testing.assert_close(
        resolve(position=(2.0, 3.2, 1.0), orientation=(half_sqrt, 0.0, 0.0, half_sqrt)),
        moved,
    )
    torch.testing.assert_close(policy, torch.full((1, 7), 0.25))

    # Every lifecycle boundary establishes a new measured reference, even with
    # the same tracked pose or when the simulation missed the invalid frame.
    epoch = 1
    for boundary in ("pause", "tracking", "reset", "epoch"):
        body_position[0, 0, 1] += 0.1
        if boundary == "pause":
            torch.testing.assert_close(resolve(active=False, epoch=epoch), policy)
        elif boundary == "tracking":
            torch.testing.assert_close(resolve(valid=False, epoch=epoch), policy)
        elif boundary == "reset":
            term.reset([0])
            torch.testing.assert_close(term.resolve_actions(policy), policy)
        else:
            epoch += 1
        result = resolve(position=(5.0, 8.0, 2.0), epoch=epoch)
        # Check physical position after the original controller's normalization.
        commanded_x = result[0, 0] * 0.15 + 0.5
        torch.testing.assert_close(commanded_x, body_position[0, 0, 1] - 2.0)
        torch.testing.assert_close(result[0, 3:], torch.tensor([0.0, 0.0, 0.0, 1.0]))
