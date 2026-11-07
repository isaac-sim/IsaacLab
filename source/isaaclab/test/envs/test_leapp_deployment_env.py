# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("leapp")

from isaaclab.envs.leapp_deployment_env import ControllerOwnedWriteSpec, LeappDeploymentEnv
from isaaclab.utils.leapp import leapp_tensor_semantics


class _BaseCameraData:
    @property
    @leapp_tensor_semantics(input_transform=lambda image: image.float())
    def rgb(self):
        raise NotImplementedError


class _CameraData(_BaseCameraData):
    def __init__(self):
        self._rgb = SimpleNamespace(torch=torch.full((1, 2, 2, 3), 127, dtype=torch.uint8))

    @property
    def rgb(self):
        return self._rgb


class _AnnotatedArticulation:
    """Small articulation stub with the same LEAPP writer semantics as the runtime API."""

    def __init__(self, joint_names):
        self.joint_names = joint_names

    def find_joints(self, names, preserve_order):
        return [self.joint_names.index(name) for name in names], names

    @leapp_tensor_semantics(kind="target/joint/effort")
    def set_joint_effort_target_index(self, **_kwargs):
        pass

    @leapp_tensor_semantics(kind="target/joint/effort")
    def set_joint_effort_target_mask(self, **_kwargs):
        pass

    @leapp_tensor_semantics(kind="target/joint/position")
    def set_joint_position_target_index(self, **_kwargs):
        pass

    @leapp_tensor_semantics(kind="kp")
    def write_joint_stiffness_to_sim_index(self, *, stiffness, joint_ids=None):
        pass


def test_read_inputs_applies_inherited_input_transform():
    """Deployment should replay the transform used to declare an exported input."""
    env = object.__new__(LeappDeploymentEnv)
    env.scene = {"camera": SimpleNamespace(data=_CameraData())}
    env.inference = SimpleNamespace(
        nodes={
            "model": SimpleNamespace(
                input_descriptions=[
                    {
                        "name": "base_camera_rgb",
                        "isaaclab_connection": "state:camera:rgb",
                    }
                ],
                output_descriptions=[],
            )
        }
    )
    env._leapp_desc = {
        "pipeline": {
            "inputs": {"model": ["base_camera_rgb"]},
            "outputs": {},
        }
    }
    env._input_mapping = {}
    env._output_mapping = {}

    env._resolve_io()
    inputs = env._read_inputs()

    camera_input = inputs["model/base_camera_rgb"]
    assert camera_input.dtype == torch.float32
    torch.testing.assert_close(camera_input, torch.full((1, 2, 2, 3), 127.0))


def test_reset_handles_inference_tensor_state_created_during_step():
    """Deployment should reset persistent LEAPP state created during inference."""

    class StatefulInference:
        def __init__(self):
            self.state = torch.ones(1)

        def reset(self):
            self.state.zero_()

        def run_policy(self, _inputs):
            self.state = torch.ones(1)
            return {}

    env = object.__new__(LeappDeploymentEnv)
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=0.01, render_interval=1),
        decimation=1,
        num_rerenders_on_reset=0,
        wait_for_textures=False,
    )
    env.sim = SimpleNamespace(
        device="cpu",
        is_rendering=False,
        forward=lambda: None,
        step=lambda *, render: None,
        render_context=SimpleNamespace(reset_scene_state_cadence=lambda: None),
    )
    env.scene = SimpleNamespace(
        reset=lambda _env_ids: None,
        write_data_to_sim=lambda: None,
        update=lambda *, dt: None,
    )
    env.event_manager = None
    env.command_manager = None
    env.inference = StatefulInference()
    env.has_rtx_sensors = False
    env._input_mapping = {}
    env._output_mapping = {}
    env._controller_owned_write_handlers = {}
    env._controller_owned_write_specs = ()
    env._sim_step_counter = 0
    env._physics_handles_decimation = False

    env.reset()
    env.step()

    assert torch.is_inference(env.inference.state)
    env.reset()
    torch.testing.assert_close(env.inference.state, torch.zeros(1))


@pytest.mark.parametrize(("handles_decimation", "expected_steps"), [(False, 2), (True, 1)])
def test_step_honors_backend_decimation_and_controller_cadence(handles_decimation, expected_steps):
    """Controller writes should follow ActionManager cadence without double-stepping physics."""
    calls = []
    spec = ControllerOwnedWriteSpec(
        capability="test",
        source_term="arm",
        kind="target/joint/effort",
        entity_name="robot",
        method_name="effort",
        cadence="action_apply",
        joint_names=("joint",),
        joint_ids=(0,),
    )
    env = object.__new__(LeappDeploymentEnv)
    env.cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.01, render_interval=2), decimation=2)
    env.sim = SimpleNamespace(
        device="cpu",
        is_rendering=False,
        step=lambda *, render: calls.append("sim_step"),
    )
    env.scene = SimpleNamespace(
        write_data_to_sim=lambda: calls.append("write"),
        update=lambda *, dt: calls.append(("update", dt)),
    )
    env.command_manager = None
    env.event_manager = SimpleNamespace(
        available_modes={"interval"},
        apply=lambda **kwargs: calls.append(("event", kwargs)),
    )
    env.inference = SimpleNamespace(run_policy=lambda _inputs: {})
    env._input_mapping = {}
    env._output_mapping = {}
    env._controller_owned_write_specs = (spec,)
    env._controller_owned_write_handlers = {"test": lambda _env, _spec: calls.append("controller")}
    env._sim_step_counter = 0
    env._physics_handles_decimation = handles_decimation

    env.step()

    assert calls.count("sim_step") == expected_steps
    assert calls.count("controller") == expected_steps
    assert env._sim_step_counter == 2
    for index, call in enumerate(calls):
        if call == "write":
            assert calls[index - 1] == "controller"
    expected_dt = 0.02 if handles_decimation else 0.01
    assert [call for call in calls if isinstance(call, tuple) and call[0] == "update"] == [
        ("update", expected_dt)
    ] * expected_steps
    assert calls[-1] == ("event", {"mode": "interval", "dt": 0.02})


def test_controller_owned_write_requires_handler_and_reorders_joints():
    """Controller requirements should fail closed and resolve exported joint order."""
    robot = _AnnotatedArticulation(["joint_b", "finger", "joint_a"])
    requirement = {
        "capability": "gravity_compensation",
        "source_term": "arm_action",
        "kind": "target/joint/effort",
        "element_names": [["joint_a", "joint_b"]],
        "cadence": "action_apply",
        "isaaclab_connection": "write:robot:set_joint_effort_target_index",
    }
    env = object.__new__(LeappDeploymentEnv)
    env.scene = {"robot": robot}
    env._leapp_desc = {
        "pipeline": {
            "configs": {"isaaclab": {"controller_owned_writes": {"schema_version": 1, "requirements": [requirement]}}}
        }
    }
    matching_cfg = SimpleNamespace(
        actions=SimpleNamespace(
            arm_action=SimpleNamespace(
                asset_name="robot",
                joint_names=["joint_a", "joint_b"],
                preserve_order=True,
                controller_owned_write_methods={
                    "set_joint_effort_target_index": "gravity_compensation",
                },
            )
        )
    )
    env.cfg = matching_cfg
    env._controller_owned_write_handlers = {}

    with pytest.raises(RuntimeError, match="gravity_compensation"):
        env._resolve_controller_owned_writes()

    env._controller_owned_write_handlers = {"gravity_compensation": lambda _env, _spec: None}
    specs = env._resolve_controller_owned_writes()
    assert specs[0].joint_names == ("joint_a", "joint_b")
    assert specs[0].joint_ids == (2, 0)

    env._controller_owned_write_handlers = {"gravity_compensation": None}
    with pytest.raises(TypeError, match="must be callable"):
        env._resolve_controller_owned_writes()

    env._controller_owned_write_handlers = {"gravity_compensation": lambda _env, _spec: None}
    overlapping_requirement = dict(requirement)
    overlapping_requirement["source_term"] = "other_action"
    env._leapp_desc["pipeline"]["configs"]["isaaclab"]["controller_owned_writes"]["requirements"] = [
        requirement,
        overlapping_requirement,
    ]
    with pytest.raises(ValueError, match="overlaps an earlier"):
        env._resolve_controller_owned_writes()

    env._leapp_desc["pipeline"]["configs"]["isaaclab"]["controller_owned_writes"]["requirements"] = [requirement]
    requirement["element_names"] = [["missing_joint"]]
    with pytest.raises(ValueError, match="unknown joints"):
        env._resolve_controller_owned_writes()


def test_simulated_gravity_handler_offsets_base_dofs_and_sanitizes_nonfinite_values():
    """Simulation gravity support should select arm DoFs and never write non-finite effort."""
    writes = []
    gravity = torch.tensor([[10.0, 11.0, 1.0, float("nan"), 3.0]])
    robot = SimpleNamespace(
        num_base_dofs=2,
        num_joints=3,
        data=SimpleNamespace(gravity_compensation_forces=SimpleNamespace(torch=gravity)),
        set_joint_effort_target_index=lambda **kwargs: writes.append(kwargs),
    )
    env = SimpleNamespace(scene={"robot": robot})
    spec = ControllerOwnedWriteSpec(
        capability="gravity_compensation",
        source_term="arm_action",
        kind="target/joint/effort",
        entity_name="robot",
        method_name="set_joint_effort_target_index",
        cadence="action_apply",
        joint_names=("joint_2", "joint_1", "joint_0"),
        joint_ids=(2, 1, 0),
    )

    LeappDeploymentEnv.apply_simulated_gravity_compensation(env, spec)

    assert writes[0]["joint_ids"] == [2, 1, 0]
    torch.testing.assert_close(writes[0]["target"], torch.tensor([[3.0, 0.0, 1.0]]))
