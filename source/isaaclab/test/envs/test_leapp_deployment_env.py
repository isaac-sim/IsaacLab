# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import importlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("leapp")

from isaaclab.assets import BaseArticulation
from isaaclab.envs.leapp_deployment_env import ControllerOwnedWriteSpec, LeappDeploymentEnv, WriteOutputSpec
from isaaclab.utils.leapp import leapp_tensor_semantics

leapp_env_module = importlib.import_module("isaaclab.envs.leapp_deployment_env")


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


def test_init_applies_seed_and_events_in_standard_lifecycle_order(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Deployment initialization should mirror the manager-based event lifecycle."""
    calls = []

    class FakePhysicsManager:
        def set_decimation(self, decimation):
            calls.append(("set_decimation", decimation))

        def handles_decimation(self):
            return True

    class FakeSimulationContext:
        @staticmethod
        def instance():
            return None

        def __init__(self, _cfg):
            calls.append("simulation")
            self.device = "cpu"
            self.stage = object()
            self.physics_manager = FakePhysicsManager()
            self.has_gui = False

        def register_interactive_scene(self, _scene):
            calls.append("register_scene")

        def reset(self):
            calls.append("simulation_reset")

        def get_setting(self, _path):
            return False

        def clear_instance(self):
            calls.append("clear")

        def stop(self):
            calls.append("stop")

    class FakeScene(dict):
        def __init__(self, _cfg):
            super().__init__()
            calls.append("scene")

        def update(self, *, dt):
            calls.append(("scene_update", dt))

    class FakeEventManager:
        available_modes = {"prestartup", "startup"}

        def __init__(self, _cfg, _env):
            calls.append("event_manager")

        def apply(self, *, mode, **_kwargs):
            calls.append(mode)

    class FakeCommandManager:
        def __init__(self, _cfg, _env):
            calls.append("command_manager")

    class FakeInferenceManager:
        nodes = {}

        def __init__(self, _path):
            calls.append("inference")

    pipeline_path = tmp_path / "policy.yaml"
    pipeline_path.write_text("pipeline:\n  inputs: {}\n  outputs: {}\n", encoding="utf-8")
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=8),
        sim=SimpleNamespace(dt=0.01),
        seed=17,
        decimation=2,
        events=object(),
        commands=object(),
        ui_window_class_type=None,
        validate=lambda: calls.append("validate"),
    )
    monkeypatch.setattr(leapp_env_module, "SimulationContext", FakeSimulationContext)
    monkeypatch.setattr(leapp_env_module, "InteractiveScene", FakeScene)
    monkeypatch.setattr(leapp_env_module, "EventManager", FakeEventManager)
    monkeypatch.setattr(leapp_env_module, "CommandManager", FakeCommandManager)
    monkeypatch.setattr(leapp_env_module, "InferenceManager", FakeInferenceManager)
    monkeypatch.setattr(leapp_env_module, "use_stage", lambda _stage: contextlib.nullcontext())
    monkeypatch.setattr(
        LeappDeploymentEnv,
        "seed",
        staticmethod(lambda seed: calls.append(("seed", seed)) or seed),
    )

    env = LeappDeploymentEnv(cfg, str(pipeline_path))

    assert cfg.scene.num_envs == 1
    assert calls.index(("seed", 17)) < calls.index("scene")
    assert calls.index("register_scene") < calls.index("event_manager")
    assert calls.index("event_manager") < calls.index("prestartup") < calls.index("simulation_reset")
    assert calls.index("simulation_reset") < calls.index(("set_decimation", 2))
    assert calls.index(("set_decimation", 2)) < calls.index("command_manager") < calls.index("startup")
    assert calls.index("startup") < calls.index("inference")
    assert env._physics_handles_decimation
    env.close()


def test_reset_reseeds_and_resets_manager_state_in_order(monkeypatch: pytest.MonkeyPatch):
    """Reset should reseed first and reset command and event manager state."""
    calls = []
    env = object.__new__(LeappDeploymentEnv)
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=0.01),
        decimation=2,
        num_rerenders_on_reset=0,
        wait_for_textures=False,
    )
    env.sim = SimpleNamespace(
        device="cpu",
        forward=lambda: calls.append("forward"),
        render_context=SimpleNamespace(reset_scene_state_cadence=lambda: calls.append("render_cadence_reset")),
    )
    env.scene = SimpleNamespace(
        reset=lambda _env_ids: calls.append("scene_reset"),
        write_data_to_sim=lambda: calls.append("write"),
        update=lambda *, dt: calls.append(("update", dt)),
    )
    env.event_manager = SimpleNamespace(
        available_modes={"reset"},
        apply=lambda **kwargs: calls.append(("event_apply", kwargs)),
        reset=lambda _env_ids: calls.append("event_reset"),
    )
    env.command_manager = SimpleNamespace(reset=lambda _env_ids: calls.append("command_reset"))
    env.inference = SimpleNamespace(reset=lambda: calls.append("inference_reset"))
    env.has_rtx_sensors = False
    env._input_mapping = {}
    env._sim_step_counter = 8
    monkeypatch.setattr(env, "seed", lambda seed: calls.append(("seed", seed)) or seed)

    env.reset(seed=23)

    assert calls[:4] == [("seed", 23), "scene_reset", ("event_apply", calls[2][1]), "command_reset"]
    assert calls[2][1]["mode"] == "reset"
    assert calls[2][1]["global_env_step_count"] == 4
    assert calls.index("command_reset") < calls.index("event_reset") < calls.index("render_cadence_reset")
    assert calls.index("render_cadence_reset") < calls.index("write")


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

    disjoint_duplicate = dict(requirement)
    disjoint_duplicate["element_names"] = [["finger"]]
    env._leapp_desc["pipeline"]["configs"]["isaaclab"]["controller_owned_writes"]["requirements"] = [
        disjoint_duplicate,
        requirement,
    ]
    with pytest.raises(ValueError, match="duplicates source term"):
        env._resolve_controller_owned_writes()

    env._leapp_desc["pipeline"]["configs"]["isaaclab"]["controller_owned_writes"]["requirements"] = [requirement]
    requirement["element_names"] = [["missing_joint"]]
    with pytest.raises(ValueError, match="unknown joints"):
        env._resolve_controller_owned_writes()

    requirement["element_names"] = [["joint_a", "joint_b"]]
    env.cfg = SimpleNamespace(actions=SimpleNamespace())
    with pytest.raises(RuntimeError, match="unexpected"):
        env._resolve_controller_owned_writes()

    env.cfg = matching_cfg
    specs = env._resolve_controller_owned_writes()
    assert specs[0].joint_names == ("joint_a", "joint_b")

    requirement["element_names"] = [["joint_a"]]
    with pytest.raises(RuntimeError, match="mismatched"):
        env._resolve_controller_owned_writes()
    requirement["element_names"] = [["joint_a", "joint_b"]]

    env._leapp_desc = {"pipeline": {"configs": {}}}
    with pytest.raises(RuntimeError, match="Re-export the policy"):
        env._resolve_controller_owned_writes()


@pytest.mark.parametrize(
    ("requirement_update", "error_match"),
    [
        ({"element_names": None}, "name every controlled joint"),
        ({"element_names": [["joint_a", "joint_a"]]}, "duplicate joint names"),
        ({"kind": "target/joint/position"}, "gravity_compensation capability"),
        (
            {"isaaclab_connection": "write:robot:set_joint_position_target_index"},
            "gravity_compensation capability",
        ),
    ],
)
def test_controller_owned_write_rejects_ambiguous_gravity_requirements(requirement_update, error_match):
    """Gravity ownership must name one unique effort target and exact controlled joints."""
    robot = _AnnotatedArticulation(["joint_a"])
    requirement = {
        "capability": "gravity_compensation",
        "source_term": "arm_action",
        "kind": "target/joint/effort",
        "element_names": [["joint_a"]],
        "cadence": "action_apply",
        "isaaclab_connection": "write:robot:set_joint_effort_target_index",
    }
    requirement.update(requirement_update)
    env = object.__new__(LeappDeploymentEnv)
    env.scene = {"robot": robot}
    env._leapp_desc = {
        "pipeline": {
            "configs": {"isaaclab": {"controller_owned_writes": {"schema_version": 1, "requirements": [requirement]}}}
        }
    }
    env._controller_owned_write_handlers = {"gravity_compensation": lambda _env, _spec: None}

    with pytest.raises(ValueError, match=error_match):
        env._resolve_controller_owned_writes()


@pytest.mark.parametrize("schema_version", [None, True, 2])
def test_controller_owned_write_rejects_unsupported_schema_versions(schema_version):
    """The deployment parser must fail closed on missing or unsupported schemas."""
    env = object.__new__(LeappDeploymentEnv)
    env._leapp_desc = {
        "pipeline": {
            "configs": {
                "isaaclab": {
                    "controller_owned_writes": {
                        "schema_version": schema_version,
                        "requirements": [],
                    }
                }
            }
        }
    }

    with pytest.raises(ValueError, match="schema_version 1"):
        env._resolve_controller_owned_writes()


def test_controller_owned_write_accepts_legacy_dict_action_config_without_requirements():
    """Deployment should preserve ActionManager's support for dictionary action configs."""
    env = object.__new__(LeappDeploymentEnv)
    env.cfg = SimpleNamespace(actions={})
    env.scene = {}
    env._leapp_desc = {
        "pipeline": {"configs": {"isaaclab": {"controller_owned_writes": {"schema_version": 1, "requirements": []}}}}
    }
    env._controller_owned_write_handlers = {}

    assert env._resolve_controller_owned_writes() == ()


def test_controller_owned_write_rejects_altered_generic_kind_and_output_alias():
    """Generic controller requirements must match writer semantics and own equivalent aliases."""

    class CustomArticulation(_AnnotatedArticulation):
        @leapp_tensor_semantics(kind="target/joint/effort")
        def set_custom_effort(self, **_kwargs):
            pass

        @leapp_tensor_semantics(kind="target/joint/effort")
        def set_custom_effort_alias(self, **_kwargs):
            pass

    robot = CustomArticulation(["joint_a"])
    requirement = {
        "capability": "feedforward_effort",
        "source_term": "arm_action",
        "kind": "target/joint/position",
        "element_names": [["joint_a"]],
        "cadence": "action_apply",
        "isaaclab_connection": "write:robot:set_custom_effort",
    }
    env = object.__new__(LeappDeploymentEnv)
    env.cfg = SimpleNamespace(
        actions={
            "arm_action": SimpleNamespace(
                asset_name="robot",
                joint_names=["joint_a"],
                preserve_order=True,
                controller_owned_write_methods={"set_custom_effort": "feedforward_effort"},
            )
        }
    )
    env.scene = {"robot": robot}
    env._leapp_desc = {
        "pipeline": {
            "configs": {"isaaclab": {"controller_owned_writes": {"schema_version": 1, "requirements": [requirement]}}}
        }
    }
    env._controller_owned_write_handlers = {"feedforward_effort": lambda _env, _spec: None}

    with pytest.raises(ValueError, match="does not match runtime method"):
        env._resolve_controller_owned_writes()

    requirement["kind"] = "target/joint/effort"
    env._controller_owned_write_specs = env._resolve_controller_owned_writes()
    with pytest.raises(ValueError, match="does not match runtime method"):
        env._validate_policy_output_ownership(
            "model/effort",
            "write:robot:set_custom_effort_alias",
            "target/joint/position",
            [["joint_a"]],
        )
    with pytest.raises(ValueError, match="unknown joints"):
        env._validate_policy_output_ownership(
            "model/effort",
            "write:robot:set_custom_effort_alias",
            "target/joint/effort",
            [["missing_joint"]],
        )
    with pytest.raises(ValueError, match="overlaps controller-owned"):
        env._validate_policy_output_ownership(
            "model/effort",
            "write:robot:set_custom_effort_alias",
            "target/joint/effort",
            [["joint_a"]],
        )


def test_policy_output_cannot_alias_controller_owned_effort_through_another_writer():
    """Equivalent effort writers must not create a second owner for the same joints."""
    env = object.__new__(LeappDeploymentEnv)
    env.scene = {"robot": _AnnotatedArticulation(["joint_a"])}
    env._controller_owned_write_specs = (
        ControllerOwnedWriteSpec(
            capability="gravity_compensation",
            source_term="arm_action",
            kind="target/joint/effort",
            entity_name="robot",
            method_name="set_joint_effort_target_index",
            cadence="action_apply",
            joint_names=("joint_a",),
            joint_ids=(0,),
        ),
    )

    with pytest.raises(ValueError, match="overlaps controller-owned"):
        env._validate_policy_output_ownership(
            "model/effort",
            "write:robot:set_joint_effort_target_mask",
            "target/joint/effort",
            [["joint_a"]],
        )


def test_resolve_io_accepts_canonical_static_gain_output():
    """Static gain outputs should validate against the articulation API's canonical semantics."""
    assert BaseArticulation.write_joint_stiffness_to_sim_index._leapp_semantics.kind == "kp"
    assert BaseArticulation.write_joint_stiffness_to_sim_mask._leapp_semantics.kind == "kp"
    assert BaseArticulation.write_joint_damping_to_sim_index._leapp_semantics.kind == "kd"
    assert BaseArticulation.write_joint_damping_to_sim_mask._leapp_semantics.kind == "kd"

    robot = _AnnotatedArticulation(["joint_a"])
    env = object.__new__(LeappDeploymentEnv)
    env.scene = {"robot": robot}
    env.command_manager = None
    env._controller_owned_write_specs = ()
    env._input_mapping = {}
    env._output_mapping = {}
    env.inference = SimpleNamespace(
        nodes={
            "model": SimpleNamespace(
                input_descriptions=[],
                output_descriptions=[
                    {
                        "name": "arm_kp_gains",
                        "kind": "kp",
                        "element_names": [["joint_a"]],
                        "isaaclab_connection": "write:robot:write_joint_stiffness_to_sim_index",
                    }
                ],
            )
        }
    )
    env._leapp_desc = {"pipeline": {"inputs": {}, "outputs": {"model": ["arm_kp_gains"]}}}

    env._resolve_io()

    assert env._output_mapping["model/arm_kp_gains"] == WriteOutputSpec(
        entity_name="robot",
        method_name="write_joint_stiffness_to_sim_index",
        value_param="stiffness",
        joint_ids=None,
    )


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
