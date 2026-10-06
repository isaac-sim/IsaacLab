# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Kitless real-Newton articulation coverage on one persistent composite scene.

Every articulation is a small locally authored fixture (see ``articulation_test_utils``). One module-scoped
scene holds isolated two-environment islands, one per fixture configuration, and is reset once. Each test drives
the islands it names and targets partial writes at environment 1 so it can prove environment 0 keeps its real
backend state. The islands of one environment share a solver world, so every test starts with all islands at
rest and restores the properties it changes. Scene gravity is off; a test that needs gravity applies it to every
world for its own duration.

Configurations that fail initialization, the rebind scenario, which swaps the live Newton state, and the
CUDA-graph check build their own small scenes. Only one simulation context can be alive, so these tests come
first and fail if selected after a composite-scene test.

The composite scene runs on the CPU without CUDA-graph capture. On CUDA, one own-scene test covers the
ordered-state republish recorded into a captured graph.
"""

from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices

launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

import logging
import sys
from collections.abc import Callable, Iterator
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import isaaclab_newton.physics.newton_manager as newton_manager_module
import newton
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.assets import Articulation
from isaaclab_newton.assets.articulation import kernels as articulation_kernels
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.physics import NewtonManager as SimulationManager
from isaaclab_physx.sim.schemas import PhysxJointCfg
from newton import JointType, Model, ModelBuilder, ModelFlags, ShapeFlags, State
from newton.selection import ArticulationView
from newton_test_utils import (
    NUM_ENVS,
    WRIST_USD_STIFFNESS,
    env_origins,
    local_usd,
    newton_sim_cfg,
    spawn_assets,
    world_gravity,
)

from pxr import UsdPhysics

import isaaclab.assets.articulation.ordering_resolvers as ordering_resolvers
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ActuatorBaseCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.assets.articulation.ordering_resolvers import get_articulation_name_ordering
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.envs.mdp import randomize_physics_scene_gravity
from isaaclab.envs.mdp.events import randomize_rigid_body_collider_offsets, randomize_rigid_body_material
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sim import SimulationContext, build_simulation_context
from isaaclab.test.utils.articulation_ordering import (
    BRANCHING_MJWARP_BODY_NAMES,
    BRANCHING_MJWARP_JOINT_NAMES,
    BRANCHING_PHYSX_BODY_NAMES,
    BRANCHING_PHYSX_JOINT_NAMES,
)
from isaaclab.utils import replace

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

_GRAVITY = (0.0, 0.0, -9.81)

_NEWTON_USER_ORDER_STATE_CACHES = (
    "_joint_pos_user",
    "_joint_vel_user",
    "_body_link_pose_w_user",
    "_body_com_vel_w_user",
)

_USD_LIMIT_PROPS = [sim_utils.UsdPhysicsDriveCfg(max_force=80.0), PhysxJointCfg(max_joint_velocity=5.0)]
"""Solver clamps authored on the limit islands' joints [N·m, rad/s]."""

_LEG_PUBLIC_JOINT_NAMES = ("LF_KFE", "RH_HAA", "LF_HAA", "RH_KFE", "LF_HFE", "RH_HFE")
_LEG_PUBLIC_BODY_NAMES = ("base", "RH_THIGH", "LF_SHANK", "RH_HIP", "LF_HIP", "RH_SHANK", "LF_THIGH")
"""Public orders that permute ``floating_two_leg.usda``'s backend joints and non-root bodies.

Neither permutation is its own inverse, so a map applied in the wrong direction shows.
"""


def _limit_actuators(
    actuator_type: type[ImplicitActuatorCfg] | type[IdealPDActuatorCfg],
) -> dict[str, ActuatorBaseCfg]:
    """Return one limit-resolution group per branching-fixture joint.

    Mirrors the single-joint implicit and explicit rows of the solver-clamp resolution contract: joint limits
    only, actuator limits only, both, and neither. Implicit actuators take no separate rated effort limit.
    """
    if actuator_type is ImplicitActuatorCfg:
        gains = {"stiffness": 10.0, "damping": 1.0}
        actuator_limits = {"actuator_velocity_limit": 1e2}
    else:
        gains = {"stiffness": 0.0, "damping": 0.1}
        actuator_limits = {"actuator_velocity_limit": 1e2, "actuator_effort_limit": 1e2}
    joint_limits = {"joint_velocity_limit": 1e5, "joint_effort_limit": 1e5}
    return {
        "joint_limit": actuator_type(joint_names_expr=["left_shoulder"], **joint_limits, **gains),
        "actuator_limit": actuator_type(joint_names_expr=["left_elbow"], **actuator_limits, **gains),
        "both_limits": actuator_type(joint_names_expr=["right_shoulder"], **joint_limits, **actuator_limits, **gains),
        "no_limit": actuator_type(joint_names_expr=["right_elbow"], **gains),
    }


def _island_cfgs() -> dict[str, ArticulationCfg]:
    """Return the composite scene's islands, each at its own offset inside every environment."""

    def island(
        name: str,
        spawn: sim_utils.UsdFileCfg,
        y: float,
        actuators: dict[str, ActuatorBaseCfg],
        init_state: ArticulationCfg.InitialStateCfg | None = None,
        **kwargs,
    ) -> ArticulationCfg:
        """Return the island ``name`` at lateral offset ``y`` [m] inside every environment."""
        init_state = replace(init_state or ArticulationCfg.InitialStateCfg(), pos=(0.0, y, 1.0))
        return ArticulationCfg(
            prim_path=f"/World/Env_[^/]*/{name}", spawn=spawn, init_state=init_state, actuators=actuators, **kwargs
        )

    return {
        # The floating island comes first: its root owns the first body and joint coordinates of each world.
        "floating": island(
            "Floating",
            local_usd("floating_two_link.usda"),
            0.0,
            {"joint": ImplicitActuatorCfg(joint_names_expr=["Joint"], stiffness=20.0, damping=2.0)},
            articulation_root_prim_path="/Root",
        ),
        "fixed": island(
            "Fixed",
            local_usd("fixed_spatial_chain.usda"),
            2.0,
            {
                "slides": ImplicitActuatorCfg(
                    joint_names_expr=["Joint_[0-2]"], stiffness=2000.0, damping=100.0, viscous_friction=0.25
                ),
                "wrist": ImplicitActuatorCfg(joint_names_expr=["Joint_[3-5]"], stiffness=None, damping=None),
            },
        ),
        "pendulum": island(
            "Pendulum",
            local_usd("revolute_pendulum.usda"),
            6.0,
            {"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
        ),
        "ordered": island(
            "Ordered",
            local_usd("floating_two_leg.usda"),
            8.0,
            {"legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=1.0)},
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*HAA": 0.2, ".*HFE": 0.3, ".*KFE": -0.6}),
            joint_ordering=_LEG_PUBLIC_JOINT_NAMES,
            body_ordering=_LEG_PUBLIC_BODY_NAMES,
        ),
        "branching": island(
            "Branching",
            local_usd("articulation_ordering_branching.usda"),
            20.0,
            {"arms": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=1.0)},
            joint_ordering="physx",
            body_ordering="physx",
        ),
        "tendon": island(
            "Hand",
            local_usd("fixed_tendon_hand.usda"),
            10.0,
            {
                # the tendons drive these joints: gains stay as authored
                "tendon_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*J(1|2)"], stiffness=None, damping=None, armature=1e-3
                )
            },
        ),
        "root_fixed": island(
            "RootFixed",
            local_usd("floating_two_link.usda", fix_root_link=True),
            12.0,
            {"joint": ImplicitActuatorCfg(joint_names_expr=["Joint"], stiffness=20.0, damping=2.0)},
        ),
        "root_freed": island(
            "RootFreed",
            local_usd("fixed_spatial_chain.usda", fix_root_link=False),
            14.0,
            {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1.0, damping=0.1)},
        ),
        "limits_implicit": island(
            "LimitsImplicit",
            local_usd("articulation_ordering_branching.usda", joint_drive_props=_USD_LIMIT_PROPS),
            16.0,
            _limit_actuators(ImplicitActuatorCfg),
        ),
        "limits_explicit": island(
            "LimitsExplicit",
            local_usd("articulation_ordering_branching.usda", joint_drive_props=_USD_LIMIT_PROPS),
            18.0,
            _limit_actuators(IdealPDActuatorCfg),
        ),
        "osc": island(
            "Osc",
            local_usd("fixed_spatial_chain.usda"),
            22.0,
            # unpowered, so the operational-space controller's efforts are the only actuation
            {"arm": ImplicitActuatorCfg(joint_names_expr=["Joint_.*"], stiffness=0.0, damping=0.0)},
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"Joint_3": 0.3, "Joint_4": -0.2, "Joint_5": 0.4}),
        ),
    }


@dataclass
class _Scene:
    """Articulations that share one real Newton model and solver lifecycle."""

    sim: SimulationContext
    articulations: dict[str, Articulation]
    cfgs: dict[str, ArticulationCfg]
    origins: torch.Tensor
    device: str

    def step(self, *names: str, num_steps: int = 1) -> None:
        """Write the named islands' commands, step every island, and update the named islands."""
        for _ in range(num_steps):
            for name in names:
                self.articulations[name].write_data_to_sim()
            self.sim.step()
            for name in names:
                self.articulations[name].update(self.sim.cfg.dt)

    def default_root_pose_w(self, articulation: Articulation) -> torch.Tensor:
        """Return the configured root pose placed at each environment origin."""
        root_pose = articulation.data.default_root_pose.torch.clone()
        root_pose[:, :3] += self.origins
        return root_pose

    def rest(self, articulation: Articulation) -> None:
        """Put an island at rest at its default configuration and hold it there.

        The targets reach the solver immediately, so the island stays at rest while other tests step the scene.
        """
        if not articulation.is_fixed_base:
            articulation.write_root_pose_to_sim_index(root_pose=self.default_root_pose_w(articulation))
            articulation.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros_like(articulation.data.default_root_vel.torch)
            )
        default_joint_pos = articulation.data.default_joint_pos.torch.clone()
        articulation.write_joint_state_to_sim_index(
            position=default_joint_pos, velocity=torch.zeros_like(default_joint_pos)
        )
        commands = articulation.actuators.target_command
        commands.set_position_index(value=default_joint_pos)
        commands.set_velocity_index(value=torch.zeros_like(default_joint_pos))
        commands.set_effort_index(value=torch.zeros_like(default_joint_pos))
        articulation.reset()
        articulation.write_data_to_sim()


##
# Kernels and model-level accessors. These build no simulation context.
##


def _selector(values: list[int], dtype: type) -> wp.array:
    """Create a CPU Warp selector with the requested integer width."""
    return wp.array(values, dtype=dtype, device="cpu")


@pytest.mark.parametrize(("env_dtype", "joint_dtype"), [(wp.int32, wp.int32), (wp.int64, wp.int64)])
def test_write_joint_limit_data_to_user_and_backend_index_accepts_index_dtypes(
    env_dtype: type, joint_dtype: type
) -> None:
    """Write partial user-order joint limits into user and backend-order buffers."""
    limits_np = np.asarray([[[1.0, 3.0], [2.0, 5.0]], [[-1.0, 1.0], [4.0, 8.0]]], dtype=np.float32)
    limits = wp.array(limits_np, dtype=wp.vec2f, device="cpu")
    env_ids = _selector([0, 1], env_dtype)
    user_ids = _selector([2, 0], joint_dtype)
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_lower = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    user_upper = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    user_limits = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
    backend_lower = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    backend_upper = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    soft_limits = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
    default_pos = wp.array(
        np.asarray([[0.0, 0.0, 4.0], [0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.float32, device="cpu"
    )
    clamped_defaults = wp.zeros(1, dtype=wp.int32, device="cpu")
    kernel = articulation_kernels.write_joint_limit_data_to_user_and_backend_index
    if env_dtype != wp.int32 or joint_dtype != wp.int32:
        kernel = articulation_kernels.write_joint_limit_data_to_user_and_backend_index_kernel(env_ids, user_ids)

    wp.launch(
        kernel,
        dim=limits.shape,
        inputs=[limits, 1.0, env_ids, user_ids, user_to_backend, True],
        outputs=[
            user_lower,
            user_upper,
            user_limits,
            backend_lower,
            backend_upper,
            soft_limits,
            default_pos,
            clamped_defaults,
        ],
        device="cpu",
    )

    np.testing.assert_allclose(user_lower.numpy(), np.asarray([[2.0, 0.0, 1.0], [4.0, 0.0, -1.0]], dtype=np.float32))
    np.testing.assert_allclose(user_upper.numpy(), np.asarray([[5.0, 0.0, 3.0], [8.0, 0.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(backend_lower.numpy(), np.asarray([[1.0, 2.0, 0.0], [-1.0, 4.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(backend_upper.numpy(), np.asarray([[3.0, 5.0, 0.0], [1.0, 8.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(
        user_limits.numpy(),
        np.asarray([[[2.0, 5.0], [0.0, 0.0], [1.0, 3.0]], [[4.0, 8.0], [0.0, 0.0], [-1.0, 1.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(default_pos.numpy(), np.asarray([[2.0, 0.0, 3.0], [4.0, 0.0, 0.0]], dtype=np.float32))
    assert clamped_defaults.numpy()[0] == 3


def test_write_joint_limit_data_to_user_and_backend_mask_reorders_backend_buffers() -> None:
    """Write masked user-order joint limits into user and backend-order buffers."""
    limits_np = np.asarray(
        [[[1.0, 2.0], [3.0, 6.0], [4.0, 9.0]], [[-2.0, 2.0], [5.0, 7.0], [8.0, 10.0]]], dtype=np.float32
    )
    limits = wp.array(limits_np, dtype=wp.vec2f, device="cpu")
    env_mask = wp.array(np.asarray([True, False], dtype=bool), dtype=wp.bool, device="cpu")
    user_mask = wp.array(np.asarray([False, True, True], dtype=bool), dtype=wp.bool, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_lower = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    user_upper = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    user_limits = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
    backend_lower = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    backend_upper = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    soft_limits = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
    default_pos = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    clamped_defaults = wp.zeros(1, dtype=wp.int32, device="cpu")

    wp.launch(
        articulation_kernels.write_joint_limit_data_to_user_and_backend_mask,
        dim=limits.shape,
        inputs=[limits, 1.0, env_mask, user_mask, user_to_backend, True],
        outputs=[
            user_lower,
            user_upper,
            user_limits,
            backend_lower,
            backend_upper,
            soft_limits,
            default_pos,
            clamped_defaults,
        ],
        device="cpu",
    )

    np.testing.assert_allclose(user_lower.numpy(), np.asarray([[0.0, 3.0, 4.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(user_upper.numpy(), np.asarray([[0.0, 6.0, 9.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(backend_lower.numpy(), np.asarray([[4.0, 0.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(backend_upper.numpy(), np.asarray([[9.0, 0.0, 6.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(
        user_limits.numpy(),
        np.asarray([[[0.0, 0.0], [3.0, 6.0], [4.0, 9.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(default_pos.numpy(), np.asarray([[0.0, 3.0, 4.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    assert clamped_defaults.numpy()[0] == 2


@pytest.mark.parametrize("index_dtype", [wp.int32, wp.int64])
def test_scatter_reset_masks_from_ids_accepts_index_dtype(index_dtype: type) -> None:
    """Set exact world and articulation reset masks from nonidentity environment IDs."""

    env_ids = _selector([2, 0], index_dtype)
    articulation_ids = wp.array(np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int32), dtype=int, device="cpu")
    world_mask = wp.zeros(3, dtype=wp.bool, device="cpu")
    fk_mask = wp.zeros(6, dtype=wp.bool, device="cpu")

    wp.launch(
        newton_manager_module._scatter_reset_masks_from_ids,
        dim=(env_ids.shape[0], articulation_ids.shape[1]),
        inputs=[env_ids, articulation_ids],
        outputs=[world_mask, fk_mask],
        device="cpu",
    )

    np.testing.assert_array_equal(world_mask.numpy(), np.asarray([True, False, True]))
    np.testing.assert_array_equal(fk_mask.numpy(), np.asarray([True, True, False, False, True, True]))


def test_num_shapes_per_body_follows_public_body_order() -> None:
    """Align Newton shape counts with the public body-name axis."""

    class _ShapeCountSurface:
        backend_num_shapes_per_body = Articulation.backend_num_shapes_per_body
        num_shapes_per_body = Articulation.num_shapes_per_body

    articulation = _ShapeCountSurface()
    articulation._num_shapes_per_body_backend = None
    articulation._root_view = SimpleNamespace(
        body_shapes=((), (object(), object()), (object(), object(), object())),
    )
    articulation.body_ordering = SimpleNamespace(
        user_to_backend_indices=(2, 0, 1),
    )

    assert articulation.num_shapes_per_body == [3, 0, 2]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize(
    "first_property",
    ["body_com_jacobian_w", "body_link_jacobian_w", "mass_matrix", "gravity_compensation_forces"],
)
def test_task_space_allocation_and_capture(monkeypatch, device, first_property):
    """Allocate only requested outputs and retain them across changed-state graph replay."""
    builder = ModelBuilder()
    builder.begin_world()
    base = builder.add_link(mass=4.0, inertia=wp.mat33(np.eye(3)), label="Robot/base")
    arm = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)), com=(0.5, 0.0, 0.0), label="Robot/arm")
    slider = builder.add_link(mass=3.0, inertia=wp.mat33(np.eye(3)), com=(0.25, 0.0, 0.0), label="Robot/slider")
    builder.add_articulation(
        [
            builder.add_joint_fixed(-1, base),
            builder.add_joint_revolute(base, arm, axis=(0.0, 1.0, 0.0), label="hinge"),
            builder.add_joint_prismatic(arm, slider, axis=(1.0, 0.0, 0.0), label="slide"),
        ],
        label="Robot",
    )
    builder.end_world()
    model = builder.finalize(device=device)
    state, control = model.state(), model.control()
    monkeypatch.setattr(SimulationManager, "get_model", lambda: model)
    monkeypatch.setattr(SimulationManager, "get_state_0", lambda: state)
    monkeypatch.setattr(SimulationManager, "get_control", lambda: control)
    view = ArticulationView(model, "Robot", exclude_joint_types=[JointType.FIXED])
    eager = ArticulationData(view, device)
    eager._apply_ordering_maps_after_resolve()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    getattr(eager, first_property)  # Compile kernels before first-use capture on a fresh container.
    data = ArticulationData(view, device)
    data._apply_ordering_maps_after_resolve()
    properties = ("body_com_jacobian_w", "body_link_jacobian_w", "mass_matrix", "gravity_compensation_forces")
    assert all(getattr(data, f"_{name}_ta") is None for name in properties)
    assert data._jacobian_buf_flat is data._mass_matrix_full_buf is data._gravity_force_full_buf is None
    with wp.ScopedCapture(device=device) as capture:
        output = getattr(data, first_property)
    required = {first_property}
    if first_property == "body_link_jacobian_w":
        required.add("body_com_jacobian_w")
    for name in properties:
        assert (getattr(data, f"_{name}_ta") is not None) == (name in required)
    assert (data._jacobian_buf_flat is not None) == (first_property != "gravity_compensation_forces")
    assert (data._mass_matrix_full_buf is not None) == (first_property == "mass_matrix")
    assert (data._gravity_force_full_buf is not None) == (first_property == "gravity_compensation_forces")

    for angle, displacement in ((0.0, 0.0), (0.7, 0.4), (-0.3, 0.2)):
        state.joint_q.assign(np.asarray([angle, displacement], dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        expected = getattr(eager, first_property).warp.numpy()
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(output.warp.numpy(), expected, atol=1e-5)
    data._create_simulation_bindings()
    data._apply_ordering_maps_after_resolve()
    assert getattr(data, first_property) is output


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_world_hinged_root_has_no_base_dofs(monkeypatch, device):
    """A root link hinged to the world adds no floating-base DoF columns to the Jacobian or mass matrix."""
    builder = ModelBuilder()
    builder.begin_world()
    base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="Robot/base")
    link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="Robot/link")
    builder.add_articulation(
        [
            builder.add_joint_revolute(-1, base, axis=(0.0, 1.0, 0.0), label="root"),
            builder.add_joint_revolute(base, link, axis=(0.0, 1.0, 0.0), label="elbow"),
        ],
        label="Robot",
    )
    builder.end_world()
    model = builder.finalize(device=device)
    state, control = model.state(), model.control()
    monkeypatch.setattr(SimulationManager, "get_model", lambda: model)
    monkeypatch.setattr(SimulationManager, "get_state_0", lambda: state)
    monkeypatch.setattr(SimulationManager, "get_control", lambda: control)
    view = ArticulationView(model, "Robot", exclude_joint_types=[JointType.FREE, JointType.FIXED])
    data = ArticulationData(view, device)
    data._apply_ordering_maps_after_resolve()

    assert data.body_link_jacobian_w.torch.shape[-1] == view.joint_dof_count == 2
    assert data.mass_matrix.torch.shape[-1] == 2


##
# Own scenes. These tests run before the composite scene exists: only one simulation context can be alive.
##


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
@pytest.mark.parametrize("invalid_field", ["joint_pos", "joint_vel", "articulation_root_prim_path"])
def test_invalid_articulation_cfg_fails_initialization(device: str, invalid_field: str) -> None:
    """Initialization fails when a default joint state exceeds the joint limits or the explicit root does not exist."""
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        spawn=local_usd("fixed_spatial_chain.usda", joint_drive_props=[PhysxJointCfg(max_joint_velocity=5.0)]),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=10.0)},
    )
    if invalid_field == "joint_pos":
        articulation_cfg.init_state.joint_pos = {"Joint_0": 10.0, "Joint_[35]": -20.0}
        error, match = ValueError, "default positions out of the limits"
    elif invalid_field == "joint_vel":
        articulation_cfg.init_state.joint_vel = {"Joint_0": 100.0, "Joint_[35]": -60.0}
        error, match = ValueError, "default velocities out of the limits"
    else:
        articulation_cfg.articulation_root_prim_path = "/non_existing_prim_path"
        error, match = KeyError, "No articulations matching pattern"
    with build_simulation_context(sim_cfg=newton_sim_cfg(device)) as sim:
        articulation = spawn_assets({"robot": articulation_cfg}, num_envs=1)["robot"]

        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(articulation) < 10

        with pytest.raises(error, match=match):
            sim.reset()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_newton_rebind_refreshes_ordered_state_and_preserves_lab_owned_actuator_gains(device: str) -> None:
    """Rebind public state to recreated Newton arrays, invalidate ordered caches, and keep Lab-owned gains.

    A full sim reset recreates the solver's state and model arrays; the scenario shallow-copies the live state and
    model, swaps in sentinel-filled arrays, and rebinds every island's data.

    * Implicit islands, floating with and without non-identity ordering and fixed with it: every binding, public
      proxy, component view, and actuator kernel input follows the new arrays, and the acceleration caches restart
      from the new velocities.
    * Explicit islands: named actuator groups own their actuator kp/kd while the solver's sim gains are zeroed for
      explicit DOFs. Rebind must NOT resync the actuator-owned values from the freshly rebuilt (sentinel) solver
      gains, while the sim-owned mirrors track them. The identity-ordering island is the control that passes with
      or without the fix.

    Finally ``_clear_callbacks`` deregisters exactly the articulation's post-step hook, without touching other
    callbacks.
    """
    leg_cfg = local_usd("floating_two_leg.usda")
    explicit_legs = IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=40.0, damping=5.0, actuator_effort_limit=80.0)
    implicit_legs = ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=40.0, damping=5.0)
    orderings = {"joint_ordering": _LEG_PUBLIC_JOINT_NAMES, "body_ordering": _LEG_PUBLIC_BODY_NAMES}
    cfgs = {
        f"{actuation}_{ordering}": ArticulationCfg(
            prim_path=f"/World/Env_[^/]*/Robot_{actuation}_{ordering}",
            spawn=leg_cfg,
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 2.0 * index, 1.0)),
            actuators={"legs": implicit_legs if actuation == "implicit" else explicit_legs},
            **(orderings if ordering == "reordered" else {}),
        )
        for index, (actuation, ordering) in enumerate(
            (actuation, ordering) for actuation in ("implicit", "explicit") for ordering in ("none", "reordered")
        )
    }
    # a fixed base publishes no root velocity binding of its own
    cfgs["implicit_fixed_reordered"] = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot_implicit_fixed_reordered",
        spawn=local_usd("fixed_spatial_chain.usda"),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 8.0, 1.0)),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=40.0, damping=5.0)},
        joint_ordering=("Joint_2", "Joint_5", "Joint_0", "Joint_4", "Joint_1", "Joint_3"),
        body_ordering=("Root", "Link_3", "Link_0", "Link_5", "Link_1", "Link_4", "Link_2"),
    )
    with build_simulation_context(sim_cfg=newton_sim_cfg(device, use_newton_actuators=False)) as sim:
        articulations = spawn_assets(cfgs)
        sim.reset()
        sim.step()
        for articulation in articulations.values():
            assert articulation.is_initialized
            articulation.update(sim.cfg.dt)

        implicit_checks = [
            _prime_ordered_state_rebind(articulations[name], sim.cfg.dt)
            for name in ("implicit_none", "implicit_reordered", "implicit_fixed_reordered")
        ]
        for name in ("explicit_none", "explicit_reordered"):
            # Prime: explicit (IdealPD) actuators keep their PD in actuator-owned records,
            # while the solver's sim gains are zeroed so it applies no PD on these DOFs.
            articulation = articulations[name]
            data = articulation.data
            assert (data.joint_ordering is not None) is name.endswith("reordered")
            np.testing.assert_allclose(articulation.actuators["legs"].stiffness.cpu().numpy(), 40.0)
            np.testing.assert_allclose(articulation.actuators["legs"].damping.cpu().numpy(), 5.0)
            np.testing.assert_allclose(data._sim_bind_joint_stiffness_sim.numpy(), 0.0)
            np.testing.assert_allclose(data._sim_bind_joint_damping_sim.numpy(), 0.0)

        # Simulate a full sim reset: shallow-copy the state and the model, swap their arrays for sentinel-filled
        # arrays (standing in for whatever the solver rebuilds), and rebind the data-side sim bindings.
        old_state = SimulationManager.get_state_0()
        old_model = SimulationManager.get_model()
        new_state = copy(old_state)
        new_model = copy(old_model)

        joint_q_values = np.arange(len(old_state.joint_q), dtype=np.float32) + 1000.0
        joint_qd_values = np.arange(len(old_state.joint_qd), dtype=np.float32) + 2000.0
        body_indices = np.arange(len(old_state.body_q), dtype=np.float32)[:, None]
        body_q_values = np.zeros((len(old_state.body_q), 7), dtype=np.float32)
        body_q_values[:, :3] = 3000.0 + 10.0 * body_indices + np.arange(3, dtype=np.float32)
        body_q_values[:, 6] = 1.0
        body_qd_values = 4000.0 + 10.0 * body_indices + np.arange(6, dtype=np.float32)
        limit_indices = np.arange(len(old_model.joint_limit_lower), dtype=np.float32)
        sentinel_ke = 12345.0
        sentinel_kd = 678.0
        device = old_state.joint_q.device
        new_state.joint_q = wp.array(joint_q_values, dtype=wp.float32, device=device)
        new_state.joint_qd = wp.array(joint_qd_values, dtype=wp.float32, device=device)
        new_state.body_q = wp.array(body_q_values, dtype=wp.transformf, device=device)
        new_state.body_qd = wp.array(body_qd_values, dtype=wp.spatial_vectorf, device=device)
        new_model.joint_limit_lower = wp.array(-5000.0 - limit_indices, dtype=wp.float32, device=device)
        new_model.joint_limit_upper = wp.array(5000.0 + limit_indices, dtype=wp.float32, device=device)
        num_dofs = len(old_model.joint_target_ke)
        new_model.joint_target_ke = wp.array(np.full(num_dofs, sentinel_ke, dtype=np.float32), device=device)
        new_model.joint_target_kd = wp.array(np.full(num_dofs, sentinel_kd, dtype=np.float32), device=device)
        SimulationManager.backend.state_0 = new_state
        SimulationManager.backend.model = new_model

        for check_rebind in implicit_checks:
            check_rebind(new_state, new_model)

        for name in ("explicit_none", "explicit_reordered"):
            articulation = articulations[name]
            data = articulation.data
            data._create_simulation_bindings()
            # The actuator-owned gains must survive the rebind unchanged...
            np.testing.assert_allclose(articulation.actuators["legs"].stiffness.cpu().numpy(), 40.0)
            np.testing.assert_allclose(articulation.actuators["legs"].damping.cpu().numpy(), 5.0)
            # ...while the sim-owned mirrors track the solver's freshly seeded (sentinel) gains.
            if data.joint_ordering is not None:
                np.testing.assert_allclose(data._joint_stiffness_user.numpy(), sentinel_ke)
                np.testing.assert_allclose(data._joint_damping_user.numpy(), sentinel_kd)
            else:
                np.testing.assert_allclose(data._sim_bind_joint_stiffness_sim.numpy(), sentinel_ke)
                np.testing.assert_allclose(data._sim_bind_joint_damping_sim.numpy(), sentinel_kd)

        # ``_clear_callbacks`` must deregister exactly this hook so it does not leak on the class-level list,
        # and leave an unrelated callback (standing in for another articulation's hook) untouched.
        articulation = articulations["implicit_reordered"]
        registered_callback = articulation._post_step_callback
        assert registered_callback is not None
        assert registered_callback in SimulationManager._post_step_callbacks

        def _other_callback() -> None:
            return None

        SimulationManager.register_post_step_callback(_other_callback)
        articulation._clear_callbacks()

        assert articulation._post_step_callback is None
        assert registered_callback not in SimulationManager._post_step_callbacks
        assert _other_callback in SimulationManager._post_step_callbacks
        SimulationManager.unregister_post_step_callback(_other_callback)


def _prime_ordered_state_rebind(articulation: Articulation, dt: float) -> Callable[[State, Model], None]:
    """Prime an implicit island's ordered state caches and return the check to run after the arrays are swapped."""
    data = articulation.data
    previous_body_com_vel = data._previous_body_com_vel.numpy().copy()
    has_ordering = data.joint_ordering is not None
    assert (data.body_ordering is not None) is has_ordering
    primed_joint_vel_values = (
        np.arange(np.prod(data._sim_bind_joint_vel.shape), dtype=np.float32).reshape(data._sim_bind_joint_vel.shape)
        + 100.0
    )
    body_velocity_shape = (*data._sim_bind_body_com_vel_w.shape, 6)
    primed_body_com_vel_values = (
        np.arange(np.prod(body_velocity_shape), dtype=np.float32).reshape(body_velocity_shape) + 200.0
    )
    data._sim_bind_joint_vel.assign(
        wp.array(primed_joint_vel_values, dtype=wp.float32, device=data._sim_bind_joint_vel.device)
    )
    data._sim_bind_body_com_vel_w.assign(
        wp.array(primed_body_com_vel_values, dtype=wp.spatial_vectorf, device=data._sim_bind_body_com_vel_w.device)
    )
    # The raw sim-bind writes above simulate the solver advancing state; in the
    # real pipeline the post-step callback republishes the passthrough shadows in
    # the same step. Mirror that here so ``joint_acc`` (which reads the passthrough
    # ``joint_vel`` shadow) observes the primed backend state. No-op under identity
    # ordering, where the getters alias the sim-bound arrays directly.
    data._refresh_user_order_state()
    # Newton decimation may collect several physics steps into one data update.
    update_dt = 2 * dt
    data.update(update_dt)
    primed_joint_acc = data.joint_acc.warp.numpy().copy()
    primed_body_com_acc_w = data.body_com_acc_w.warp.numpy().copy()
    assert data._joint_acc.timestamp == data._sim_timestamp
    assert data._body_com_acc_w.timestamp == data._sim_timestamp
    assert np.any(primed_joint_acc != 0.0)
    assert np.any(primed_body_com_acc_w != 0.0)
    body_user_to_backend = (
        np.asarray(articulation.body_ordering.user_to_backend_indices)
        if articulation.body_ordering is not None
        else np.arange(articulation.num_bodies)
    )
    np.testing.assert_allclose(
        primed_body_com_acc_w,
        ((primed_body_com_vel_values - previous_body_com_vel) / update_dt)[:, body_user_to_backend],
        rtol=1e-5,
    )

    public_to_binding = {
        "joint_pos": "_sim_bind_joint_pos",
        "joint_vel": "_sim_bind_joint_vel",
        "body_link_pose_w": "_sim_bind_body_link_pose_w",
        "body_com_vel_w": "_sim_bind_body_com_vel_w",
    }
    public_to_shadow = {
        "joint_pos": "_joint_pos_user",
        "joint_vel": "_joint_vel_user",
        "body_link_pose_w": "_body_link_pose_w_user",
        "body_com_vel_w": "_body_com_vel_w_user",
    }
    old_bindings = {name: getattr(data, name) for name in public_to_binding.values()}
    old_binding_ptrs = {name: int(array.ptr) for name, array in old_bindings.items()}
    old_public_proxies = {name: getattr(data, name) for name in public_to_binding}
    component_sources = {
        "root_link_pos_w": ("root_link_pose_w", slice(None, 3)),
        "root_link_quat_w": ("root_link_pose_w", slice(3, None)),
        "body_com_lin_vel_w": ("body_com_vel_w", slice(None, 3)),
        "body_com_ang_vel_w": ("body_com_vel_w", slice(3, None)),
    }
    old_components = {name: getattr(data, name) for name in component_sources}
    implicit_executor = articulation.actuators._implicit_executor
    assert implicit_executor is not None
    actuator_state_inputs = [implicit_executor.kernel_inputs]
    data.joint_pos_limits.torch.clone()
    # The Tier-1 state shadows are plain wp.arrays (no timestamp): they are
    # allocated for non-identity ordering and stay ``None`` for identity ordering.
    for cache_name in _NEWTON_USER_ORDER_STATE_CACHES:
        assert (getattr(data, cache_name) is not None) is has_ordering

    def check_rebind(new_state: State, new_model: Model) -> None:
        body_velocities = articulation.root_view.get_link_velocities(new_state)
        assert body_velocities is not None
        new_source_bindings = {
            "_sim_bind_joint_pos": articulation.root_view.get_dof_positions(new_state)[:, 0],
            "_sim_bind_joint_vel": articulation.root_view.get_dof_velocities(new_state)[:, 0],
            "_sim_bind_body_link_pose_w": articulation.root_view.get_link_transforms(new_state)[:, 0],
            "_sim_bind_body_com_vel_w": body_velocities[:, 0],
            "_sim_bind_joint_pos_limits_lower": articulation.root_view.get_attribute("joint_limit_lower", new_model)[
                :, 0
            ],
            "_sim_bind_joint_pos_limits_upper": articulation.root_view.get_attribute("joint_limit_upper", new_model)[
                :, 0
            ],
        }
        for binding_name, new_source in new_source_bindings.items():
            if binding_name in old_binding_ptrs:
                assert int(new_source.ptr) != old_binding_ptrs[binding_name]

        data._create_simulation_bindings()

        for binding_name, old_binding in old_bindings.items():
            rebound = getattr(data, binding_name)
            assert rebound is not old_binding
            assert int(rebound.ptr) != old_binding_ptrs[binding_name]
            assert int(rebound.ptr) == int(new_source_bindings[binding_name].ptr)

        for inputs in actuator_state_inputs:
            assert inputs[3].ptr == data.joint_pos.warp.ptr
            assert inputs[4].ptr == data.joint_vel.warp.ptr

        assert data._joint_acc.timestamp == -1.0
        assert data._body_com_acc_w.timestamp == -1.0

        joint_user_to_backend = (
            np.asarray(articulation.joint_ordering.user_to_backend_indices)
            if articulation.joint_ordering is not None
            else np.arange(articulation.num_joints)
        )
        body_user_to_backend = (
            np.asarray(articulation.body_ordering.user_to_backend_indices)
            if articulation.body_ordering is not None
            else np.arange(articulation.num_bodies)
        )
        expected_previous_joint_vel = new_source_bindings["_sim_bind_joint_vel"].numpy()[:, joint_user_to_backend]
        expected_previous_body_com_vel = new_source_bindings["_sim_bind_body_com_vel_w"].numpy()
        np.testing.assert_array_equal(data._previous_joint_vel.numpy(), expected_previous_joint_vel)
        np.testing.assert_array_equal(data._previous_body_com_vel.numpy(), expected_previous_body_com_vel)

        joint_acc = data.joint_acc.warp.numpy()
        body_com_acc_w = data.body_com_acc_w.warp.numpy()
        np.testing.assert_array_equal(joint_acc, np.zeros_like(expected_previous_joint_vel))
        np.testing.assert_array_equal(
            body_com_acc_w, np.zeros_like(expected_previous_body_com_vel[:, body_user_to_backend])
        )
        assert data._joint_acc.timestamp == data._sim_timestamp
        assert data._body_com_acc_w.timestamp == data._sim_timestamp
        assert not np.array_equal(joint_acc, primed_joint_acc)
        assert not np.array_equal(body_com_acc_w, primed_body_com_acc_w)

        expected_public = {
            "joint_pos": new_source_bindings["_sim_bind_joint_pos"].numpy()[:, joint_user_to_backend],
            "joint_vel": new_source_bindings["_sim_bind_joint_vel"].numpy()[:, joint_user_to_backend],
            "body_link_pose_w": new_source_bindings["_sim_bind_body_link_pose_w"].numpy()[:, body_user_to_backend],
            "body_com_vel_w": new_source_bindings["_sim_bind_body_com_vel_w"].numpy()[:, body_user_to_backend],
        }
        for property_name, expected in expected_public.items():
            proxy = getattr(data, property_name)
            assert proxy is not old_public_proxies[property_name]
            binding_name = public_to_binding[property_name]
            assert int(proxy.warp.ptr) != old_binding_ptrs[binding_name]
            if has_ordering:
                shadow = getattr(data, public_to_shadow[property_name])
                assert int(proxy.warp.ptr) == int(shadow.ptr)
            else:
                assert int(proxy.warp.ptr) == int(getattr(data, binding_name).ptr)
            np.testing.assert_array_equal(proxy.warp.numpy(), expected)

        for name, (parent, selection) in component_sources.items():
            component = getattr(data, name)
            expected = getattr(data, parent).torch[..., selection]
            assert component is not old_components[name]
            assert component.torch.data_ptr() == expected.data_ptr()
            assert component.torch.stride() == expected.stride()
            torch.testing.assert_close(component.torch, expected)

        expected_limits = np.stack(
            (
                new_source_bindings["_sim_bind_joint_pos_limits_lower"].numpy()[:, joint_user_to_backend],
                new_source_bindings["_sim_bind_joint_pos_limits_upper"].numpy()[:, joint_user_to_backend],
            ),
            axis=-1,
        )
        np.testing.assert_array_equal(data.joint_pos_limits.warp.numpy(), expected_limits)

    return check_rebind


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_newton_ordered_state_publishes_inside_captured_cuda_graph(device: str) -> None:
    """Republish the user-order state shadows from inside a captured CUDA graph.

    With CUDA-graph capture a step replays the recorded launches, so the post-step reorder of the Tier-1 shadows
    must be recorded into the graph rather than run from Python. The shadows are clobbered after the graph exists
    and must equal the reordered backend state after one replayed step, without a state property read in between.
    """
    articulation_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        spawn=local_usd("floating_two_leg.usda"),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        actuators={"legs": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=40.0, damping=5.0)},
        joint_ordering=_LEG_PUBLIC_JOINT_NAMES,
        body_ordering=_LEG_PUBLIC_BODY_NAMES,
    )
    sim_cfg = newton_sim_cfg(device, use_newton_actuators=False, use_cuda_graph=True)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        articulation = spawn_assets({"robot": articulation_cfg})["robot"]
        sim.reset()
        # the first step captures the graph that later steps replay
        sim.step()
        assert SimulationManager._graph is not None
        data = articulation.data
        joint_u2b = np.asarray(articulation.joint_ordering.user_to_backend_indices)
        body_u2b = np.asarray(articulation.body_ordering.user_to_backend_indices)

        # a sentinel far from the resting backend state, so a stale shadow cannot pass for a republished one
        data._joint_pos_user.fill_(1000.0)
        data._joint_vel_user.fill_(1000.0)
        data._body_link_pose_w_user.fill_(wp.transformf(1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 1.0))
        data._body_com_vel_w_user.fill_(wp.spatial_vectorf(1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0))
        sim.step()

        np.testing.assert_allclose(data._joint_pos_user.numpy(), data._sim_bind_joint_pos.numpy()[:, joint_u2b])
        np.testing.assert_allclose(data._joint_vel_user.numpy(), data._sim_bind_joint_vel.numpy()[:, joint_u2b])
        np.testing.assert_allclose(
            data._body_link_pose_w_user.numpy(), data._sim_bind_body_link_pose_w.numpy()[:, body_u2b]
        )
        np.testing.assert_allclose(
            data._body_com_vel_w_user.numpy(), data._sim_bind_body_com_vel_w.numpy()[:, body_u2b]
        )


##
# Composite scene.
##


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CPU))
def composite_scene(request: pytest.FixtureRequest) -> Iterator[_Scene]:
    """Initialize every articulation island once for this module."""
    device = request.param
    sim_cfg = newton_sim_cfg(device, use_newton_actuators=False, newton_contacts=True)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        cfgs = _island_cfgs()
        articulations = spawn_assets(cfgs)
        sim.reset()
        yield _Scene(sim=sim, articulations=articulations, cfgs=cfgs, origins=env_origins(device), device=device)


@pytest.fixture
def scene(composite_scene: _Scene) -> _Scene:
    """Hand each test the composite scene with every island at rest.

    Islands of one environment share a solver world, so a state one test leaves behind must not reach the next.
    """
    for articulation in composite_scene.articulations.values():
        composite_scene.rest(articulation)
    return composite_scene


def _model_attribute(articulation: Articulation, name: str) -> torch.Tensor:
    """Read one live Newton model attribute through the articulation's view, in backend order."""
    return wp.to_torch(articulation.root_view.get_attribute(name, SimulationManager.get_model()))[:, 0]


def test_articulation_initialization_and_partial_state(scene: _Scene) -> None:
    """Initialize every island and prove partial joint-state writes against live backend state.

    Covers fixed and floating roots, a root fixed and a root freed by spawner fragments, the per-island buffer
    shapes and actuator models, a partial write with unsorted int64 selectors on a reordered articulation, body
    state refresh after joint writes without a step, and joint position limit writes that keep or clamp the
    default joint positions.
    """
    device = scene.device
    expected_fixed_base = {
        "floating": False,
        "fixed": True,
        "pendulum": True,
        "ordered": False,
        "tendon": True,
        "root_fixed": True,
        "root_freed": False,
        "limits_implicit": False,
        "limits_explicit": False,
        "branching": False,
        "osc": True,
    }
    for name, articulation in scene.articulations.items():
        # Check that the framework doesn't hold excessive strong references.
        assert sys.getrefcount(articulation) < 10, name
        assert articulation.is_initialized, name
        assert articulation.is_fixed_base == expected_fixed_base[name], name
        assert articulation.num_instances == 2
        assert articulation.data.root_pos_w.torch.shape == (2, 3)
        assert articulation.data.root_quat_w.torch.shape == (2, 4)
        assert articulation.data.joint_pos.torch.shape == (2, articulation.num_joints)
        assert articulation.data.body_mass.torch.shape == (2, articulation.num_bodies)
        assert articulation.data.body_inertia.torch.shape == (2, articulation.num_bodies, 9)
        for actuator_name, actuator in articulation.actuators.items():
            is_implicit_model_cfg = isinstance(scene.cfgs[name].actuators[actuator_name], ImplicitActuatorCfg)
            assert actuator.is_implicit_model == is_implicit_model_cfg

    floating = scene.articulations["floating"]
    assert floating.num_bodies == 2
    assert floating.num_joints == 1
    assert floating.joint_names == ["Joint"]
    assert floating.data.body_com_pos_b.shape == (2, 2)
    assert scene.articulations["tendon"].num_fixed_tendons > 0

    # Newton consumes the base manager's world joint without relocating the root API.
    root = sim_utils.get_first_matching_child_prim(
        "/World/Env_0/RootFixed", lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI), stage=scene.sim.stage
    )
    assert root is not None and root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert sim_utils.find_global_fixed_joint_prim("/World/Env_0/RootFixed", stage=scene.sim.stage) is not None

    # A partial write to environment 1 leaves environment 0 untouched.
    scene.rest(floating)
    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    joint_ids = torch.tensor([0], dtype=torch.int32, device=device)
    initial_joint_pos = floating.data.joint_pos.torch.clone()
    initial_joint_vel = floating.data.joint_vel.torch.clone()
    target_joint_pos = torch.tensor([[0.25]], device=device)
    target_joint_vel = torch.tensor([[-0.5]], device=device)
    floating.write_joint_state_to_sim_index(
        position=target_joint_pos, velocity=target_joint_vel, env_ids=env_ids, joint_ids=joint_ids
    )
    torch.testing.assert_close(floating.data.joint_pos.torch[env_ids], target_joint_pos)
    torch.testing.assert_close(floating.data.joint_vel.torch[env_ids], target_joint_vel)
    torch.testing.assert_close(floating.data.joint_pos.torch[:1], initial_joint_pos[:1])
    torch.testing.assert_close(floating.data.joint_vel.torch[:1], initial_joint_vel[:1])

    # The same contract on a reordered articulation, with unsorted int64 selectors.
    articulation = scene.articulations["ordered"]
    scene.rest(articulation)
    num_articulations = articulation.num_instances
    default_joint_pos = articulation.data.default_joint_pos.torch.clone()
    authored_limits = articulation.data.joint_pos_limits.torch.clone()

    limits = torch.stack(
        (
            -5.0 - torch.rand(num_articulations, articulation.num_joints, device=device),
            5.0 + torch.rand(num_articulations, articulation.num_joints, device=device),
        ),
        dim=-1,
    )
    articulation.write_joint_position_limit_to_sim_index(limits=limits)

    # Check new limits are in place and the in-limit defaults are preserved
    torch.testing.assert_close(articulation.data.joint_pos_limits.torch, limits)
    torch.testing.assert_close(articulation.data.default_joint_pos.torch, default_joint_pos)

    # Write joint state with unsorted int64 selectors
    env_ids = torch.tensor([1, 0], dtype=torch.int64, device=device)
    joint_ids = torch.tensor([articulation.num_joints - 1, 0], dtype=torch.int64, device=device)
    position = torch.tensor([[0.21, 0.11], [0.22, 0.12]], device=device)
    velocity = torch.tensor([[1.21, 1.11], [1.22, 1.12]], device=device)
    expected_position = articulation.data.joint_pos.torch.clone()
    expected_velocity = articulation.data.joint_vel.torch.clone()
    articulation.write_joint_state_to_sim_index(
        position=position, velocity=velocity, env_ids=env_ids, joint_ids=joint_ids
    )
    expected_position[env_ids[:, None], joint_ids[None, :]] = position
    expected_velocity[env_ids[:, None], joint_ids[None, :]] = velocity
    torch.testing.assert_close(articulation.data.joint_pos.torch, expected_position)
    torch.testing.assert_close(articulation.data.joint_vel.torch, expected_velocity)

    # A joint state write moves the bodies and refreshes the derived body state without a step.
    joint_pos_limits = articulation.data.joint_pos_limits.torch
    joint_vel_limits = articulation.data.joint_vel_limits.torch.clamp(max=10.0)
    pos_dist = torch.distributions.Uniform(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    vel_dist = torch.distributions.Uniform(-joint_vel_limits, joint_vel_limits)
    original_body_link_pose_w = articulation.data.body_link_pose_w.torch.clone()
    original_body_com_vel_w = articulation.data.body_com_vel_w.torch.clone()
    articulation.write_joint_position_to_sim_index(position=pos_dist.sample())
    articulation.write_joint_velocity_to_sim_index(velocity=vel_dist.sample())
    body_link_pose_w = articulation.data.body_link_pose_w.torch
    body_com_vel_w = articulation.data.body_com_vel_w.torch
    original_body_states = torch.cat([original_body_link_pose_w, original_body_com_vel_w], dim=-1)
    body_state_w = torch.cat([body_link_pose_w, body_com_vel_w], dim=-1)
    assert torch.count_nonzero(original_body_states[:, 1:] != body_state_w[:, 1:]) > (
        len(original_body_states[:, 1:]) / 2
    )
    # skip lin_vel because it differs from link frame, this should be fine because we are only checking
    # if velocity update is triggered, which can be determined by comparing angular velocity
    body_link_vel_w = articulation.data.body_link_vel_w.torch
    torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])
    # validate link - com consistency
    body_com_pos_b = articulation.data.body_com_pos_b.torch
    body_com_quat_b = articulation.data.body_com_quat_b.torch
    expected_com_pos, expected_com_quat = math_utils.combine_frame_transforms(
        body_link_pose_w[..., :3].view(-1, 3),
        body_link_pose_w[..., 3:].view(-1, 4),
        body_com_pos_b.view(-1, 3),
        body_com_quat_b.view(-1, 4),
    )
    torch.testing.assert_close(expected_com_pos.view(num_articulations, -1, 3), articulation.data.body_com_pos_w.torch)
    torch.testing.assert_close(
        expected_com_quat.view(num_articulations, -1, 4), articulation.data.body_com_quat_w.torch
    )

    # Set new joint limits with indexing that invalidate the selected default joint positions
    env_ids = torch.arange(1, device=device, dtype=torch.int32)
    joint_ids = torch.nonzero(default_joint_pos[0].abs() > 0.1).squeeze(-1)[:2].to(torch.int32)
    assert len(joint_ids) == 2
    limits = torch.stack(
        (
            -0.1 * torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device),
            0.1 * torch.rand(env_ids.shape[0], joint_ids.shape[0], device=device),
        ),
        dim=-1,
    )
    articulation.write_joint_position_limit_to_sim_index(limits=limits, env_ids=env_ids, joint_ids=joint_ids)

    # Check new limits are in place and the defaults are clamped into them
    torch.testing.assert_close(articulation.data.joint_pos_limits.torch[env_ids][:, joint_ids], limits)
    default_joint_pos_torch = articulation.data.default_joint_pos.torch
    within_bounds = (default_joint_pos_torch[env_ids][:, joint_ids] >= limits[..., 0]) & (
        default_joint_pos_torch[env_ids][:, joint_ids] <= limits[..., 1]
    )
    assert torch.all(within_bounds)

    # Islands share their environment's solver world, so leave the sampled states and narrow limits behind.
    articulation.write_joint_position_limit_to_sim_index(limits=authored_limits)
    scene.rest(articulation)
    scene.rest(floating)


def test_articulation_joint_and_body_properties_round_trip(scene: _Scene, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove joint and body property writes reach the live Newton model and notify the solver.

    Covers partial friction and inertial writes with their exact model-change notifications, per-environment
    static friction rows in the model, passive viscous friction kept apart from the actuator's derivative gain
    through both writer selectors, and actuator gains taken from the configuration or, when unset, from the
    USD drives.
    """
    device = scene.device
    articulation = scene.articulations["floating"]
    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    body_ids = torch.tensor([1], dtype=torch.int32, device=device)
    joint_ids = torch.tensor([0], dtype=torch.int32, device=device)
    notifications = []
    add_model_change = SimulationManager.add_model_change

    def record_model_change(change: ModelFlags) -> None:
        notifications.append(change)
        add_model_change(change)

    monkeypatch.setattr(SimulationManager, "add_model_change", staticmethod(record_model_change))
    initial_friction = articulation.data.joint_friction_coeff.torch.clone()
    friction = torch.tensor([[0.3]], device=device)
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=friction, env_ids=env_ids, joint_ids=joint_ids
    )
    torch.testing.assert_close(articulation.data.joint_friction_coeff.torch[env_ids][:, joint_ids], friction)
    torch.testing.assert_close(articulation.data.joint_friction_coeff.torch[:1], initial_friction[:1])
    torch.testing.assert_close(_model_attribute(articulation, "joint_friction")[env_ids][:, joint_ids], friction)
    torch.testing.assert_close(_model_attribute(articulation, "joint_friction")[:1], initial_friction[:1])
    assert notifications == [ModelFlags.JOINT_DOF_PROPERTIES]

    notifications.clear()
    initial_mass = articulation.data.body_mass.torch.clone()
    masses = torch.tensor([[3.0]], device=device)
    articulation.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_mass.torch[env_ids][:, body_ids], masses)
    torch.testing.assert_close(articulation.data.body_mass.torch[:1], initial_mass[:1])
    torch.testing.assert_close(_model_attribute(articulation, "body_mass")[env_ids][:, body_ids], masses)
    assert notifications == [ModelFlags.BODY_INERTIAL_PROPERTIES]

    notifications.clear()
    initial_com = articulation.data.body_com_pos_b.torch.clone()
    coms = torch.tensor([[[0.05, -0.02, 0.01]]], device=device)
    articulation.set_coms_index(coms=coms, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_com_pos_b.torch[env_ids][:, body_ids], coms)
    torch.testing.assert_close(articulation.data.body_com_pos_b.torch[:1], initial_com[:1])
    assert notifications == [ModelFlags.BODY_INERTIAL_PROPERTIES]

    notifications.clear()
    initial_inertia = articulation.data.body_inertia.torch.clone()
    inertias = torch.diag_embed(torch.tensor([[[2.0, 3.0, 4.0]]], device=device)).reshape(1, 1, 9)
    articulation.set_inertias_index(inertias=inertias, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_inertia.torch[env_ids][:, body_ids], inertias)
    torch.testing.assert_close(articulation.data.body_inertia.torch[:1], initial_inertia[:1])
    assert notifications == [ModelFlags.BODY_INERTIAL_PROPERTIES]
    monkeypatch.undo()

    # restore the configured friction and inertial properties for the tests that drive this island
    articulation.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=initial_friction)
    articulation.set_masses_index(masses=initial_mass)
    articulation.set_coms_index(coms=initial_com)
    articulation.set_inertias_index(inertias=initial_inertia)

    # Passive viscous joint damping is distinct from actuator derivative gains.
    articulation = scene.articulations["fixed"]
    slide_joint_ids = articulation.actuators["slides"].joint_indices
    expected_viscous_friction = torch.full((articulation.num_instances, 3), 0.25, device=device)
    torch.testing.assert_close(
        articulation.data.joint_viscous_friction_coeff.torch[:, slide_joint_ids], expected_viscous_friction
    )
    torch.testing.assert_close(
        _model_attribute(articulation, "joint_damping")[:, slide_joint_ids], expected_viscous_friction
    )
    expected_pd_damping = torch.full_like(expected_viscous_friction, 100.0)
    torch.testing.assert_close(articulation.data.joint_damping.torch[:, slide_joint_ids], expected_pd_damping)
    torch.testing.assert_close(
        _model_attribute(articulation, "joint_target_kd")[:, slide_joint_ids], expected_pd_damping
    )
    for writer, value in (
        (articulation.write_joint_viscous_friction_coefficient_to_sim_index, 0.25),
        (articulation.write_joint_viscous_friction_coefficient_to_sim_mask, 0.5),
    ):
        values = torch.full((articulation.num_instances, articulation.num_joints), value, device=device)
        writer(joint_viscous_friction_coeff=values)
        torch.testing.assert_close(articulation.data.joint_viscous_friction_coeff.torch, values)
        torch.testing.assert_close(_model_attribute(articulation, "joint_damping"), values)

    # Distinct per-env rows catch writers that ignore the env index
    friction = torch.rand(articulation.num_instances, articulation.num_joints, device=device)
    assert not torch.allclose(friction[0], friction[1])
    articulation.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=friction)
    torch.testing.assert_close(_model_attribute(articulation, "joint_friction"), friction)
    # restore the fixture's passive joint properties
    articulation.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=torch.zeros_like(friction))
    viscous_friction = torch.zeros_like(friction)
    viscous_friction[:, slide_joint_ids] = 0.25
    articulation.write_joint_viscous_friction_coefficient_to_sim_index(joint_viscous_friction_coeff=viscous_friction)

    # Configured gains reach the actuator; unset gains fall back to the USD drives, which author angular gains
    # per degree.
    slides, wrist = articulation.actuators["slides"], articulation.actuators["wrist"]
    torch.testing.assert_close(slides.stiffness, torch.full((2, 3), 2000.0, device=device))
    torch.testing.assert_close(slides.damping, torch.full((2, 3), 100.0, device=device))
    usd_stiffness_per_radian = (torch.tensor(WRIST_USD_STIFFNESS, device=device) * 180.0 / torch.pi).expand(2, -1)
    torch.testing.assert_close(wrist.stiffness, usd_stiffness_per_radian, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(wrist.damping, 0.1 * usd_stiffness_per_radian, rtol=1e-5, atol=1e-5)


def test_articulation_drive_and_dynamics(scene: _Scene) -> None:
    """Prove fixed-root actuation, live body velocities, and a partial drive command.

    Fixed roots, including one fixed by a spawner fragment, hold their default state while the joints move.
    The reported body velocities track the finite-difference motion of the driven links, which stay zero if
    a fixed-base fallback zeroes the body velocity binding.
    """
    articulation = scene.articulations["fixed"]
    root_fixed = scene.articulations["root_fixed"]
    scene.rest(articulation)
    scene.rest(root_fixed)
    fixed_roots = {name: scene.default_root_pose_w(scene.articulations[name]) for name in ("fixed", "root_fixed")}
    default_root_vel = articulation.data.default_root_vel.torch.clone()
    initial_joint_pos = articulation.data.joint_pos.torch.clone()

    # command a step away from the default pose on environment 1 only
    joint_pos_target = articulation.data.default_joint_pos.torch.clone()
    joint_pos_target[1, 0] += 0.1
    articulation.actuators.target_command.set_position_index(value=joint_pos_target)
    prev_body_pos = articulation.data.body_link_pos_w.torch.clone()
    reported_max = []
    fin_diff_max = []
    for _ in range(20):
        scene.step("fixed", "root_fixed")
        for name, default_root_pose in fixed_roots.items():
            torch.testing.assert_close(scene.articulations[name].data.root_link_pose_w.torch, default_root_pose)
            torch.testing.assert_close(scene.articulations[name].data.root_com_vel_w.torch, default_root_vel)
        body_pos = articulation.data.body_link_pos_w.torch
        reported_max.append(articulation.data.body_lin_vel_w.torch[1].norm(dim=-1).amax())
        fin_diff_max.append(((body_pos[1] - prev_body_pos[1]) / scene.sim.cfg.dt).norm(dim=-1).amax())
        prev_body_pos = body_pos.clone()
    reported_max = torch.stack(reported_max).amax()
    fin_diff_max = torch.stack(fin_diff_max).amax()

    # the commanded motion genuinely moves the bodies
    assert fin_diff_max > 0.1
    # and the reported body velocities track it (identically zero under the regression)
    assert reported_max > 0.5 * fin_diff_max
    assert articulation.data.joint_pos.torch[1, 0] > initial_joint_pos[1, 0] + 0.05
    # the environment that was not commanded stays at rest
    torch.testing.assert_close(articulation.data.joint_pos.torch[0], initial_joint_pos[0], atol=1e-6, rtol=0)


def test_floating_articulation_root_and_wrench_response(scene: _Scene, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove floating-root state writes, center-of-mass updates, and wrench delivery.

    A root pose or velocity write reads back in the written frame and refreshes the derived frame without a
    step; a partial write changes only the selected environment. Forces on several bodies and forces at a
    position, in the local and the global frame, reach the solver as the matching net force and moment, and a
    reset clears the wrenches of the selected environments only.
    """
    device = scene.device
    articulation = scene.articulations["floating"]
    scene.rest(articulation)
    num_articulations = articulation.num_instances
    env_pos = scene.origins
    assert articulation.data.root_link_pose_w.torch.shape == (2, 7)
    assert articulation.data.root_com_pose_w.torch.shape == (2, 7)
    assert articulation.data.body_link_pose_w.torch.shape == (2, articulation.num_bodies, 7)
    assert articulation.data.body_com_pose_w.torch.shape == (2, articulation.num_bodies, 7)

    # Resolve root body index by name (ordering may differ across physics backends)
    root_idx = articulation.find_bodies("Root")[0][0]

    # change center of mass offset from link frame on the root body
    original_com = articulation.data.body_com_pos_b.torch.clone()
    # Populate the derived world-frame cache so a missing invalidation would surface as a stale read.
    original_com_w = articulation.data.body_com_pos_w.torch.clone()
    new_com = original_com.clone()
    new_com[:, root_idx] = torch.tensor([1.0, 0.0, 0.0], device=device)
    env_ids = torch.arange(num_articulations, device=device, dtype=torch.int32)
    # Full poses are accepted too; Newton uses the position and ignores the orientation.
    com_poses = articulation.data.body_com_pose_b.torch.clone()
    com_poses[..., :3] = new_com
    articulation.set_coms_index(coms=com_poses, env_ids=env_ids)
    torch.testing.assert_close(articulation.data.body_com_pos_b.torch, new_com, atol=1e-5, rtol=1e-5)
    articulation.set_coms_index(coms=original_com, env_ids=env_ids)
    torch.testing.assert_close(articulation.data.body_com_pos_b.torch, original_com, atol=1e-5, rtol=1e-5)
    _ = articulation.data.body_com_pos_w.torch
    articulation.set_coms_index(coms=new_com, env_ids=env_ids)

    torch.testing.assert_close(articulation.data.body_com_pos_b.torch, new_com, atol=1e-5, rtol=1e-5)
    # Without a sim step the links stay put, so the world-frame CoM is the link pose applied to the new offset.
    link_pos_w = articulation.data.body_link_pos_w.torch
    link_quat_w = articulation.data.body_link_quat_w.torch
    expected_com_w = link_pos_w + math_utils.quat_apply(link_quat_w, new_com)
    updated_com_w = articulation.data.body_com_pos_w.torch
    torch.testing.assert_close(updated_com_w, expected_com_w, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(updated_com_w, original_com_w)

    def random_root_state(num_envs: int) -> torch.Tensor:
        state = torch.rand(num_envs, 13, device=device)
        state[..., :3] += env_pos[:num_envs]
        # make quaternion a unit vector
        state[..., 3:7] = torch.nn.functional.normalize(state[..., 3:7], dim=-1)
        return state

    for state_location in ("com", "link", "root"):

        def write_root_state(state: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
            if state_location == "com":
                articulation.write_root_com_pose_to_sim_index(root_pose=state[..., :7], env_ids=env_ids)
                articulation.write_root_com_velocity_to_sim_index(root_velocity=state[..., 7:], env_ids=env_ids)
            elif state_location == "link":
                articulation.write_root_link_pose_to_sim_index(root_pose=state[..., :7], env_ids=env_ids)
                articulation.write_root_link_velocity_to_sim_index(root_velocity=state[..., 7:], env_ids=env_ids)
            elif state_location == "root":
                articulation.write_root_pose_to_sim_index(root_pose=state[..., :7], env_ids=env_ids)
                articulation.write_root_velocity_to_sim_index(root_velocity=state[..., 7:], env_ids=env_ids)

        def read_written_root_state() -> torch.Tensor:
            # the root pose is the link pose and the root velocity is the center-of-mass velocity
            pose_w = (
                articulation.data.root_com_pose_w if state_location == "com" else articulation.data.root_link_pose_w
            )
            vel_w = articulation.data.root_link_vel_w if state_location == "link" else articulation.data.root_com_vel_w
            return torch.cat((pose_w.torch, vel_w.torch), dim=-1)

        rand_state = random_root_state(num_articulations)
        scene.step("floating")

        # Prime the lazily-derived caches at the current sim timestamp. Without this they would
        # recompute on first access after the write regardless of invalidation; priming them makes a
        # missing reset_pose/reset_velocity observable as a stale read in the assertions below.
        _ = articulation.data.root_link_pose_w.torch
        _ = articulation.data.root_com_pose_w.torch
        _ = articulation.data.root_link_vel_w.torch
        _ = articulation.data.root_com_vel_w.torch

        write_root_state(rand_state)

        body_com_pose_b = articulation.data.body_com_pose_b.torch
        if state_location == "com":
            # the com pose/vel was written, so the derived link pose/vel must be refreshed
            root_com_pose_w = articulation.data.root_com_pose_w.torch
            root_com_vel_w = articulation.data.root_com_vel_w.torch
            expected_root_link_pos, expected_root_link_quat = math_utils.combine_frame_transforms(
                root_com_pose_w[:, :3],
                root_com_pose_w[:, 3:],
                math_utils.quat_rotate(
                    math_utils.quat_inv(body_com_pose_b[:, root_idx, 3:7]), -body_com_pose_b[:, root_idx, :3]
                ),
                math_utils.quat_inv(body_com_pose_b[:, root_idx, 3:7]),
            )
            expected_root_link_pose = torch.cat((expected_root_link_pos, expected_root_link_quat), dim=1)
            root_link_pose_w = articulation.data.root_link_pose_w.torch
            root_link_vel_w = articulation.data.root_link_vel_w.torch
            torch.testing.assert_close(expected_root_link_pose, root_link_pose_w)
            # skip lin_vel because it differs from the link frame; angular velocity is frame-independent
            # and only matches when the derived velocity was actually refreshed after the write
            torch.testing.assert_close(root_com_vel_w[:, 3:], root_link_vel_w[:, 3:])
        else:
            # the link pose/vel was written, so the derived com pose/vel must be refreshed
            root_link_pose_w = articulation.data.root_link_pose_w.torch
            root_link_vel_w = articulation.data.root_link_vel_w.torch
            expected_com_pos, expected_com_quat = math_utils.combine_frame_transforms(
                root_link_pose_w[:, :3],
                root_link_pose_w[:, 3:],
                body_com_pose_b[:, root_idx, :3],
                body_com_pose_b[:, root_idx, 3:7],
            )
            expected_com_pose = torch.cat((expected_com_pos, expected_com_quat), dim=1)
            root_com_pose_w = articulation.data.root_com_pose_w.torch
            root_com_vel_w = articulation.data.root_com_vel_w.torch
            torch.testing.assert_close(expected_com_pose, root_com_pose_w)
            torch.testing.assert_close(root_link_vel_w[:, 3:], root_com_vel_w[:, 3:])

        # the written state reads back in the written frame
        torch.testing.assert_close(read_written_root_state(), rand_state)

        # a partial write only changes the selected environment
        partial_state = random_root_state(1)
        write_root_state(partial_state, env_ids=torch.tensor([0], device=device, dtype=torch.int32))
        expected_state = rand_state.clone()
        expected_state[0] = partial_state[0]
        torch.testing.assert_close(read_written_root_state(), expected_state)

    articulation.set_coms_index(coms=original_com)

    # Wrenches reach the solver as the matching net force and moment. Environment 0 receives none and stays at
    # rest. The bodies lie on the x axis and the joint turns about z, so the island turns about x as one rigid
    # body whose inertia is the sum of the body inertias.
    total_mass = articulation.data.body_mass.torch[0].sum()
    rigid_inertia_xx = articulation.data.body_inertia.torch[0, :, 0].sum()
    dt = scene.sim.cfg.dt
    env_1 = torch.tensor([1], dtype=torch.int32, device=device)
    zeros = torch.zeros((1, articulation.num_bodies, 3), device=device)

    def one_step_response() -> tuple[torch.Tensor, torch.Tensor]:
        """Step once; return environment 1's center-of-mass linear velocity and root angular velocity."""
        scene.step("floating")
        torch.testing.assert_close(
            articulation.data.body_com_vel_w.torch[0], torch.zeros(2, 6, device=device), atol=1e-6, rtol=0.0
        )
        body_masses = articulation.data.body_mass.torch[1].unsqueeze(-1)
        com_lin_vel = (body_masses * articulation.data.body_com_lin_vel_w.torch[1]).sum(dim=0) / total_mass
        return com_lin_vel, articulation.data.root_com_ang_vel_w.torch[1]

    # Opposite forces on the two bodies form a pure moment about -z that turns both links.
    scene.rest(articulation)
    forces = zeros.clone()
    forces[0, :, 1] = torch.tensor([10.0, -10.0], device=device)
    articulation.permanent_wrench_composer.set_forces_and_torques_index(
        forces=forces, torques=zeros, env_ids=env_1, body_ids=torch.arange(2, device=device, dtype=torch.int32)
    )
    _, ang_vel = one_step_response()
    assert ang_vel[2] < -0.1
    torch.testing.assert_close(ang_vel[:2], torch.zeros(2, device=device), atol=1e-5, rtol=0.0)

    # A force at a position offset along y, set then added in the local frame and set then added in the global
    # frame, applies twice the force and twice its moment about x.
    force = 1.0
    for is_global in (False, True):
        scene.rest(articulation)
        positions = torch.tensor([[[0.0, 0.1, 0.0]]], device=device)
        if is_global:
            positions += articulation.data.body_com_pos_w.torch[1:, :1]
        forces = torch.tensor([[[0.0, 0.0, force]]], device=device)
        for write in (
            articulation.permanent_wrench_composer.set_forces_and_torques_index,
            articulation.permanent_wrench_composer.add_forces_and_torques_index,
        ):
            write(
                forces=forces,
                torques=zeros[:, :1],
                positions=positions,
                env_ids=env_1,
                body_ids=torch.tensor([root_idx], dtype=torch.int32, device=device),
                is_global=is_global,
            )
        lin_vel, ang_vel = one_step_response()
        torch.testing.assert_close(lin_vel[2], 2.0 * force * dt / total_mass, rtol=1e-3, atol=0.0)
        torch.testing.assert_close(ang_vel[0], 2.0 * 0.1 * force * dt / rigid_inertia_xx, rtol=1e-3, atol=0.0)

    # Check that the gains come from the configuration and that reset works per environment
    expected_stiffness = torch.full((articulation.num_instances, articulation.num_joints), 20.0, device=device)
    torch.testing.assert_close(articulation.actuators["joint"].stiffness, expected_stiffness)
    torch.testing.assert_close(articulation.actuators["joint"].damping, torch.full_like(expected_stiffness, 2.0))
    actuator = articulation.actuators["joint"]
    actuator_reset = actuator.reset
    reset_env_ids = []

    def record_actuator_reset(env_ids: torch.Tensor | None = None) -> None:
        reset_env_ids.append(env_ids)
        actuator_reset(env_ids)

    monkeypatch.setattr(actuator, "reset", record_actuator_reset)
    articulation.reset()
    assert reset_env_ids == [None]

    instantaneous_composer = articulation.instantaneous_wrench_composer
    permanent_composer = articulation.permanent_wrench_composer

    # Reset should zero external forces and torques
    assert not instantaneous_composer.active
    assert not permanent_composer.active
    assert torch.count_nonzero(instantaneous_composer.out_force_b.torch) == 0
    assert torch.count_nonzero(instantaneous_composer.out_torque_b.torch) == 0
    assert torch.count_nonzero(permanent_composer.out_force_b.torch) == 0
    assert torch.count_nonzero(permanent_composer.out_torque_b.torch) == 0

    # A partial reset clears only the selected environment's wrenches
    num_bodies = articulation.num_bodies
    permanent_composer.set_forces_and_torques_index(
        forces=torch.ones((num_articulations, num_bodies, 3), device=device),
        torques=torch.ones((num_articulations, num_bodies, 3), device=device),
    )
    instantaneous_composer.add_forces_and_torques_index(
        forces=torch.ones((num_articulations, num_bodies, 3), device=device),
        torques=torch.ones((num_articulations, num_bodies, 3), device=device),
    )
    articulation.reset(env_ids=torch.tensor([0], device=device))
    assert instantaneous_composer.active
    assert permanent_composer.active
    assert torch.count_nonzero(instantaneous_composer.out_force_b.torch) == num_bodies * 3
    assert torch.count_nonzero(instantaneous_composer.out_torque_b.torch) == num_bodies * 3
    assert torch.count_nonzero(permanent_composer.out_force_b.torch) == num_bodies * 3
    assert torch.count_nonzero(permanent_composer.out_torque_b.torch) == num_bodies * 3
    articulation.reset()


def test_branching_fixture_physx_ordering_reorders_newton_to_bfs(scene: _Scene) -> None:
    """Resolve the documented Newton ``joint_ordering="physx"`` sim-to-sim workflow on a branching asset.

    Mirrors :func:`isaaclab_physx.test.assets.test_articulation.test_branching_fixture_resolves_distinct_conventions`
    with the backend roles swapped: here the live backend is Newton (depth-first, so its native view is
    the MJWarp order), and the request is ``physx``/``body_ordering="physx"``. Cross-backend discovery must
    resolve the breadth-first PhysX order and reorder the public joint/body axes to it. This is the headline
    workflow documented in
    ``docs/source/overview/core-concepts/physical-backends/joint_and_body_ordering.rst``.

    The same articulation also checks the MJWarp-order emulation used for cross-backend discovery and that
    selected inertial-property writes keep Newton's inverse arrays current under the body ordering.
    """
    articulation = scene.articulations["branching"]
    device = scene.device

    # Newton's native traversal is depth-first, so the live backend view already reflects MJWarp order.
    assert tuple(articulation.backend_joint_names) == BRANCHING_MJWARP_JOINT_NAMES
    assert tuple(articulation.backend_body_names) == BRANCHING_MJWARP_BODY_NAMES

    # The same-backend "mjwarp" request takes an identity fast path, so also force the cross-backend
    # emulation (the temporary Newton USD builder a PhysX-backed articulation would use) and compare its
    # independently rebuilt view against the live backend view. BFS and DFS differ on this branching fixture.
    emulated_names = ordering_resolvers._get_mjwarp_names_from_newton_usd_builder(articulation)
    assert emulated_names is not None
    assert emulated_names["joint"] == tuple(articulation.backend_joint_names)
    assert emulated_names["body"] == tuple(articulation.backend_body_names)
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="joint") == tuple(
        articulation.backend_joint_names
    )
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="body") == tuple(articulation.backend_body_names)

    # Cross-backend discovery (bypassing the same-backend fast path) resolves the breadth-first PhysX order.
    assert get_articulation_name_ordering(articulation, "physx", kind="joint") == BRANCHING_PHYSX_JOINT_NAMES
    assert get_articulation_name_ordering(articulation, "physx", kind="body") == BRANCHING_PHYSX_BODY_NAMES

    # The requested PhysX ordering reorders the public joint/body axes to the BFS convention.
    assert tuple(articulation.joint_names) == BRANCHING_PHYSX_JOINT_NAMES
    assert tuple(articulation.body_names) == BRANCHING_PHYSX_BODY_NAMES
    assert articulation.joint_ordering is not None
    assert articulation.body_ordering is not None

    # Selected mass and inertia writes with int64 selectors update Newton's inverse arrays in backend order.
    env_ids = torch.tensor([0], dtype=torch.int64, device=device)
    body_ids = torch.tensor([2, articulation.num_bodies - 1], dtype=torch.int64, device=device)
    backend_body_ids = torch.tensor(
        [articulation.data.body_ordering.user_to_backend_indices[index] for index in body_ids.tolist()],
        dtype=torch.int64,
        device=device,
    )
    assert backend_body_ids[0] != body_ids[0]
    model = SimulationManager.get_model()
    initial_masses = articulation.data.body_mass.torch.clone()
    initial_inertias = articulation.data.body_inertia.torch.clone()

    masses = articulation.data.body_mass.torch[env_ids][:, body_ids].clone() + torch.tensor([[1.0, 2.0]], device=device)
    articulation.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
    model_inv_mass = wp.to_torch(articulation.root_view.get_attribute("body_inv_mass", model)[:, 0])
    torch.testing.assert_close(model_inv_mass[env_ids][:, backend_body_ids], masses.reciprocal())

    inertia_matrices = torch.diag_embed(torch.tensor([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]], device=device))
    articulation.set_inertias_index(inertias=inertia_matrices.reshape(1, 2, 9), env_ids=env_ids, body_ids=body_ids)
    model_inv_inertia = wp.to_torch(articulation.root_view.get_attribute("body_inv_inertia", model)[:, 0])
    torch.testing.assert_close(model_inv_inertia[env_ids][:, backend_body_ids], torch.linalg.inv(inertia_matrices))

    articulation.set_masses_index(masses=initial_masses)
    articulation.set_inertias_index(inertias=initial_inertias)


def test_newton_ordered_state_publishes_in_step_and_refreshes_same_timestamp_writes(scene: _Scene) -> None:
    """Republish the user-order Tier-1 shadows inside the sim step and refresh them after same-timestamp writes.

    With non-identity ordering the passthrough state getters no longer reorder on read; the post-step
    callback republishes the shadows from live backend state inside the stepped region. We deliberately
    clobber the four shadows, step the simulation WITHOUT reading any state property, and assert the shadows
    again equal the reordered backend state -- which can only hold if the hook ran inside the step. Under the
    lazy design (no hook), only a property read would refresh them, so the clobbered shadows would stay stale.

    Root pose and velocity writes at the current simulation timestamp must then refresh the ordered body
    pose and velocity, including after another consumer has resolved the shared forward kinematics.
    """
    device = scene.device
    articulation = scene.articulations["ordered"]
    scene.rest(articulation)
    data = articulation.data
    assert data.joint_ordering is not None
    assert data.body_ordering is not None
    joint_u2b = np.asarray(articulation.joint_ordering.user_to_backend_indices)
    body_u2b = np.asarray(articulation.body_ordering.user_to_backend_indices)

    # Clobber every Tier-1 shadow with a large sentinel so a stale (unrepublished) shadow
    # is unmistakably detectable -- the true backend joint/velocity state sits near zero,
    # so a plain zero-fill would coincide with it. Then step WITHOUT touching joint_pos /
    # joint_vel / body_link_pose_w / body_com_vel_w.
    data._joint_pos_user.fill_(1000.0)
    data._joint_vel_user.fill_(1000.0)
    data._body_link_pose_w_user.fill_(wp.transformf(1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0, 1.0))
    data._body_com_vel_w_user.fill_(wp.spatial_vectorf(1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0))
    scene.sim.step()

    np.testing.assert_allclose(data._joint_pos_user.numpy(), data._sim_bind_joint_pos.numpy()[:, joint_u2b])
    np.testing.assert_allclose(data._joint_vel_user.numpy(), data._sim_bind_joint_vel.numpy()[:, joint_u2b])
    np.testing.assert_allclose(
        data._body_link_pose_w_user.numpy(), data._sim_bind_body_link_pose_w.numpy()[:, body_u2b]
    )
    np.testing.assert_allclose(data._body_com_vel_w_user.numpy(), data._sim_bind_body_com_vel_w.numpy()[:, body_u2b])

    articulation.update(scene.sim.cfg.dt)
    root_body_idx = articulation.find_bodies("base")[0][0]
    sim_timestamp = data._sim_timestamp

    cached_body_pose = data.body_link_pose_w.torch[:, root_body_idx].clone()
    written_root_pose = data.root_link_pose_w.torch.clone()
    written_root_pose[:, 0] += 0.25
    articulation.write_root_link_pose_to_sim_index(root_pose=written_root_pose)
    assert data._sim_timestamp == sim_timestamp
    torch.testing.assert_close(data.root_link_pose_w.torch, written_root_pose)
    SimulationManager.forward()  # Another consumer may resolve shared FK before this view reads.
    refreshed_body_pose = data.body_link_pose_w.torch[:, root_body_idx]
    torch.testing.assert_close(refreshed_body_pose, written_root_pose)
    assert not torch.equal(refreshed_body_pose, cached_body_pose)

    # Populate the velocity cache after the pose write so only the velocity write can invalidate it.
    cached_body_vel = data.body_com_vel_w.torch[:, root_body_idx].clone()
    written_root_vel = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], device=device).repeat(2, 1)
    articulation.write_root_com_velocity_to_sim_index(root_velocity=written_root_vel)
    assert data._sim_timestamp == sim_timestamp
    torch.testing.assert_close(data.root_com_vel_w.torch, written_root_vel)
    refreshed_body_vel = data.body_com_vel_w.torch[:, root_body_idx]
    torch.testing.assert_close(refreshed_body_vel, written_root_vel)
    assert not torch.equal(refreshed_body_vel, cached_body_vel)
    scene.rest(articulation)


@pytest.mark.parametrize("island", ["fixed", "ordered"], ids=["identity", "reordered"])
def test_write_data_to_sim_writes_joint_targets_in_backend_order(scene: _Scene, island: str) -> None:
    """Write the published joint position targets into the backend buffer in backend joint order.

    Implicit actuators on the Lab path publish the processed targets, which must reach the solver-bound
    buffer permuted to backend order. The Newton-actuator path is covered with the native actuators.
    """
    device = scene.device
    articulation = scene.articulations[island]
    scene.rest(articulation)
    assert (articulation.data.joint_ordering is not None) is (island == "ordered")
    assert articulation._has_newton_actuators is False

    # Distinct per-joint targets away from the defaults, so a skipped or unpermuted write is visible.
    target = articulation.data.default_joint_pos.torch.clone()
    target += 0.01 * torch.arange(1, articulation.num_joints + 1, device=device)
    articulation.set_joint_position_target_index(target=target)
    articulation.write_data_to_sim()

    source = articulation.actuators.output_command.position.torch
    torch.testing.assert_close(source, target)
    user_to_backend = (
        list(articulation.joint_ordering.user_to_backend_indices)
        if articulation.joint_ordering is not None
        else list(range(articulation.num_joints))
    )
    expected_backend_target = torch.empty_like(source)
    expected_backend_target[:, user_to_backend] = source
    torch.testing.assert_close(wp.to_torch(articulation.data._sim_bind_joint_position_target), expected_backend_target)
    scene.rest(articulation)


@pytest.mark.parametrize("island", ["limits_implicit", "limits_explicit"], ids=["implicit", "explicit"])
def test_setting_joint_limits_from_cfg(scene: _Scene, island: str) -> None:
    """Test the velocity and effort limit resolution for implicit and explicit actuators.

    This test verifies that:
    1. The solver clamps ``joint_velocity_limit`` and ``joint_effort_limit`` are applied to the simulation;
       when unset, the USD-authored values are kept
    2. The actuator limits keep their configured values and are never pushed to the solver
    3. When unset, the actuator velocity limit falls back to the solver clamp, implicit actuators track the
       solver effort clamp, and an explicit actuator's effort limit falls back to the USD-authored value

    Each joint of the island belongs to one group: joint limits only, actuator limits only, both, or neither.
    """
    articulation = scene.articulations[island]
    articulation_cfg = scene.cfgs[island]
    newton_vel_limit = _model_attribute(articulation, "joint_velocity_limit")
    newton_effort_limit = _model_attribute(articulation, "joint_effort_limit")
    usd_vel_limit = next(
        p.max_joint_velocity for p in articulation_cfg.spawn.joint_drive_props if isinstance(p, PhysxJointCfg)
    )
    usd_effort_limit = next(
        p.max_force for p in articulation_cfg.spawn.joint_drive_props if isinstance(p, sim_utils.UsdPhysicsDriveCfg)
    )

    # check data buffers
    torch.testing.assert_close(articulation.data.joint_vel_limits.torch, newton_vel_limit)
    torch.testing.assert_close(articulation.data.joint_effort_limits.torch, newton_effort_limit)

    for group_name, actuator in articulation.actuators.items():
        group_cfg = articulation_cfg.actuators[group_name]
        joint_limit = group_cfg.joint_velocity_limit
        actuator_limit = group_cfg.actuator_velocity_limit
        joint_ids = actuator.joint_indices
        group_vel_limit = newton_vel_limit[:, joint_ids]
        group_effort_limit = newton_effort_limit[:, joint_ids]

        # the solver clamps come from the joint limits when set, otherwise the USD-authored values
        expected_vel_limit = usd_vel_limit if joint_limit is None else joint_limit
        expected_effort_limit = usd_effort_limit if joint_limit is None else joint_limit
        torch.testing.assert_close(group_vel_limit, torch.full_like(group_vel_limit, expected_vel_limit))
        torch.testing.assert_close(group_effort_limit, torch.full_like(group_effort_limit, expected_effort_limit))

        # the actuator velocity limit keeps its configured value and is not pushed to the solver;
        # when unset it falls back to the solver clamp
        if actuator_limit is not None:
            torch.testing.assert_close(
                actuator.actuator_velocity_limit, torch.full_like(group_vel_limit, actuator_limit)
            )
            assert not torch.allclose(actuator.actuator_velocity_limit, group_vel_limit)
        else:
            torch.testing.assert_close(actuator.actuator_velocity_limit, group_vel_limit)

        if island == "limits_implicit":
            # without a separately configured rated limit, the implicit actuator limits track the solver clamp
            torch.testing.assert_close(actuator.joint_effort_limit, group_effort_limit)
            torch.testing.assert_close(actuator.actuator_effort_limit, group_effort_limit)
        elif actuator_limit is not None:
            torch.testing.assert_close(
                actuator.actuator_effort_limit, torch.full_like(group_effort_limit, actuator_limit)
            )
        else:
            # an unset explicit actuator effort limit falls back to the USD-authored value, not the solver clamp
            torch.testing.assert_close(
                actuator.actuator_effort_limit, torch.full_like(group_effort_limit, usd_effort_limit)
            )


def test_set_material_properties(scene: _Scene) -> None:
    """Material and collider-offset randomization write through to the articulation's shapes in the Newton model.

    The event terms write the asset's view-level shape bindings; the assertions read the flat Newton model
    arrays at the collision shapes of the selected bodies and environments. Material randomization runs once on
    a body subset and once on every body.
    """
    articulation = scene.articulations["floating"]
    num_articulations = articulation.num_instances
    device = scene.device

    # Resolve this articulation's shapes per environment and body from the flat Newton model.
    model = SimulationManager.get_model()
    body_world = model.body_world.numpy()
    body_labels = list(model.body_label)
    shape_body = model.shape_body.numpy()
    is_collision_shape = (model.shape_flags.numpy() & int(ShapeFlags.COLLIDE_SHAPES)) != 0

    def robot_shapes(env_index: int, selected_body_names: list[str] | None = None) -> np.ndarray:
        prefix = f"/World/Env_{env_index}/Floating/"
        bodies = [
            body
            for body in np.flatnonzero(body_world == env_index)
            if body_labels[body].startswith(prefix)
            and (selected_body_names is None or body_labels[body].rsplit("/", 1)[-1] in selected_body_names)
        ]
        return np.flatnonzero(np.isin(shape_body, bodies) & is_collision_shape)

    env = SimpleNamespace(scene={"robot": articulation}, sim=scene.sim, device=device, num_envs=num_articulations)
    env_ids = torch.tensor([num_articulations - 1], device=device)
    other_env_shapes = robot_shapes(0)
    configured_mu = model.shape_material_mu.numpy().copy()
    configured_restitution = model.shape_material_restitution.numpy().copy()

    # Randomize the materials in the last environment, with degenerate ranges.
    for body_subset, friction, restitution_value in ((True, 0.55, 0.15), (False, 0.65, 0.25)):
        asset_cfg = SceneEntityCfg("robot")
        selected_body_names = None
        if body_subset:
            asset_cfg.body_ids, selected_body_names = articulation.find_bodies(["Child"])
        selected_shapes = robot_shapes(num_articulations - 1, selected_body_names)
        unselected_shapes = np.setdiff1d(robot_shapes(num_articulations - 1), selected_shapes)
        assert len(selected_shapes) > 0 and len(other_env_shapes) > 0
        assert body_subset == (len(unselected_shapes) > 0)
        original_mu = model.shape_material_mu.numpy().copy()
        original_restitution = model.shape_material_restitution.numpy().copy()
        params = {
            "static_friction_range": (friction, friction),
            "dynamic_friction_range": (friction, friction),
            "restitution_range": (restitution_value, restitution_value),
            "num_buckets": 1,
            "asset_cfg": asset_cfg,
        }
        material_term = randomize_rigid_body_material(
            EventTermCfg(func=randomize_rigid_body_material, mode="startup", params=params), env
        )
        material_term(env, env_ids, **params)
        scene.step("floating")

        mu = model.shape_material_mu.numpy()
        restitution = model.shape_material_restitution.numpy()
        np.testing.assert_allclose(mu[selected_shapes], friction)
        np.testing.assert_allclose(restitution[selected_shapes], restitution_value)
        for shapes in (unselected_shapes, other_env_shapes):
            np.testing.assert_array_equal(mu[shapes], original_mu[shapes])
            np.testing.assert_array_equal(restitution[shapes], original_restitution[shapes])

    # Randomize the collider offsets of every shape in the last environment.
    original_margin = model.shape_margin.numpy().copy()
    original_gap = model.shape_gap.numpy().copy()
    offset_params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "rest_offset_distribution_params": (0.01, 0.01),
        "contact_offset_distribution_params": (0.03, 0.03),
    }
    offset_term = randomize_rigid_body_collider_offsets(
        EventTermCfg(func=randomize_rigid_body_collider_offsets, mode="startup", params=offset_params), env
    )
    offset_term(env, env_ids, **offset_params)
    scene.step("floating")

    # Newton maps the rest offset to the shape margin and the contact offset to margin + gap.
    randomized_shapes = robot_shapes(num_articulations - 1)
    np.testing.assert_allclose(model.shape_margin.numpy()[randomized_shapes], 0.01)
    np.testing.assert_allclose(model.shape_gap.numpy()[randomized_shapes], 0.02, atol=1e-6)
    np.testing.assert_array_equal(model.shape_margin.numpy()[other_env_shapes], original_margin[other_env_shapes])
    np.testing.assert_array_equal(model.shape_gap.numpy()[other_env_shapes], original_gap[other_env_shapes])

    # restore the configured shape properties, which the other tests' contacts rely on
    model.shape_material_mu.assign(configured_mu)
    model.shape_material_restitution.assign(configured_restitution)
    model.shape_margin.assign(original_margin)
    model.shape_gap.assign(original_gap)
    SimulationManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)


def test_hand_with_tendons_targets_only_given_envs_and_properties_reach_solver(scene: _Scene) -> None:
    """Command fixed tendons for one environment and write fixed-tendon properties through to the MuJoCo solver.

    ``set_fixed_tendon_position_target_index`` is declared backend-neutral and documented to accept
    partial data. Newton took ``env_ids`` and never forwarded it, so a partial command was sized
    against every instance and raised rather than commanding the environment asked for.

    Written fixed tendon stiffness, damping, and position limits reach the MuJoCo solver through both the
    index and the mask setters and writers.
    """
    device = scene.device
    articulation = scene.articulations["tendon"]
    scene.rest(articulation)
    num_articulations = articulation.num_instances
    assert articulation.num_fixed_tendons == 2

    target = torch.full((1, articulation.num_fixed_tendons), 1.0, dtype=torch.float32, device=device)
    articulation.set_fixed_tendon_position_target_index(target=target, env_ids=[0])
    scene.step("tendon", num_steps=30)

    # Both environments start from the same pose, so any divergence comes from the command -- and identical
    # poses would mean it reached both.
    commanded, untouched = articulation.data.joint_pos.torch[0], articulation.data.joint_pos.torch[1]
    assert not torch.allclose(commanded, untouched)
    # Each tendon ``XJ0`` is the sum of joints ``XJ1`` and ``XJ2``, so the commanded environment's tendon lengths
    # must have moved toward the 1.0 target and away from the uncommanded environment, which the actuator holds
    # at its 0.0 control.
    for tendon_name in articulation.fixed_tendon_names:
        joint_ids, _ = articulation.find_joints([tendon_name[:-1] + "1", tendon_name[:-1] + "2"])
        assert commanded[joint_ids].sum() > untouched[joint_ids].sum()
    articulation.set_fixed_tendon_position_target_index(target=torch.zeros_like(target), env_ids=[0])

    shape = (num_articulations, articulation.num_fixed_tendons)
    limits = torch.tensor([-0.1, 0.2], device=device).expand(*shape, 2)
    authored_stiffness = articulation.data.fixed_tendon_stiffness.torch.clone()
    authored_damping = articulation.data.fixed_tendon_damping.torch.clone()
    authored_limits = articulation.data.fixed_tendon_pos_limits.torch.clone()

    articulation.set_fixed_tendon_stiffness_mask(stiffness=torch.full(shape, 12.0, device=device))
    articulation.set_fixed_tendon_damping_index(damping=torch.full(shape, 3.0, device=device))
    articulation.set_fixed_tendon_position_limit_index(limit=limits)
    articulation.write_fixed_tendon_properties_to_sim_mask()
    scene.step("tendon")

    solver_model = SimulationManager._solver.mjw_model
    np.testing.assert_allclose(solver_model.tendon_stiffness.numpy(), 12.0)
    np.testing.assert_allclose(solver_model.tendon_damping.numpy(), 3.0)
    np.testing.assert_allclose(solver_model.tendon_range.numpy(), limits.cpu().numpy(), rtol=1e-6)
    torch.testing.assert_close(articulation.data.fixed_tendon_pos_limits.torch, limits)

    articulation.set_fixed_tendon_stiffness_index(stiffness=42.0, env_ids=[1], fixed_tendon_ids=[1])
    articulation.set_fixed_tendon_damping_index(damping=6.0, env_ids=[1], fixed_tendon_ids=[1])
    articulation.write_fixed_tendon_properties_to_sim_index(env_ids=[1], fixed_tendon_ids=[1])
    scene.step("tendon")
    expected_stiffness = np.full(shape, 12.0)
    expected_damping = np.full(shape, 3.0)
    expected_stiffness[1, 1] = 42.0
    expected_damping[1, 1] = 6.0
    np.testing.assert_allclose(solver_model.tendon_stiffness.numpy(), expected_stiffness)
    np.testing.assert_allclose(solver_model.tendon_damping.numpy(), expected_damping)

    # restore the authored tendon properties, which the other tests' dynamics rely on
    articulation.set_fixed_tendon_stiffness_index(stiffness=authored_stiffness)
    articulation.set_fixed_tendon_damping_index(damping=authored_damping)
    articulation.set_fixed_tendon_position_limit_index(limit=authored_limits)
    articulation.write_fixed_tendon_properties_to_sim_index()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("island", ["fixed", "floating", "ordered"], ids=["fixed", "floating", "floating_reordered"])
def test_get_gravity_compensation_forces_matches_jacobian_gravity(scene: _Scene, island: str) -> None:
    """``g(q)`` must equal ``-sum_b J_com_b^T (m_b * g_w)`` at the current configuration.

    Newton computes the gravity compensation force through an RNEA pass
    (``eval_inverse_dynamics_passive``); the static identity above derives the same
    quantity independently from the COM-referenced Jacobian and the per-body masses,
    pinning the sign convention, the DoF ordering (including the 6 floating-base
    entries), and the flat-buffer view gather in one assertion. Non-default joint
    positions and — for floating-base — a rotated, lifted root pose guard the corner
    fixed upstream in newton#2625 (wrong gravity compensation under non-identity
    root pose).

    On the reordered island a nonidentity joint ordering is active and both sides of the identity must be
    expressed in user joint order: the Jacobian gather applies the user->backend permutation, so a
    ``gather_dof_force_rows`` that skips it returns backend-ordered forces and breaks the identity row-wise.

    The same islands also pin:

    * the per-articulation shapes of the Jacobian, mass matrix and gravity compensation
      accessors in a scene whose articulations have different DoF counts, and a symmetric,
      positive-definite mass matrix. Fixed-base: ``body_link_jacobian_w`` drops the fixed-root row, so its
      shape is ``(N, num_bodies - 1, 6, num_joints)``. Floating-base: every body row is kept and
      ``num_base_dofs`` floating-base columns/entries are prepended on the DoF axis, matching the
      cross-library convention (Pinocchio, Drake, MuJoCo, RBDL, OCS2, iDynTree). A zero-padded
      model-wide sizing would surface as wrong shapes or non-positive mass-matrix diagonals;
    * that every dynamics accessor reflects a manual joint write without a sim step (the FK
      trigger before ``eval_jacobian``, ``eval_mass_matrix`` and the RNEA pass);
    * the link-origin Jacobian contract: ``J · q_dot`` must encode the link-origin twist
      ``v_origin = v_com - omega x (R · body_com_pos_b)``, which the IsaacLab task-space
      controllers (IK / OSC / RMPFlow) rely on. Newton's ``eval_jacobian`` natively produces
      COM-referenced rows, so the ground truth reads Newton's per-body twist directly from the
      ArticulationView state. The fixed island's links carry center-of-mass offsets.
    """
    articulation = scene.articulations[island]
    num_articulations = articulation.num_instances
    device = scene.device
    scene.rest(articulation)

    # Sanity: the islands have different DoF counts, so a regression to model-wide sizing would manifest as wrong
    # shapes on the smaller views.
    model = SimulationManager.get_model()
    island_dofs = [other.num_joints + other.num_base_dofs for other in scene.articulations.values()]
    assert model.max_dofs_per_articulation == max(island_dofs) > min(island_dofs)

    num_dofs = articulation.num_joints + articulation.num_base_dofs
    num_jacobian_bodies = articulation.num_bodies - 1 if articulation.is_fixed_base else articulation.num_bodies

    with world_gravity(_GRAVITY):
        J = articulation.data.body_link_jacobian_w.torch
        assert J.shape == torch.Size((num_articulations, num_jacobian_bodies, 6, num_dofs)), tuple(J.shape)
        assert J.dtype == torch.float32

        g = articulation.data.gravity_compensation_forces.torch
        assert g.shape == torch.Size((num_articulations, num_dofs)), tuple(g.shape)
        assert g.dtype == torch.float32
        assert g.abs().max() > 1e-3, "gravity compensation is all-zero"

        scene.step(island)

        M = articulation.data.mass_matrix.torch
        assert M.shape == torch.Size((num_articulations, num_dofs, num_dofs)), tuple(M.shape)
        assert M.dtype == torch.float32
        diag = M.diagonal(dim1=-2, dim2=-1)
        assert (diag > 1e-6).all(), f"mass matrix has non-positive diagonal entries: min={diag.min()}"

        # The joint-space inertia is symmetric by construction; asymmetry means a wrong-axis gather or a
        # half-populated buffer. OSC inverts ``J M^-1 J^T`` every step, so ``M`` must also be positive-definite.
        asym = (M - M.transpose(-1, -2)).abs().max().item()
        assert asym < 1e-4, f"|M - M^T|_max = {asym:.3e} — mass matrix is not symmetric"
        # A tiny jitter tolerates the float32 eigenvalue floor without masking real non-PD bugs.
        eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype).expand_as(M)
        torch.linalg.cholesky(M + 1e-6 * eye)

        # Read every accessor at the stepped joint state.
        J_link_0 = articulation.data.body_link_jacobian_w.torch.clone()
        J_com_0 = articulation.data.body_com_jacobian_w.torch.clone()
        M_0 = articulation.data.mass_matrix.torch.clone()
        g_0 = articulation.data.gravity_compensation_forces.torch.clone()

        # Non-trivial configuration via manual writes (no sim step, so the assert
        # compares both quantities at exactly this state): random joint offsets,
        # and for floating-base a non-identity root pose.
        torch.manual_seed(0)
        q = articulation.data.default_joint_pos.torch + 0.3 * torch.randn(
            num_articulations, articulation.num_joints, device=device
        )
        articulation.write_joint_position_to_sim_index(position=q)
        if not articulation.is_fixed_base:
            root_pose = scene.default_root_pose_w(articulation)
            root_pose[:, 2] += 1.0
            # (x, y, z, w) quaternion — 30 deg roll about x.
            root_pose[:, 3:] = torch.tensor([0.2588, 0.0, 0.0, 0.9659], device=device)
            articulation.write_root_pose_to_sim_index(root_pose=root_pose)
            # Guard against a vacuous identity: if root-pose FK invalidation ever
            # regressed, both sides would be evaluated at the stale identity pose and
            # agree trivially, voiding the newton#2625 rotated-root coverage.
            torch.testing.assert_close(articulation.data.root_link_pose_w.torch, root_pose, atol=1e-5, rtol=0.0)

        # With the FK trigger, forward() refreshes body_q to the written state before each accessor evaluates.
        # Without it, body_q stays at the previous state and every accessor returns its previous value.
        assert not torch.allclose(J_link_0, articulation.data.body_link_jacobian_w.torch, atol=1e-3), (
            "body_link_jacobian_w did not change after manual joint write; FK trigger likely missing"
        )
        assert not torch.allclose(J_com_0, articulation.data.body_com_jacobian_w.torch, atol=1e-3), (
            "body_com_jacobian_w did not change after manual joint write; FK trigger likely missing"
        )
        assert not torch.allclose(M_0, articulation.data.mass_matrix.torch, atol=1e-3), (
            "mass_matrix did not change after manual joint write; FK trigger likely missing"
        )
        assert not torch.allclose(g_0, articulation.data.gravity_compensation_forces.torch, atol=1e-3), (
            "gravity_compensation_forces did not change after manual joint write; FK trigger likely missing"
        )

        g_meas = articulation.data.gravity_compensation_forces.torch

        # Independent derivation: generalized gravity load tau_g = sum_b J_lin_b^T (m_b g_w);
        # the compensation force is its negation. The COM-referenced Jacobian is exactly the
        # right lever arm for a point gravity force acting at each body's COM.
        J_com = articulation.data.body_com_jacobian_w.torch  # (N, B_jac, 6, D)
        masses = articulation.data.body_mass.torch  # (N, num_bodies)
        if articulation.is_fixed_base:
            # jacobi_body_idx == body_idx - 1 for fixed-base (fixed-root row excluded).
            masses = masses[:, 1:]
        gravity_w = wp.to_torch(model.gravity[: model.world_count])  # (num_worlds, 3)
        assert gravity_w.shape[0] == num_articulations, "fixture must place one articulation per world"
        f_gravity = masses.unsqueeze(-1) * gravity_w.unsqueeze(1)  # (N, B_jac, 3)
        g_expected = -torch.einsum("nbij,nbi->nj", J_com[:, :, 0:3, :], f_gravity)

        torch.testing.assert_close(g_meas, g_expected, atol=1e-2, rtol=1e-3)

    # Link-origin Jacobian contract. Reproducible non-trivial q_dot — large enough to drive omega
    # well above the floor where COM offset effects would round into noise. The root is at rest so
    # the actuated-only J slice predicts the whole body twist.
    qdot = torch.randn(num_articulations, articulation.num_joints, device=device) * 0.5
    articulation.write_joint_velocity_to_sim_index(velocity=qdot)
    if not articulation.is_fixed_base:
        articulation.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num_articulations, 6, device=device))
    # Refresh kinematics without integrating: for floating bases, an actuator step can
    # introduce root motion that is intentionally absent from the actuated-only J slice.
    scene.sim.forward()
    articulation.update(scene.sim.cfg.dt)

    # body_link_jacobian_w prepends ``num_base_dofs`` floating-base columns; slice past
    # them so the joint axis aligns with joint_vel (actuated-only).
    J = articulation.data.body_link_jacobian_w.torch[..., articulation.num_base_dofs :]
    qdot_view = articulation.data.joint_vel.torch
    v_pred = torch.einsum("nbij,nj->nbi", J, qdot_view)  # (N, B_jac, 6)
    v_pred_lin = v_pred[..., 0:3]
    v_pred_ang = v_pred[..., 3:6]

    # Ground truth from Newton state. ``get_link_velocities`` returns shape
    # (num_instances, 1, num_bodies, 6) — per-articulation grouping with
    # one articulation per instance — so we squeeze the inner dim.
    state = SimulationManager.get_state_0()
    body_qd_view = wp.to_torch(articulation.root_view.get_link_velocities(state)).squeeze(1)
    body_v_com = body_qd_view[..., :3]
    body_omega = body_qd_view[..., 3:]

    # World-frame COM-to-origin offset, derived from already-computed
    # data layer outputs (avoids quaternion-convention pitfalls).
    body_com_pos_w = articulation.data.body_com_pos_w.torch  # (N, num_bodies, 3)
    body_link_pos_w = articulation.data.body_link_pos_w.torch  # (N, num_bodies, 3)
    c_world = body_com_pos_w - body_link_pos_w

    if articulation.is_fixed_base:
        body_v_com = body_v_com[:, 1:]
        body_omega = body_omega[:, 1:]
        c_world = c_world[:, 1:]
    elif articulation.body_ordering is not None:
        # the view reports backend body order; the public Jacobian rows follow the public body order
        body_u2b = list(articulation.body_ordering.user_to_backend_indices)
        body_v_com = body_v_com[:, body_u2b]
        body_omega = body_omega[:, body_u2b]

    # Expected v_origin = v_com - omega x c_world.
    v_origin_expected = body_v_com - torch.cross(body_omega, c_world, dim=-1)

    # Tolerance: 5 mm absolute; a missing COM-offset correction is centimeters off at these velocities.
    torch.testing.assert_close(v_pred_ang, body_omega, atol=5e-3, rtol=1e-2)
    torch.testing.assert_close(v_pred_lin, v_origin_expected, atol=5e-3, rtol=1e-2)
    scene.rest(articulation)


def test_body_root_state(scene: _Scene) -> None:
    """Body link and center-of-mass states must follow a revolute pendulum with a center-of-mass offset.

    The pendulum turns freely about z; the fixed pivot holds its default state, the angular velocity is the same
    in the link and center-of-mass frames, and the link and center-of-mass positions and velocities match the
    rigid-rotation ground truth. The center-of-mass offset is set through ``set_coms_index``.
    """
    articulation = scene.articulations["pendulum"]
    scene.rest(articulation)
    num_articulations = articulation.num_instances
    num_bodies = articulation.num_bodies
    device = scene.device

    # Resolve body indices by name (ordering may differ across physics backends)
    root_idx = articulation.body_names.index("CenterPivot")
    arm_idx = articulation.body_names.index("Arm")
    link_offset = [1.0, 0.0, 0.0]  # the offset from CenterPivot to Arm frames
    offset = [0.5, 0.0, 0.0]
    initial_com = articulation.data.body_com_pos_b.torch.clone()
    coms = initial_com.clone()
    coms[:, arm_idx] = torch.tensor(offset, device=device)
    articulation.set_coms_index(coms=coms)
    torch.testing.assert_close(articulation.data.body_com_pos_b.torch, coms)

    articulation.write_joint_velocity_to_sim_index(velocity=torch.tensor([[2.0], [-1.5]], device=device))
    pivot_pos_w = scene.default_root_pose_w(articulation)[:, :3]
    default_root_pose = scene.default_root_pose_w(articulation)
    default_root_vel = articulation.data.default_root_vel.torch.clone()
    for _ in range(50):
        scene.step("pendulum")

        # the fixed pivot holds its default state
        torch.testing.assert_close(articulation.data.root_link_pose_w.torch, default_root_pose)
        torch.testing.assert_close(articulation.data.root_com_vel_w.torch, default_root_vel)

        root_link_vel_w = articulation.data.root_link_vel_w.torch
        root_com_pose_w = articulation.data.root_com_pose_w.torch
        root_com_vel_w = articulation.data.root_com_vel_w.torch
        body_link_pose_w = articulation.data.body_link_pose_w.torch
        body_link_vel_w = articulation.data.body_link_vel_w.torch
        body_com_pose_w = articulation.data.body_com_pose_w.torch
        body_com_vel_w = articulation.data.body_com_vel_w.torch
        joint_pos = articulation.data.joint_pos.torch.unsqueeze(-1)
        joint_vel = articulation.data.joint_vel.torch.unsqueeze(-1)

        # the angular velocity is the same in the link and center-of-mass frames
        torch.testing.assert_close(root_com_vel_w[..., 3:], root_link_vel_w[..., 3:])
        torch.testing.assert_close(body_com_vel_w[..., 3:], body_link_vel_w[..., 3:])

        # the arm's link origin circles the pivot at the link offset; the pivot's link does not move
        lin_vel_gt = torch.zeros(num_articulations, num_bodies, 3, device=device)
        vx = -(link_offset[0]) * joint_vel * torch.sin(joint_pos)
        vy = (link_offset[0]) * joint_vel * torch.cos(joint_pos)
        lin_vel_gt[:, arm_idx, :] = torch.cat([vx, vy, torch.zeros_like(vx)], dim=-1).squeeze(-2)
        torch.testing.assert_close(lin_vel_gt[:, root_idx, :], root_link_vel_w[..., :3], atol=1e-3, rtol=1e-1)
        torch.testing.assert_close(lin_vel_gt, body_link_vel_w[..., :3], atol=1e-3, rtol=1e-1)

        # the arm's center of mass circles the pivot at the link offset plus the center-of-mass offset
        pos_gt = torch.zeros(num_articulations, num_bodies, 3, device=device)
        px = (link_offset[0] + offset[0]) * torch.cos(joint_pos)
        py = (link_offset[0] + offset[0]) * torch.sin(joint_pos)
        pos_gt[:, arm_idx, :] = torch.cat([px, py, torch.zeros_like(px)], dim=-1).squeeze(-2)
        pos_gt += pivot_pos_w.unsqueeze(-2)
        torch.testing.assert_close(pos_gt[:, root_idx, :], root_com_pose_w[..., :3], atol=1e-3, rtol=1e-1)
        torch.testing.assert_close(pos_gt, body_com_pose_w[..., :3], atol=1e-3, rtol=1e-1)

        # the center-of-mass orientation is the link orientation composed with the body-frame offset rotation
        com_quat_b = articulation.data.body_com_quat_b.torch
        com_quat_w = math_utils.quat_mul(body_link_pose_w[..., 3:], com_quat_b)
        torch.testing.assert_close(com_quat_w, body_com_pose_w[..., 3:])
        torch.testing.assert_close(com_quat_w[:, root_idx, :], root_com_pose_w[..., 3:])

    # the pendulum turned far enough for the trigonometric ground truth to discriminate
    assert (articulation.data.joint_pos.torch.abs() > 0.5).all()
    articulation.set_coms_index(coms=initial_com)
    scene.rest(articulation)


@pytest.mark.isaacsim_ci
def test_gravity_vec_w_tracks_model_gravity(scene: _Scene) -> None:
    """Per-env mutations to Newton's ``model.gravity`` reach ``GRAVITY_VEC_W`` and ``projected_gravity_b``.

    Regression for the pre-fix snapshot: ``GRAVITY_VEC_W`` used to be env 0's
    gravity broadcast to every env, hiding per-env gravity randomization (e.g.
    :class:`~isaaclab.envs.mdp.randomize_physics_scene_gravity`).
    """
    articulation = scene.articulations["floating"]
    scene.rest(articulation)
    num_articulations = articulation.num_instances
    device = scene.device

    # GRAVITY_VEC_W must share storage with Newton's per-env gravity array.
    model = SimulationManager.get_model()
    model_gravity_arr = model.gravity[: model.world_count]
    global_gravity = wp.to_torch(model.gravity)[-1].clone()
    assert articulation.data.GRAVITY_VEC_W.warp.ptr == model_gravity_arr.ptr
    assert articulation.data.GRAVITY_VEC_W.shape == (num_articulations,)

    # Randomize the per-env gravity through the public event term.
    new_gravity = torch.tensor(
        [[0.1 * (i + 1), 0.2 * (i + 1), -3.0 - float(i)] for i in range(num_articulations)],
        device=device,
        dtype=torch.float32,
    )
    with world_gravity((0.0, 0.0, 0.0)):
        env = SimpleNamespace(sim=scene.sim, device=device, num_envs=num_articulations)
        params = {"gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "operation": "abs"}
        event = randomize_physics_scene_gravity(EventTermCfg(func=randomize_physics_scene_gravity, params=params), env)
        for row, values in enumerate(new_gravity.tolist()):
            event(env, torch.tensor([row], device=device), (values, values), operation="abs")

        # Live view: new per-env values are visible immediately, no invalidation step.
        torch.testing.assert_close(articulation.data.GRAVITY_VEC_W.torch, new_gravity)
        torch.testing.assert_close(wp.to_torch(model.gravity)[-1], global_gravity)

        # Recompute the lazily-cached projected_gravity_b without sim.step. Project against the same quat
        # buffer the kernel reads so the expectation holds for any orientation.
        root_pose = scene.default_root_pose_w(articulation)
        # (x, y, z, w) quaternion — 30 deg roll about x.
        half_angle = torch.tensor(torch.pi / 12.0, device=device)
        root_pose[:, 3:] = torch.stack(
            (half_angle.sin(), torch.zeros_like(half_angle), torch.zeros_like(half_angle), half_angle.cos())
        )
        articulation.write_root_pose_to_sim_index(root_pose=root_pose)
        articulation.update(scene.sim.cfg.dt)
        root_quat = articulation.data.root_link_quat_w.torch
        expected = math_utils.quat_apply_inverse(root_quat, torch.nn.functional.normalize(new_gravity, dim=-1))
        torch.testing.assert_close(articulation.data.projected_gravity_b.torch, expected, atol=1e-5, rtol=1e-5)
    scene.rest(articulation)


def test_body_q_consistent_after_root_write(scene: _Scene) -> None:
    """Test that body_q is fresh when collide() runs after a root pose write.

    Regression test for a NaN bug where collide() used stale body_q after env
    reset because eval_fk was not called between write_root_pose and collide.

    The scene uses Newton's collision pipeline, and the test patches ``_simulate_physics_only`` to capture
    body_q at the moment collide() is called and asserts it matches joint_q.
    """
    device = scene.device
    articulation = scene.articulations["floating"]
    scene.rest(articulation)
    model = SimulationManager.get_model()
    jc_starts = model.joint_coord_world_start.numpy()
    body_starts = model.body_world_start.numpy()
    # The floating island is authored first, so its root owns the first body and joint coordinates of each world.
    assert model.body_label[int(body_starts[0])] == "/World/Env_0/Floating/Root"

    scene.step("floating", num_steps=5)

    # Teleport env 0 by 10m (simulating a reset)
    new_pose = scene.default_root_pose_w(articulation)
    new_pose[0, 0] += 10.0
    new_pose[0, 1] += 5.0
    articulation.write_root_pose_to_sim_index(
        root_pose=new_pose[0:1],
        env_ids=torch.tensor([0], device=device, dtype=torch.int32),
    )

    # Patch _simulate_physics_only to capture body_q before collide runs
    captured = {}
    original_simulate = SimulationManager._simulate_physics_only.__func__

    @classmethod  # type: ignore[misc]
    def _patched_simulate(cls) -> None:
        if cls._needs_collision_pipeline:
            bq = wp.to_torch(cls.backend.state_0.body_q)
            jq = wp.to_torch(cls.backend.state_0.joint_q)
            b0 = int(body_starts[0])
            jc0 = int(jc_starts[0])
            captured["bq_root"] = bq[b0, :3].clone()
            captured["jq_root"] = jq[jc0 : jc0 + 3].clone()
        original_simulate(cls)

    with patch.object(SimulationManager, "_simulate_physics_only", _patched_simulate):
        scene.sim.step()
    articulation.update(scene.sim.cfg.dt)

    assert captured, "collision pipeline did not run — _needs_collision_pipeline is False"

    bq_root = captured["bq_root"]
    jq_root = captured["jq_root"]
    diff = (jq_root - bq_root).abs().max().item()
    assert diff < 0.01, (
        f"body_q was stale when collide() ran: diff={diff:.4f}m, jq={jq_root.tolist()}, bq={bq_root.tolist()}"
    )
    scene.rest(articulation)


def test_joint_position_limit_clamping_respects_logging(scene: _Scene, caplog) -> None:
    """Logging level controls reporting without changing clamping or reusing a stale violation count."""
    articulation = scene.articulations["ordered"]
    logger = type(articulation).__module__
    original_limits = articulation.data.joint_pos_limits.torch.clone()
    original_defaults = articulation.data.default_joint_pos.torch.clone()
    limits = torch.zeros_like(original_limits)
    limits[..., 1] = 0.5
    try:
        for level in (logging.WARNING, logging.INFO):
            articulation.data.default_joint_pos.torch.fill_(1.0)
            caplog.clear()
            with caplog.at_level(level, logger=logger):
                articulation.write_joint_position_limit_to_sim_index(limits=limits, warn_limit_violation=False)
            assert [record.levelno for record in caplog.records] == ([logging.INFO] if level == logging.INFO else [])
            torch.testing.assert_close(articulation.data.default_joint_pos.torch, torch.full_like(limits[..., 1], 0.5))
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=logger):
            articulation.write_joint_position_limit_to_sim_mask(limits=limits, warn_limit_violation=False)
        assert not caplog.records
    finally:
        articulation.write_joint_position_limit_to_sim_index(limits=original_limits)
        articulation.data.default_joint_pos.torch.copy_(original_defaults)


##
# Operational-space control. The OSC island's six-DOF chain carries center-of-mass offsets and its joints are
# unpowered, so a wrong Jacobian, mass matrix, gravity force, or DoF ordering pushes the controller's steady-state
# error well past the bounds.
##


def _compute_ee_pose_root(robot: Articulation, ee_frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the end-effector position [m] and quaternion ``(x, y, z, w)`` in the root frame."""
    ee_pose_w = robot.data.body_pose_w.torch[:, ee_frame_idx]
    root_pose_w = robot.data.root_pose_w.torch
    return math_utils.subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )


def _compute_jacobian_root_frame(robot: Articulation, ee_jacobi_idx: int, arm_joint_ids: list[int]) -> torch.Tensor:
    """Return the end-effector Jacobian sliced to ``arm_joint_ids`` and rotated to the root frame, shape [N, 6, D]."""
    jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, arm_joint_ids]
    base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(robot.data.root_pose_w.torch[:, 3:7]))
    jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
    return jacobian


def _make_osc(device: str) -> OperationalSpaceController:
    """Return a fixed-impedance absolute-pose controller with inertial decoupling and no gravity compensation."""
    return OperationalSpaceController(
        OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=500.0,
            motion_damping_ratio_task=1.0,
        ),
        num_envs=NUM_ENVS,
        device=device,
    )


def _osc_chain(scene: _Scene) -> tuple[Articulation, int, int, list[int]]:
    """Step the OSC island once and return it with its end-effector body and Jacobian indices and arm joints."""
    robot = scene.articulations["osc"]
    scene.step("osc")
    ee_frame_idx = robot.find_bodies("Link_5")[0][0]
    # the fixed root has no Jacobian row
    return robot, ee_frame_idx, ee_frame_idx - 1, robot.find_joints(["Joint_.*"])[0]


def _run_osc(
    scene: _Scene, osc: OperationalSpaceController, target_pose_b: torch.Tensor, num_steps: int, *, gravity: bool
) -> tuple[list[float], list[float]]:
    """Close the OSC loop for ``num_steps`` steps; return the per-step max position and rotation errors."""
    robot, ee_frame_idx, ee_jacobi_idx, arm_joint_ids = _osc_chain(scene)
    pos_history: list[float] = []
    rot_history: list[float] = []
    for _ in range(num_steps):
        jacobian_b = _compute_jacobian_root_frame(robot, ee_jacobi_idx, arm_joint_ids)
        mass_matrix = robot.data.mass_matrix.torch[:, arm_joint_ids, :][:, :, arm_joint_ids]
        gravity_forces = robot.data.gravity_compensation_forces.torch[:, arm_joint_ids] if gravity else None
        ee_pos_b, ee_quat_b = _compute_ee_pose_root(robot, ee_frame_idx)
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
        # OSC's damping term ``kd * ee_vel_b`` needs the end-effector velocity ``J · q_dot``; a zero velocity
        # leaves the impedance undamped.
        joint_vel = robot.data.joint_vel.torch[:, arm_joint_ids]
        ee_vel_b = torch.bmm(jacobian_b, joint_vel.unsqueeze(-1)).squeeze(-1)

        osc.set_command(target_pose_b, current_ee_pose_b=ee_pose_b)
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            mass_matrix=mass_matrix,
            gravity=gravity_forces,
        )
        robot.actuators.target_command.set_effort_index(value=joint_efforts, joint_ids=arm_joint_ids)
        scene.step("osc")

        pos_error, rot_error = math_utils.compute_pose_error(
            ee_pos_b, ee_quat_b, target_pose_b[:, 0:3], target_pose_b[:, 3:7]
        )
        pos_history.append(pos_error.norm(dim=-1).max().item())
        rot_history.append(rot_error.norm(dim=-1).max().item())
    return pos_history, rot_history


@pytest.mark.isaacsim_ci
def test_osc_tracking_accuracy(scene: _Scene) -> None:
    """OSC pose tracking sentinel for the Jacobian and mass-matrix bridge.

    OSC runs with ``gravity_compensation=False`` and scene gravity disabled so the sentinel isolates the J/M
    bridge; the gravity-compensation path is covered by :func:`test_osc_gravity_compensation_precision`.
    ``inertial_dynamics_decoupling=True`` exercises ``mass_matrix`` and the COM-referenced J → M_b → J product.
    """
    robot, ee_frame_idx, _, _ = _osc_chain(scene)
    ee_pos_b, ee_quat_b = _compute_ee_pose_root(robot, ee_frame_idx)
    target_pose_b = torch.cat([ee_pos_b + ee_pos_b.new_tensor((0.05, 0.0, 0.0)), ee_quat_b], dim=-1)
    pos_history, rot_history = _run_osc(scene, _make_osc(scene.device), target_pose_b, 150, gravity=False)

    pos_mean = sum(pos_history[-100:]) / 100
    rot_mean = sum(rot_history[-100:]) / 100

    # Regression sentinel: assert on tail mean rather than min. With ``current_ee_vel_b = J · q_dot`` providing
    # OSC's damping term and no joint PD, the impedance settles to machine precision. A wrong J, wrong mass
    # matrix, or DoF mis-ordering pushes the steady-state error well past the 5 mm bound because OSC consumes
    # both ``body_link_jacobian_w`` and ``mass_matrix`` per step.
    assert pos_mean < 5e-3, f"OSC pos_mean {pos_mean:.5f} > 5 mm — bridge regression?"
    assert rot_mean < 5e-2, f"OSC rot_mean {rot_mean:.5f} > 0.05 rad — bridge regression?"


@pytest.mark.isaacsim_ci
def test_osc_gravity_compensation_precision(scene: _Scene) -> None:
    """Two-phase EE hold: gravity sag without compensation, tight hold with it.

    Same OSC pose-hold loop as :func:`test_osc_tracking_accuracy`, but with gravity on and the target pinned
    to the initial EE pose, so any steady-state error is pure gravity sag. Phase 1 runs with
    ``gravity_compensation=False`` and must sag past a floor; phase 2 flips ``osc.cfg.gravity_compensation``
    — read per :meth:`compute` call, so the flag is the only variable across phases (the gravity tensor is
    fetched and passed in both) — and must recover the hold to under 0.1 mm.

    The floor assertion keeps the test discriminating: if the task stiffness is ever raised high enough to
    mask gravity, phase 1 stops clearing the floor and the test fails loudly instead of silently passing on a
    non-discriminating setup. The gravity feed-forward consumes ``gravity_compensation_forces`` (Newton RNEA via
    ``eval_inverse_dynamics_passive``) live in the loop, covering the FK-staleness refresh on every step of
    phase 2. Both phases reach a true steady state, enforced by tail-half stationarity guards.
    """
    robot, ee_frame_idx, _, _ = _osc_chain(scene)
    osc = _make_osc(scene.device)
    # Hold the initial EE pose: phase-1 steady-state error is pure gravity sag.
    ee_pos_b, ee_quat_b = _compute_ee_pose_root(robot, ee_frame_idx)
    target_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    def _stationary_tail_mean(history: list[float], label: str) -> float:
        """Mean of the last 100 samples, asserting the two tail halves agree within 25%.

        The relative check carries a 10 µm absolute floor: at the solver noise floor of the compensated hold,
        tail jitter is far below the 0.1 mm verdict threshold and cannot flip the outcome.
        """
        a = sum(history[-100:-50]) / 50
        b = sum(history[-50:]) / 50
        mean = (a + b) / 2.0
        assert abs(a - b) < 0.25 * max(mean, 1e-5), (
            f"{label} not stationary: tail halves {a:.6f} vs {b:.6f} — extend the phase"
        )
        return mean

    with world_gravity((0.0, 0.0, -9.81)):
        hist_off, _ = _run_osc(scene, osc, target_pose_b, 200, gravity=True)
        osc.cfg.gravity_compensation = True
        hist_on, _ = _run_osc(scene, osc, target_pose_b, 200, gravity=True)

    pos_off = _stationary_tail_mean(hist_off, "phase-1 sag")
    pos_on = _stationary_tail_mean(hist_on, "phase-2 hold")

    assert pos_off > 1.2e-2, f"uncompensated sag {pos_off:.5f} < 1.2 cm — setup no longer discriminates gravity"
    assert pos_on < 1e-4, f"compensated hold {pos_on:.6f} > 0.1 mm — gravity compensation inaccurate"
    assert pos_on < pos_off / 10.0, f"compensation only improved sag {pos_off:.5f} -> {pos_on:.6f} (<10x)"
