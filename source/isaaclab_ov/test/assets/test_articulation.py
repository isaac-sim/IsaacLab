# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Real OVPhysX articulation coverage on one module-scoped composite scene per device.

Each device builds one scene of isolated articulation islands from the local branching fixture,
resets it once, and keeps it alive for every test that uses it. Each island belongs to one test
unless noted, so those tests do not depend on each other's order. Partial writes select a subset of
environments, joints, or bodies and prove the unselected entries keep their real backend state.

The CPU scene covers the host-resident property path, where CPU-only OVPhysX bindings are read and
written directly, and the host actuator path. The CUDA scene covers the rest, including pinned-host
staging of CPU-only bindings and the native Newton actuator path.

Host-side units (data caches, tendon scoping, kernels, actuator control) need no simulation and run first.
Initialization failures need their own scenes. A second simulation context cannot start while a shared
scene is alive, so these tests must run before any shared-scene test in the same session; they are
defined first and pytest runs them before it creates the module-scoped scenes.
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import Articulation, kernels  # noqa: E402
from isaaclab_ov.assets.articulation import actuator_control  # noqa: E402
from isaaclab_ov.assets.articulation.actuator_control import OvPhysxActuatorControl  # noqa: E402
from isaaclab_ov.assets.articulation.articulation_data import ArticulationData  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg, OvPhysxManager  # noqa: E402
from isaaclab_ov.test.fixtures.views import MockOvPhysxBindingSet  # noqa: E402
from isaaclab_physx.sim.schemas import PhysxJointCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.actuators import DelayedPDActuatorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, get_articulation_name_ordering  # noqa: E402
from isaaclab.envs.mdp import randomize_actuator_gains, randomize_rigid_body_material  # noqa: E402
from isaaclab.managers import EventTermCfg, SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context  # noqa: E402
from isaaclab.test.utils import DeviceScope, test_devices  # noqa: E402
from isaaclab.test.utils.articulation_ordering import (  # noqa: E402
    BRANCHING_MJWARP_BODY_NAMES,
    BRANCHING_MJWARP_JOINT_NAMES,
    BRANCHING_PHYSX_BODY_NAMES,
    BRANCHING_PHYSX_JOINT_NAMES,
)
from isaaclab.utils import replace  # noqa: E402
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache  # noqa: E402

from isaaclab_assets import FRANKA_PANDA_CFG  # noqa: E402

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"
_STIFFNESS, _DAMPING, _MAX_FORCE = 5.0, 0.5, 100.0
_ALL_DEVICES = test_devices()
_CUDA_DEVICES = test_devices(DeviceScope.CUDA)
# CUDA-only scene tests keep the CPU row as a skip so that pytest groups every test of one scene device together
# and builds each module-scoped scene once.
_CUDA_SCENES = [
    device if device in _CUDA_DEVICES else pytest.param(device, marks=pytest.mark.skip(reason="CUDA-only path"))
    for device in _ALL_DEVICES
]
_ROTATED_JOINT_NAMES = (*BRANCHING_PHYSX_JOINT_NAMES[1:], BRANCHING_PHYSX_JOINT_NAMES[0])
# The USD default max force (1e10) is effectively unlimited, so the limit islands author a finite one.
_LIMIT_DRIVE_PROPS = (sim_utils.UsdPhysicsDriveCfg(max_force=80.0), PhysxJointCfg(max_joint_velocity=5.0))
_ROOT_PRESERVING_REVERSED_BODY_NAMES = (BRANCHING_PHYSX_BODY_NAMES[0], *reversed(BRANCHING_PHYSX_BODY_NAMES[1:]))


def _implicit(**kwargs) -> dict[str, ImplicitActuatorCfg]:
    return {"joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=_STIFFNESS, damping=_DAMPING, **kwargs)}


@dataclass(frozen=True)
class _IslandCfg:
    """Authoring recipe for one isolated articulation island."""

    actuators: dict
    num_envs: int = 2
    fixed_base: bool = True
    joint_ordering: tuple[str, ...] | str | None = None
    body_ordering: tuple[str, ...] | str | None = None
    drives: bool = True
    joint_drive_props: tuple | None = None
    y_axis_joints: bool = False
    com_offset: float | None = None
    distinct_coms: bool = False
    reversed_elbow: bool = False
    collision_shapes: bool = False
    tendons: bool = False
    articulation_root_prim_path: str | None = None
    devices: DeviceScope = DeviceScope.ALL


_ISLANDS: dict[str, _IslandCfg] = {
    # Nonidentity MJWarp order through cross-backend discovery; shared by the state, property,
    # material, and FK-refresh tests, which each check deltas against their own starting state.
    "ordered": _IslandCfg(_implicit(), joint_ordering="mjwarp", body_ordering="mjwarp", collision_shapes=True),
    "drive": _IslandCfg(_implicit(), joint_ordering=_ROTATED_JOINT_NAMES, devices=DeviceScope.CUDA),
    "dynamics": _IslandCfg(
        {},
        joint_ordering="mjwarp",
        body_ordering="mjwarp",
        drives=False,
        y_axis_joints=True,
        com_offset=0.2,
        devices=DeviceScope.CUDA,
    ),
    "reversed": _IslandCfg(
        {}, num_envs=1, drives=False, y_axis_joints=True, com_offset=0.2, reversed_elbow=True, devices=DeviceScope.CUDA
    ),
    "reversed_ordered": _IslandCfg(
        {},
        num_envs=1,
        joint_ordering=BRANCHING_MJWARP_JOINT_NAMES,
        body_ordering=BRANCHING_MJWARP_BODY_NAMES,
        drives=False,
        y_axis_joints=True,
        com_offset=0.2,
        reversed_elbow=True,
        devices=DeviceScope.CUDA,
    ),
    "floating": _IslandCfg(
        _implicit(),
        fixed_base=False,
        joint_ordering=tuple(reversed(BRANCHING_PHYSX_JOINT_NAMES)),
        devices=DeviceScope.CUDA,
    ),
    "tendon": _IslandCfg(_implicit(), tendons=True, devices=DeviceScope.CUDA),
    "native": _IslandCfg(
        {
            "joints": IdealPDActuatorCfg(
                joint_names_expr=[".*"], stiffness=_STIFFNESS, damping=_DAMPING, actuator_effort_limit=_MAX_FORCE
            )
        },
        joint_ordering=_ROTATED_JOINT_NAMES,
        devices=DeviceScope.CUDA,
    ),
    "delayed": _IslandCfg(
        {
            "joint": DelayedPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=20.0,
                damping=1.0,
                actuator_effort_limit=80.0,
                min_delay=1,
                max_delay=1,
            )
        },
        devices=DeviceScope.CUDA,
    ),
    # Implicit shoulders beside explicit elbows; the CPU scene keeps the explicit group on the host path.
    "mixed": _IslandCfg(
        {
            "cart": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder"], joint_effort_limit=400.0, stiffness=20.0, damping=0.0
            ),
            "pole": IdealPDActuatorCfg(
                joint_names_expr=[".*_elbow"], stiffness=20.0, damping=0.0, actuator_effort_limit=400.0
            ),
        },
        joint_ordering=_ROTATED_JOINT_NAMES,
    ),
    "staging": _IslandCfg({}, num_envs=4, drives=False, devices=DeviceScope.CUDA),
    # Unset gains, so the actuators adopt the gains of the authored drives.
    "usd_gains": _IslandCfg(
        {"joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
        devices=DeviceScope.CUDA,
    ),
    # A floating articulation whose root API sits on a rigid body below the asset prim.
    "non_root": _IslandCfg(
        _implicit(), fixed_base=False, articulation_root_prim_path="/base", devices=DeviceScope.CUDA
    ),
    "com_order": _IslandCfg(
        _implicit(), num_envs=1, body_ordering=_ROOT_PRESERVING_REVERSED_BODY_NAMES, distinct_coms=True
    ),
    # The authored limits are not the defaults, so solver values equal to them prove the USD path.
    "limits_usd": _IslandCfg(_implicit(), num_envs=1, drives=False, joint_drive_props=_LIMIT_DRIVE_PROPS),
    "limits_cfg": _IslandCfg(
        _implicit(joint_velocity_limit=1e5, joint_effort_limit=1e5),
        num_envs=1,
        drives=False,
        joint_drive_props=_LIMIT_DRIVE_PROPS,
    ),
}


@dataclass
class _ArticulationScene:
    """Articulation islands that share one real OVPhysX lifecycle."""

    sim: SimulationContext
    device: str
    native_actuators: bool
    islands: dict[str, Articulation]

    def step(self, articulation: Articulation, count: int = 1) -> None:
        """Step the shared scene while writing and refreshing one island."""
        for _ in range(count):
            articulation.write_data_to_sim()
            self.sim.step()
            articulation.update(self.sim.cfg.dt)


def _apply_api_schema(prim, schema_name: str) -> None:
    """Author an applied-schema token without loading the Kit PhysX schema module."""
    schemas = list(prim.GetAppliedSchemas())
    schemas.append(schema_name)
    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = schemas
    prim.SetMetadata("apiSchemas", api_schemas)


def _author_tendons(stage, prim_path: str) -> None:
    """Author one spatial tendon and one fixed tendon per arm."""
    root_prim = stage.GetPrimAtPath(f"{prim_path}/base")
    _apply_api_schema(root_prim, "PhysxTendonAttachmentRootAPI:root")
    root_prim.CreateAttribute("physxTendon:root:localPos", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(0.0))
    root_prim.CreateAttribute("physxTendon:root:stiffness", Sdf.ValueTypeNames.Float).Set(5.0)
    root_prim.CreateAttribute("physxTendon:root:damping", Sdf.ValueTypeNames.Float).Set(0.5)
    root_prim.CreateAttribute("physxTendon:root:limitStiffness", Sdf.ValueTypeNames.Float).Set(1.0)
    root_prim.CreateAttribute("physxTendon:root:offset", Sdf.ValueTypeNames.Float).Set(0.0)
    leaf_prim = stage.GetPrimAtPath(f"{prim_path}/left_tip")
    _apply_api_schema(leaf_prim, "PhysxTendonAttachmentLeafAPI:leaf")
    leaf_prim.CreateAttribute("physxTendon:leaf:localPos", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(0.0))
    leaf_prim.CreateAttribute("physxTendon:leaf:parentAttachment", Sdf.ValueTypeNames.Token).Set("root")
    leaf_prim.CreateRelationship("physxTendon:leaf:parentLink").SetTargets([root_prim.GetPath()])
    leaf_prim.CreateAttribute("physxTendon:leaf:restLength", Sdf.ValueTypeNames.Float).Set(0.5)
    leaf_prim.CreateAttribute("physxTendon:leaf:lowerLimit", Sdf.ValueTypeNames.Float).Set(0.0)
    leaf_prim.CreateAttribute("physxTendon:leaf:upperLimit", Sdf.ValueTypeNames.Float).Set(2.0)

    for index, side in enumerate(("left", "right")):
        instance = f"t{index}"
        shoulder = stage.GetPrimAtPath(f"{prim_path}/{side}_shoulder")
        _apply_api_schema(shoulder, f"PhysxTendonAxisRootAPI:{instance}")
        for name, value in (
            ("stiffness", 10.0),
            ("damping", 1.0),
            ("limitStiffness", 0.0),
            ("offset", 0.0),
            ("restLength", 0.2 + 0.1 * index),
            ("lowerLimit", -1.0),
            ("upperLimit", 1.0),
        ):
            shoulder.CreateAttribute(f"physxTendon:{instance}:{name}", Sdf.ValueTypeNames.Float).Set(value)
        shoulder.CreateAttribute(f"physxTendon:{instance}:gearing", Sdf.ValueTypeNames.FloatArray).Set([1.0])
        elbow = stage.GetPrimAtPath(f"{prim_path}/{side}_elbow")
        _apply_api_schema(elbow, f"PhysxTendonAxisAPI:{instance}")
        elbow.CreateAttribute(f"physxTendon:{instance}:gearing", Sdf.ValueTypeNames.FloatArray).Set([1.0])


def _spawn_island(name: str, cfg: _IslandCfg, y_offset: float) -> Articulation:
    """Spawn one articulation island and author its USD variations."""
    island_path = f"/World/{name}"
    sim_utils.create_prim(island_path, "Xform", translation=(0.0, y_offset, 0.0))
    for env_index in range(cfg.num_envs):
        sim_utils.create_prim(f"{island_path}/Env_{env_index}", "Xform", translation=(3.0 * env_index, 0.0, 0.0))
    articulation = Articulation(
        ArticulationCfg(
            prim_path=f"{island_path}/Env_[^/]*/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(_FIXTURE),
                joint_drive_props=None if cfg.joint_drive_props is None else list(cfg.joint_drive_props),
            ),
            actuators=cfg.actuators,
            joint_ordering=cfg.joint_ordering,
            body_ordering=cfg.body_ordering,
            articulation_root_prim_path=cfg.articulation_root_prim_path,
        )
    )
    stage = sim_utils.get_current_stage()
    for env_index in range(cfg.num_envs):
        prim_path = f"{island_path}/Env_{env_index}/Robot"
        if cfg.articulation_root_prim_path is not None:
            stage.GetPrimAtPath(prim_path).RemoveAPI(UsdPhysics.ArticulationRootAPI)
            UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(prim_path + cfg.articulation_root_prim_path))
        for joint_name in BRANCHING_PHYSX_JOINT_NAMES:
            joint_prim = stage.GetPrimAtPath(f"{prim_path}/{joint_name}")
            if cfg.drives:
                drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                drive.CreateStiffnessAttr(_STIFFNESS)
                drive.CreateDampingAttr(_DAMPING)
                drive.CreateMaxForceAttr(_MAX_FORCE)
            if cfg.y_axis_joints:
                UsdPhysics.RevoluteJoint(joint_prim).GetAxisAttr().Set("Y")
        for body_index, body_name in enumerate(BRANCHING_PHYSX_BODY_NAMES):
            mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"{prim_path}/{body_name}"))
            if cfg.com_offset is not None:
                mass.CreateCenterOfMassAttr(Gf.Vec3f(cfg.com_offset, 0.0, 0.0))
                # Isotropic inertia makes the energy check independent of body rotation.
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1))
            if cfg.distinct_coms:
                # Non-identity principal axes make the native setter's float32 normalization observable.
                mass.CreateCenterOfMassAttr(Gf.Vec3f(0.01 * body_index, -0.02, 0.03))
                mass.CreatePrincipalAxesAttr(Gf.Quatf(0.9, 0.1 * body_index, 0.2, -0.3).GetNormalized())
        if cfg.reversed_elbow:
            joint = UsdPhysics.RevoluteJoint.Get(stage, f"{prim_path}/left_elbow")
            body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            joint.GetBody0Rel().SetTargets(body1)
            joint.GetBody1Rel().SetTargets(body0)
        if cfg.fixed_base:
            fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/fixed_root")
            fixed_joint.GetBody1Rel().SetTargets([f"{prim_path}/base"])
        if cfg.collision_shapes:
            # Offset along the joint axes so the shapes never meet while the joints rotate.
            for body_name, height in (("left_tip", 0.3), ("right_tip", -0.3)):
                shape = UsdGeom.Cube.Define(stage, f"{prim_path}/{body_name}/shape")
                shape.CreateSizeAttr(0.05)
                shape.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, height))
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        if cfg.tendons:
            _author_tendons(stage, prim_path)
    return articulation


@pytest.fixture(scope="module")
def scene(request: pytest.FixtureRequest) -> Iterator[_ArticulationScene]:
    """Initialize every articulation island for one device once for this module."""
    device = request.param
    device_scope = DeviceScope.CUDA if device.startswith("cuda") else DeviceScope.CPU
    # The native Newton actuator runtime requires CUDA; the CPU scene keeps explicit actuators on the host.
    native_actuators = device.startswith("cuda")
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, use_newton_actuators=native_actuators)
    with build_simulation_context(device=device, sim_cfg=sim_cfg, auto_add_lighting=False) as sim:
        islands = {
            name: _spawn_island(name, cfg, y_offset=3.0 * index)
            for index, (name, cfg) in enumerate(_ISLANDS.items())
            if cfg.devices & device_scope
        }
        sim.reset()
        yield _ArticulationScene(sim=sim, device=device, native_actuators=native_actuators, islands=islands)


def _read_binding_to_torch(articulation: Articulation, tensor_type: int, device: str | torch.device) -> torch.Tensor:
    """Read an OVPhysX attribute on its native device and move it to *device*."""
    return wp.to_torch(articulation.root_view.get_attribute(tensor_type)).to(device)


def _spawn_own_articulation(**cfg) -> Articulation:
    """Spawn one branching articulation for a test that needs its own scene."""
    return Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(_FIXTURE), joint_drive_props=list(_LIMIT_DRIVE_PROPS)),
            actuators=_implicit(),
            **cfg,
        )
    )


##
# Host-side units: data caches, joint directions, tendon scoping, kernels, and actuator control.
# They need no simulation, so they run before the scenes below.
##


def test_cached_read_launches_reset_on_ordering_and_invalidation():
    """Ordering installation and simulation invalidation should discard recorded reads."""

    class MinimalData(ArticulationData):
        def __dir__(self):
            return []

    class Buffer:
        timestamp = 1.0

    data = MinimalData.__new__(MinimalData)
    read_launch_cache = Mock()
    data._read_launch_cache = read_launch_cache
    data._configure_ordering_buffers = lambda: None
    data._make_jacobian_body_user_to_backend = lambda: object()
    data.joint_ordering = None
    data._body_com_jacobian_w = Buffer()
    data._mass_matrix = Buffer()
    data._gravity_compensation_forces = Buffer()

    data._apply_ordering_maps_after_resolve()

    read_launch_cache.clear.assert_called_once_with()
    assert data._body_com_jacobian_w.timestamp == -1.0
    assert data._mass_matrix.timestamp == -1.0
    assert data._gravity_compensation_forces.timestamp == -1.0

    data._is_primed = True
    data._sim_timestamp = 1.0
    data._invalidate_initialize_callback(None)

    assert read_launch_cache.clear.call_count == 2
    assert data._is_primed is False
    assert data._sim_timestamp == 0.0


def test_static_property_reads_are_not_invalidated_by_simulation_steps():
    """Joint properties and body mass/inertia should be read once per invalidation, not once per step.

    On OVPhysX these are blocking CPU-only binding reads whose cost scales with the number of
    environments, so re-reading them every physics step made per-step consumers (such as the
    native actuator telemetry sync) host-bound. State buffers must still refresh every step.
    """

    class Buffer:
        def __init__(self, shape):
            self.data = wp.zeros(shape, dtype=wp.float32, device="cpu")
            self.timestamp = -1.0

    data = ArticulationData.__new__(ArticulationData)
    data.device = "cpu"
    data.num_instances = 1
    data.num_joints = 2
    data._sim_timestamp = 1.0
    data.body_ordering = None
    data._get_binding = lambda tensor_type: object()
    reads: list[int] = []
    data._binding_read = lambda tensor_type, dst: reads.append(tensor_type)

    # Joint properties: one read across several steps, one more after explicit invalidation
    # (simulation reinitialization).
    data.joint_ordering = None
    stiffness = Buffer((1, 2))
    for _ in range(3):
        data._sim_timestamp += 1.0
        data._read_joint_property_binding(TT.DOF_STIFFNESS, stiffness, None)
    assert reads.count(TT.DOF_STIFFNESS) == 1
    stiffness.timestamp = -1.0
    data._read_joint_property_binding(TT.DOF_STIFFNESS, stiffness, None)
    assert reads.count(TT.DOF_STIFFNESS) == 2

    # Body properties behave the same; body state buffers still refresh every step.
    mass, link_pose = Buffer((1, 2)), Buffer((1, 2))
    for _ in range(3):
        data._sim_timestamp += 1.0
        data._refresh_reordered_body_buffer(mass, None, TT.BODY_MASS, static=True)
        data._refresh_reordered_body_buffer(link_pose, None, TT.LINK_POSE)
    assert reads.count(TT.BODY_MASS) == 1
    assert reads.count(TT.LINK_POSE) == 3
    mass.timestamp = -1.0
    data._refresh_reordered_body_buffer(mass, None, TT.BODY_MASS, static=True)
    assert reads.count(TT.BODY_MASS) == 2

    # Under a non-identity joint ordering the property is gathered once, then served from cache.
    data.joint_ordering = SimpleNamespace(user_to_backend=wp.array([1, 0], dtype=wp.int32, device="cpu"))
    data._read_launch_cache = _WarpLaunchCache("cpu")
    user_buffer, backend_buffer = Buffer((1, 2)), Buffer((1, 2))
    data._binding_read = lambda tensor_type, dst: (reads.append(tensor_type), dst.assign([[1.0, 2.0]]))
    for _ in range(3):
        data._sim_timestamp += 1.0
        data._read_joint_property_binding(TT.DOF_DAMPING, user_buffer, backend_buffer)
    assert reads.count(TT.DOF_DAMPING) == 1
    torch.testing.assert_close(wp.to_torch(user_buffer.data), torch.tensor([[2.0, 1.0]]))


def test_joint_dof_sign_resolution_traverses_instance_proxies():
    """Resolve reversed joints inside an instanceable articulation."""
    source_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source_stage, "/Robot")
    UsdGeom.Xform.Define(source_stage, "/Robot/base")
    UsdGeom.Xform.Define(source_stage, "/Robot/link")
    joint = UsdPhysics.RevoluteJoint.Define(source_stage, "/Robot/joint")
    joint.GetBody0Rel().SetTargets(["/Robot/link"])
    joint.GetBody1Rel().SetTargets(["/Robot/base"])
    stage = Usd.Stage.CreateInMemory()
    instance = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    instance.GetReferences().AddReference(source_stage.GetRootLayer().identifier, "/Robot")
    instance.SetInstanceable(True)

    articulation = Mock(
        cfg=Mock(prim_path="/World/Robot"),
        _joint_names=["joint"],
        _body_names=["base", "link"],
    )

    assert Articulation._resolve_joint_dof_signs(articulation, stage) == (-1,)


def _define_tendon_joint(stage: Usd.Stage, path: str, schema_name: str) -> None:
    """Define a revolute joint prim with a tendon schema marker."""
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    schemas = Sdf.TokenListOp()
    schemas.explicitItems = [schema_name]
    joint.GetPrim().SetMetadata("apiSchemas", schemas)


def _make_articulation_root_stage_usda() -> str:
    """Serialize one relevant articulation subtree and unrelated joints in memory."""
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/envs", "Xform")
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    stage.DefinePrim("/World/envs/env_0/Robot/root", "Xform")
    stage.DefinePrim("/World/unrelated", "Xform")

    _define_tendon_joint(
        stage,
        "/World/envs/env_0/Robot/root/fixed_joint",
        "PhysxTendonAxisRootAPI:inst0",
    )
    _define_tendon_joint(
        stage,
        "/World/envs/env_0/Robot/root/spatial_joint",
        "PhysxTendonAttachmentRootAPI:inst0",
    )
    _define_tendon_joint(
        stage,
        "/World/unrelated/unrelated_fixed_joint",
        "PhysxTendonAxisRootAPI:inst0",
    )
    _define_tendon_joint(
        stage,
        "/World/unrelated/unrelated_spatial_joint",
        "PhysxTendonAttachmentLeafAPI:inst0",
    )

    return stage.Flatten().ExportToString()


def _make_articulation_shell() -> Articulation:
    """Create a minimal ovphysx articulation shell for tendon processing tests."""
    articulation = object.__new__(Articulation)
    bindings = MockOvPhysxBindingSet(
        num_instances=1,
        num_joints=2,
        num_bodies=2,
        num_fixed_tendons=1,
        num_spatial_tendons=1,
    )
    # The migrated Articulation reads tendon counts off its OvPhysxView; inject the mock
    # view over these bindings so the metadata passthrough resolves without a real view.
    object.__setattr__(articulation, "_root_view", bindings.view)
    object.__setattr__(articulation, "_articulation_root_path", "/World/envs/env_0/Robot/root")
    object.__setattr__(articulation, "_initialize_handle", None)
    object.__setattr__(articulation, "_invalidate_initialize_handle", None)
    object.__setattr__(articulation, "_prim_deletion_handle", None)
    object.__setattr__(articulation, "_debug_vis_handle", None)
    object.__setattr__(
        articulation,
        "_data",
        SimpleNamespace(
            _num_fixed_tendons=0,
            _num_spatial_tendons=0,
            fixed_tendon_names=[],
            spatial_tendon_names=[],
        ),
    )
    return articulation


def test_process_tendons_scopes_to_articulation_root():
    """Tendon discovery should ignore joints that live outside the current articulation subtree."""
    articulation = _make_articulation_shell()
    stage_usda = _make_articulation_root_stage_usda()
    old_stage_usda = OvPhysxManager._stage_usda
    OvPhysxManager._stage_usda = stage_usda
    try:
        articulation._process_tendons()
    finally:
        OvPhysxManager._stage_usda = old_stage_usda

    # the tendon is reported by its schema INSTANCE name, matching PhysX; scope leakage would
    # add the identically-named instance from /World/unrelated, giving two entries
    assert articulation.fixed_tendon_names == ["inst0"]
    assert articulation.spatial_tendon_names == ["spatial_joint"]


def _selector(values: list[int], dtype: type) -> wp.array:
    return wp.array(values, dtype=dtype, device="cpu")


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
def test_root_worker_accepts_selector_widths(env_dtype: type) -> None:
    env_ids = _selector([1, 0], env_dtype)
    data = wp.array(
        [[11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=wp.transformf,
        device="cpu",
    )
    output = wp.zeros(2, dtype=wp.transformf, device="cpu")
    sim_env_ids = wp.empty(2, dtype=wp.int32, device="cpu")
    kernel = kernels.set_root_link_pose_to_sim_index
    if env_dtype == wp.int64:
        kernel = kernels.set_root_link_pose_to_sim_index_kernel(env_ids)

    wp.launch(kernel, dim=2, inputs=[data, env_ids], outputs=[output, sim_env_ids], device="cpu")

    np.testing.assert_array_equal(output.numpy(), data.numpy()[[1, 0]])
    np.testing.assert_array_equal(sim_env_ids.numpy(), [1, 0])


@pytest.mark.parametrize(("env_dtype", "item_dtype"), [(wp.int32, wp.int32), (wp.int64, wp.int64)])
def test_item_worker_accepts_selector_widths(env_dtype: type, item_dtype: type) -> None:
    env_ids = _selector([1, 0], env_dtype)
    item_ids = _selector([2, 0], item_dtype)
    data = wp.array([[11.0, 12.0], [21.0, 22.0]], dtype=wp.float32, device="cpu")
    output = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    kernel = kernels.write_2d_data_to_buffer_with_indices
    if env_dtype != wp.int32 or item_dtype != wp.int32:
        kernel = kernels.write_2d_data_to_buffer_with_indices_kernel(env_ids, item_ids)

    wp.launch(kernel, dim=(2, 2), inputs=[data, env_ids, item_ids], outputs=[output], device="cpu")

    np.testing.assert_array_equal(output.numpy(), [[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]])


def test_prepare_native_actuators_leaves_implicit_only_articulation_on_standard_path(monkeypatch):
    """Keep implicit-only articulations on the unchanged solver-drive path."""
    runtime_prepare_calls = []
    runtime = SimpleNamespace(
        prepare=lambda *args, **kwargs: runtime_prepare_calls.append(True), wrapper=None, adapter=None
    )
    articulation = SimpleNamespace(
        _sim_cfg=SimpleNamespace(use_newton_actuators=True),
        cfg=SimpleNamespace(prim_path="/World/Robot"),
    )
    monkeypatch.setattr(actuator_control, "PhysxActuatorRuntime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(actuator_control, "find_first_matching_prim", lambda _: None)

    control = OvPhysxActuatorControl(articulation)
    native_groups = control.prepare_native_actuators(
        collection=None,
        actuator_cfgs={"implicit": ImplicitActuatorCfg(joint_names_expr=["joint"], stiffness=10.0, damping=1.0)},
    )

    assert native_groups == set()
    assert not control.native_actuator_path_active
    assert not articulation._has_newton_actuators
    assert runtime_prepare_calls == []


@pytest.mark.parametrize(
    "module_name",
    [
        "isaaclab_physx.assets.articulation.actuator_control",
        "isaaclab_ov.assets.articulation.actuator_control",
    ],
)
def test_host_actuator_control_import_does_not_probe_optional_newton_runtime(monkeypatch, module_name):
    """Import host controls without probing an unrequested Newton optional dependency."""
    original_find_spec = importlib.util.find_spec

    def reject_newton_probe(name, *args, **kwargs):
        if name.startswith("isaaclab_newton"):
            raise AssertionError("host actuator-control import eagerly probed Newton")
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", reject_newton_probe)
    importlib.reload(importlib.import_module(module_name))


##
# Tests that own their simulation context. Keep them above the composite scene.
##


@pytest.mark.parametrize("device", _CUDA_DEVICES)
def test_mimic_finger_follows_commanded_finger(device: str) -> None:
    """The passive Franka finger follows its driven leader after a position command.

    Lab submits the same host-side targets on both devices; the mimic constraint itself is solved by PhysX.
    """
    cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, gravity=(0.0, 0.0, 0.0), use_newton_actuators=False)
    with build_simulation_context(device=device, sim_cfg=cfg) as sim:
        articulation = Articulation(replace(FRANKA_PANDA_CFG, prim_path="/World/Franka"))
        sim.reset()
        leader_id = articulation.find_joints("panda_finger_joint1")[0][0]
        follower_id = articulation.find_joints("panda_finger_joint2")[0][0]
        initial_leader = articulation.data.joint_pos.torch[:, leader_id].clone()
        target = articulation.data.joint_pos.torch.clone()
        target[:, [leader_id, follower_id]] = 0.01
        articulation.actuators.target_command.set_position_index(value=target)
        for _ in range(50):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(sim.cfg.dt)
        leader = articulation.data.joint_pos.torch[:, leader_id]
        follower = articulation.data.joint_pos.torch[:, follower_id]
        assert torch.all(torch.abs(leader - initial_leader) > 0.005)
        torch.testing.assert_close(follower, leader, rtol=0.0, atol=5.0e-4)


@pytest.mark.parametrize(
    ("device", "variant"),
    # CUDA replays native clones for every compatible variant; CPU serializes the full stage, one variant.
    [
        (device, variant)
        for device in _CUDA_DEVICES
        for variant in ("geometry", "joint_type", "d6_translation", "disabled_joint")
    ]
    + [(device, "geometry") for device in test_devices(DeviceScope.CPU)]
    # The cloner rejects incompatible layouts from USD before any device-specific physics.
    + [(_ALL_DEVICES[0], variant) for variant in ("d6_rotation", "fixed_tendon")],
)
def test_heterogeneous_articulation_clone_indexed_state(device, variant, tmp_path):
    """Compatible variants preserve indexed state; incompatible action/tendon layouts are rejected."""
    variants = []
    for shape in (UsdGeom.Cube, UsdGeom.Sphere):
        stage = Usd.Stage.CreateInMemory()
        robot = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
        stage.SetDefaultPrim(robot)
        UsdPhysics.ArticulationRootAPI.Apply(robot)
        for name in ("Base", "Tip"):
            body = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
            body.AddTranslateOp().Set((0.0, 0.0, 0.4 if name == "Tip" else 0.0))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
            geometry = shape.Define(stage, f"/Robot/{name}/Geometry")
            geometry.AddScaleOp().Set((0.1, 0.1, 0.1))
            UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
        fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/Fixed")
        fixed.CreateBody1Rel().SetTargets(["/Robot/Base"])
        joint_schema = UsdPhysics.RevoluteJoint
        if variant.startswith("d6_"):
            joint_schema = UsdPhysics.Joint
        elif variant == "joint_type" and shape is UsdGeom.Sphere:
            joint_schema = UsdPhysics.PrismaticJoint
        joint = joint_schema.Define(stage, "/Robot/Joint")
        joint.CreateBody0Rel().SetTargets(["/Robot/Base"])
        joint.CreateBody1Rel().SetTargets(["/Robot/Tip"])
        joint.CreateLocalPos0Attr((0.0, 0.0, 0.4))
        if variant == "disabled_joint":
            joint = UsdPhysics.Joint.Define(stage, "/Robot/Disabled")
            joint.CreateBody0Rel().SetTargets(["/Robot/Base"])
            joint.CreateBody1Rel().SetTargets(["/Robot/Tip"])
            joint.CreateLocalPos0Attr((0.0, 0.0, 0.4))
            joint.CreateJointEnabledAttr(False)
        if variant.startswith("d6_") or variant == "disabled_joint":
            free_axis = "rotY" if variant != "d6_translation" and shape is UsdGeom.Sphere else "rotX"
            for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
                if axis == free_axis or (axis == "transX" and variant == "d6_translation" and shape is UsdGeom.Sphere):
                    continue
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr(1.0)
                limit.CreateHighAttr(-1.0)
        elif variant == "fixed_tendon" and shape is UsdGeom.Sphere:
            prim = joint.GetPrim()
            prim.AddAppliedSchema("PhysxTendonAxisRootAPI:tendon")
            prim.CreateAttribute("physxTendon:tendon:stiffness", Sdf.ValueTypeNames.Float).Set(123.0)
            prim.CreateAttribute("physxTendon:tendon:gearing", Sdf.ValueTypeNames.FloatArray).Set([1.0])
        path = str(tmp_path / f"{shape.__name__}.usda")
        stage.Export(path)
        variants.append(sim_utils.UsdFileCfg(usd_path=path))

    with build_simulation_context(
        device=device, sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device=device, gravity=(0.0, 0.0, 0.0))
    ) as sim:
        cfg = InteractiveSceneCfg(num_envs=6, env_spacing=2.0)
        cfg.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=variants),
            actuators={"joint": ImplicitActuatorCfg(joint_names_expr=["Joint.*"], stiffness=0.0, damping=0.0)},
        )
        if variant in ("d6_rotation", "fixed_tendon"):
            with pytest.raises(ValueError, match="incompatible.*(rotation axes|tendon)"):
                InteractiveScene(cfg)
            return
        scene = InteractiveScene(cfg)
        if variant == "joint_type":
            with pytest.raises(RuntimeError, match="heterogeneous or empty view"):
                sim.reset()
            return
        sim.reset()
        robot = scene["robot"]
        assert robot.num_joints == 1
        assert robot.num_bodies == 2
        assert robot.root_view.prim_paths == [f"/World/envs/env_{i}/Robot" for i in range(scene.num_envs)]
        torch.testing.assert_close(robot.data.root_pos_w.torch, scene.env_origins)
        selected = torch.tensor([5, 3], device=device)
        positions = torch.tensor([[0.25], [-0.4]], device=device)
        robot.write_joint_state_to_sim(positions, torch.zeros_like(positions), env_ids=selected)
        sim.step()
        scene.update(sim.get_physics_dt())
        expected = torch.zeros((scene.num_envs, 1), device=device)
        expected[selected] = positions
        torch.testing.assert_close(robot.data.joint_pos.torch, expected, atol=1e-4, rtol=0.0)
        for env_id in range(scene.num_envs):
            binding = sim.physics_manager.get_physx_instance().create_tensor_binding(
                pattern=f"/World/envs/env_{env_id}/Robot", tensor_type=TT.DOF_POSITION
            )
            try:
                actual = torch.empty(binding.shape, device=device)
                binding.read(actual)
                torch.testing.assert_close(actual, expected[env_id : env_id + 1], atol=1e-4, rtol=0.0)
            finally:
                binding.destroy()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
@pytest.mark.parametrize(
    ("init_state", "match"),
    [
        (ArticulationCfg.InitialStateCfg(joint_pos={"left_shoulder": 10.0}), "default positions out of the limits"),
        (ArticulationCfg.InitialStateCfg(joint_vel={"left_shoulder": 100.0}), "default velocities out of the limits"),
    ],
    ids=["position", "velocity"],
)
def test_out_of_range_default_joint_state(device: str, init_state: ArticulationCfg.InitialStateCfg, match: str) -> None:
    """Reject default joint positions outside the joint limits and default velocities above the velocity limits."""
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device)
    with build_simulation_context(device=device, sim_cfg=sim_cfg, auto_add_lighting=False) as sim:
        articulation = _spawn_own_articulation(init_state=init_state)
        joint = UsdPhysics.RevoluteJoint.Get(sim.stage, "/World/Robot/left_shoulder")
        joint.CreateLowerLimitAttr(-90.0)
        joint.CreateUpperLimitAttr(90.0)
        with pytest.raises(ValueError, match=match):
            sim.reset()
        assert not articulation.is_initialized


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_setting_invalid_articulation_root_prim_path(device: str) -> None:
    """Reject an explicit articulation root path that matches no prim."""
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device)
    with build_simulation_context(device=device, sim_cfg=sim_cfg, auto_add_lighting=False) as sim:
        articulation = _spawn_own_articulation(articulation_root_prim_path="/non_existing_prim_path")
        with pytest.raises(RuntimeError, match="Failed to find articulation root prim"):
            sim.reset()
        assert not articulation.is_initialized


@pytest.mark.parametrize("scene", _ALL_DEVICES, indirect=True)
def test_articulation_initialization_and_partial_state(scene: _ArticulationScene) -> None:
    """Prove ordering and indexed state writes against the real OVPhysX view."""
    articulation = scene.islands["ordered"]
    device = scene.device
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert tuple(articulation.joint_names) == BRANCHING_MJWARP_JOINT_NAMES
    assert tuple(articulation.body_names) == BRANCHING_MJWARP_BODY_NAMES
    assert articulation.joint_ordering is not None
    assert articulation.body_ordering is not None
    assert articulation.num_instances == 2

    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    joint_ids = torch.tensor([articulation.num_joints - 1, 0], dtype=torch.int32, device=device)
    target_position = torch.tensor([[0.21, -0.13]], device=device)
    target_velocity = torch.tensor([[0.41, -0.23]], device=device)
    expected_position = articulation.data.joint_pos.torch.clone()
    expected_velocity = articulation.data.joint_vel.torch.clone()
    expected_position[env_ids[:, None], joint_ids] = target_position
    expected_velocity[env_ids[:, None], joint_ids] = target_velocity
    articulation.write_joint_state_to_sim_index(
        position=target_position,
        velocity=target_velocity,
        env_ids=env_ids,
        joint_ids=joint_ids,
    )
    torch.testing.assert_close(articulation.data.joint_pos.torch, expected_position)
    torch.testing.assert_close(articulation.data.joint_vel.torch, expected_velocity)
    backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute(TT.DOF_POSITION)),
        expected_position[:, backend_to_user],
    )

    # A position-only partial write (a distinct kernel) preserves every unselected backend joint.
    backend_before = wp.to_torch(articulation.root_view.get_attribute(TT.DOF_POSITION)).clone()
    backend_joint_id = articulation.joint_ordering.user_to_backend_indices[1]
    selected_value = backend_before[0, backend_joint_id] + 0.001
    articulation.write_joint_position_to_sim_index(
        position=selected_value.reshape(1, 1),
        env_ids=wp.array([0], dtype=wp.int32, device=device),
        joint_ids=wp.array([1], dtype=wp.int32, device=device),
    )
    expected_backend = backend_before.clone()
    expected_backend[0, backend_joint_id] = selected_value
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute(TT.DOF_POSITION)), expected_backend, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("scene", _ALL_DEVICES, indirect=True)
def test_articulation_joint_and_body_properties_round_trip(scene: _ArticulationScene) -> None:
    """Prove selected joint and body properties reach real OVPhysX bindings and public getters."""
    articulation = scene.islands["ordered"]
    device = scene.device
    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    joint_ids = torch.tensor([articulation.num_joints - 1, 0], dtype=torch.int32, device=device)
    backend_friction_before = wp.to_torch(articulation.root_view.get_attribute(TT.DOF_FRICTION_PROPERTIES)).clone()
    public_friction_before = [
        articulation.data.joint_friction_coeff.torch.clone(),
        articulation.data.joint_dynamic_friction_coeff.torch.clone(),
        articulation.data.joint_viscous_friction_coeff.torch.clone(),
    ]
    static_friction = torch.tensor([[0.9, 0.7]], device=device)
    dynamic_friction = torch.tensor([[0.4, 0.3]], device=device)
    viscous_friction = torch.tensor([[0.11, 0.22]], device=device)
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=static_friction,
        joint_dynamic_friction_coeff=dynamic_friction,
        joint_viscous_friction_coeff=viscous_friction,
        env_ids=env_ids,
        joint_ids=joint_ids,
    )
    backend_joint_ids = torch.as_tensor(articulation.joint_ordering.user_to_backend_indices)[joint_ids.cpu()]
    expected_backend_friction = backend_friction_before.clone()
    expected_backend_friction[1, backend_joint_ids, 0] = static_friction.cpu()
    expected_backend_friction[1, backend_joint_ids, 1] = dynamic_friction.cpu()
    expected_backend_friction[1, backend_joint_ids, 2] = viscous_friction.cpu()
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute(TT.DOF_FRICTION_PROPERTIES)),
        expected_backend_friction,
    )
    # The public getters expose the same coefficients.
    for getter, before, written in zip(
        (
            articulation.data.joint_friction_coeff,
            articulation.data.joint_dynamic_friction_coeff,
            articulation.data.joint_viscous_friction_coeff,
        ),
        public_friction_before,
        (static_friction, dynamic_friction, viscous_friction),
    ):
        expected = before.clone()
        expected[env_ids[:, None], joint_ids] = written
        torch.testing.assert_close(getter.torch, expected)

    body_ids = torch.tensor([articulation.num_bodies - 1, 1], dtype=torch.int32, device=device)
    backend_body_ids = torch.as_tensor(articulation.body_ordering.user_to_backend_indices)[body_ids.cpu()]
    raw_mass_before = wp.to_torch(articulation.root_view.get_attribute(TT.BODY_MASS)).clone()
    masses = torch.tensor([[2.5, 3.5]], device=device)
    articulation.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
    expected_raw_mass = raw_mass_before.clone()
    expected_raw_mass[1, backend_body_ids] = masses.cpu()
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_attribute(TT.BODY_MASS)), expected_raw_mass)

    raw_com_before = wp.to_torch(articulation.root_view.get_attribute(TT.BODY_COM_POSE)).clone()
    coms = articulation.data.body_com_pose_b.torch[env_ids][:, body_ids].clone()
    coms[0, 0, :3] = torch.tensor([0.02, -0.01, 0.03], device=device)
    coms[0, 1, :3] = torch.tensor([-0.03, 0.01, 0.02], device=device)
    articulation.set_coms_index(coms=wp.from_torch(coms, dtype=wp.transformf), env_ids=env_ids, body_ids=body_ids)
    expected_raw_com = raw_com_before.clone()
    expected_raw_com[1, backend_body_ids] = coms.cpu()
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_attribute(TT.BODY_COM_POSE)), expected_raw_com)

    raw_inertia_before = wp.to_torch(articulation.root_view.get_attribute(TT.BODY_INERTIA)).clone()
    inertias = articulation.data.body_inertia.torch[env_ids][:, body_ids].clone()
    inertias[0, 0, 0] *= 1.2
    inertias[0, 1, 4] *= 1.3
    articulation.set_inertias_index(inertias=inertias, env_ids=env_ids, body_ids=body_ids)
    expected_raw_inertia = raw_inertia_before.clone()
    expected_raw_inertia[1, backend_body_ids] = inertias.cpu()
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_attribute(TT.BODY_INERTIA)), expected_raw_inertia)
    torch.testing.assert_close(articulation.data.body_mass.torch[env_ids][:, body_ids], masses)
    torch.testing.assert_close(articulation.data.body_com_pose_b.torch[env_ids][:, body_ids], coms)
    torch.testing.assert_close(articulation.data.body_inertia.torch[env_ids][:, body_ids], inertias)


@pytest.mark.parametrize("scene", _ALL_DEVICES, indirect=True)
def test_set_material_properties(scene: _ArticulationScene) -> None:
    """Randomize per-shape articulation materials through the OVPhysX material event.

    OVPhysX exposes per-collision-shape material as the
    ``articulation_shape_friction_and_restitution`` tensor binding (shape ``[N, S, 3]`` =
    static friction, dynamic friction, restitution), addressed through the
    :class:`~isaaclab_ov.sim.views.OvPhysxView`. The binding is CPU-native, so the
    buffer lives in host memory.
    """
    articulation = scene.islands["ordered"]
    num_articulations = articulation.num_instances
    device = scene.device
    sim = scene.sim

    # The ranges exclude the asset's default material, so values inside them prove the write happened.
    static_range, dynamic_range, restitution_range = (1.5, 2.0), (1.2, 1.4), (0.6, 0.8)
    view = articulation.root_view
    materials_before = wp.to_torch(view.get_attribute(TT.SHAPE_FRICTION_AND_RESTITUTION)).clone()
    assert materials_before.shape[1] > 0
    assert (materials_before[..., 0] < static_range[0]).all(), f"default material overlaps: {materials_before}"
    params = {
        "static_friction_range": static_range,
        "dynamic_friction_range": dynamic_range,
        "restitution_range": restitution_range,
        "num_buckets": 16,
        "asset_cfg": SceneEntityCfg("robot"),
    }
    env = SimpleNamespace(scene={"robot": articulation}, sim=sim)
    cfg = EventTermCfg(func=randomize_rigid_body_material, mode="reset", params=params)
    randomize = randomize_rigid_body_material(cfg, env)

    # Randomize only the last environment; the others keep their materials.
    randomize(env, torch.tensor([num_articulations - 1], device=device), **cfg.params)
    scene.step(articulation)

    materials = wp.to_torch(view.get_attribute(TT.SHAPE_FRICTION_AND_RESTITUTION))
    torch.testing.assert_close(materials[:-1], materials_before[:-1])
    eps = 1e-5
    for component, (lo, hi) in enumerate((static_range, dynamic_range, restitution_range)):
        values = materials[-1, :, component]
        assert ((values >= lo - eps) & (values <= hi + eps)).all(), values


@pytest.mark.parametrize("scene", _ALL_DEVICES, indirect=True)
@pytest.mark.parametrize("island", ["limits_cfg", "limits_usd"])
def test_setting_velocity_and_effort_limits_write_to_solver(scene: _ArticulationScene, island: str) -> None:
    """Test that the resolved joint velocity and effort limits reach the OVPhysX solver.

    The full limit-resolution matrix (config override vs. USD default, implicit and explicit
    actuators, actuator-limit soft fallback) is covered on the Newton backend and at unit
    level. This test only verifies the OVPhysX write path: the configured limits (or the
    USD-authored defaults when unset) land in the native solver buffers and match
    ``data.joint_vel_limits`` and ``data.joint_effort_limits``.
    """
    articulation = scene.islands[island]
    device = scene.device
    actuator_cfg = _ISLANDS[island].actuators["joints"]
    joint_drive_props = _ISLANDS[island].joint_drive_props

    physx_vel_limit = _read_binding_to_torch(articulation, TT.DOF_MAX_VELOCITY, device)
    torch.testing.assert_close(articulation.data.joint_vel_limits.torch, physx_vel_limit)
    # the solver clamp comes from joint_velocity_limit when set, otherwise the USD-authored value
    if actuator_cfg.joint_velocity_limit is None:
        limit = next(p.max_joint_velocity for p in joint_drive_props if isinstance(p, PhysxJointCfg))
    else:
        limit = actuator_cfg.joint_velocity_limit
    expected_velocity_limit = torch.full_like(physx_vel_limit, limit)
    torch.testing.assert_close(physx_vel_limit, expected_velocity_limit)

    physx_effort_limit = _read_binding_to_torch(articulation, TT.DOF_MAX_FORCE, device)
    torch.testing.assert_close(articulation.data.joint_effort_limits.torch, physx_effort_limit)
    # the solver keeps the USD-authored limit unless the user overrides it explicitly
    if actuator_cfg.joint_effort_limit is None:
        limit = next(p.max_force for p in joint_drive_props if isinstance(p, sim_utils.UsdPhysicsDriveCfg))
    else:
        limit = actuator_cfg.joint_effort_limit
    expected_effort_limit = torch.full_like(physx_effort_limit, limit)
    torch.testing.assert_close(physx_effort_limit, expected_effort_limit)


@pytest.mark.parametrize("scene", _ALL_DEVICES, indirect=True)
def test_effort_binding_excludes_implicit_pd(scene: _ArticulationScene) -> None:
    """Submit implicit feedforward and explicit PD effort without submitting implicit PD telemetry.

    The CUDA scene runs the explicit group on the native Newton actuator path; the CPU scene keeps it
    on the host actuator path.
    """
    articulation = scene.islands["mixed"]
    stiffness, effort_limit = 20.0, 400.0
    joint_names = articulation.joint_names
    shoulder_ids = [joint_names.index(name) for name in ("left_shoulder", "right_shoulder")]
    elbow_ids = [joint_names.index(name) for name in ("left_elbow", "right_elbow")]
    initial_pos = articulation.data.joint_pos.torch.clone()
    position_target = initial_pos.clone()
    position_target[:, shoulder_ids] += 0.25
    position_target[:, elbow_ids] += 0.5
    feedforward = torch.zeros_like(initial_pos)
    feedforward[:, shoulder_ids] = 1.5
    feedforward[:, elbow_ids] = -0.75
    articulation.actuators.target_command.set_position_index(value=position_target)
    articulation.actuators.target_command.set_effort_index(value=feedforward)
    articulation.write_data_to_sim()

    expected_pd_effort = torch.clamp(
        stiffness * (position_target - initial_pos) + feedforward, -effort_limit, effort_limit
    )
    applied_effort = articulation.actuators.applied_effort.torch
    torch.testing.assert_close(applied_effort, expected_pd_effort)
    assert torch.all(applied_effort[:, shoulder_ids] != feedforward[:, shoulder_ids])
    expected_force = expected_pd_effort.clone()
    expected_force[:, shoulder_ids] = feedforward[:, shoulder_ids]
    backend_to_user = [joint_names.index(name) for name in articulation.root_view.dof_names]
    backend_force = _read_binding_to_torch(articulation, TT.DOF_ACTUATION_FORCE, scene.device)
    torch.testing.assert_close(backend_force, expected_force[:, backend_to_user])

    expected_stiffness = torch.zeros_like(initial_pos)
    expected_stiffness[:, shoulder_ids] = stiffness
    torch.testing.assert_close(
        _read_binding_to_torch(articulation, TT.DOF_STIFFNESS, scene.device),
        expected_stiffness[:, backend_to_user],
    )
    assert articulation._actuator_control.native_actuator_path_active == scene.native_actuators


@pytest.mark.parametrize("scene", _ALL_DEVICES, indirect=True)
def test_com_orientation_write_invalidates_static_inertia_cache_with_body_ordering(scene: _ArticulationScene) -> None:
    """Partial COM writes and COM rotations follow non-identity body order.

    COM pose is a CPU-resident OVPhysX binding, and the raw ``root_view.set_attribute`` restore
    below forbids cross-device staging, so the raw COM tensors stay on the host.
    """
    articulation = scene.islands["com_order"]
    device = scene.device
    assert articulation.body_ordering is not None

    public_body_id = 1
    backend_body_id = articulation.body_ordering.user_to_backend_indices[public_body_id]
    assert backend_body_id != public_body_id

    # A partial ordered COM write lands on the selected backend body and preserves every other one.
    backend_before = _read_binding_to_torch(articulation, TT.BODY_COM_POSE, "cpu").clone()
    assert torch.unique(backend_before[0], dim=0).shape[0] > 1
    articulation.data._body_com_pose_b.timestamp = -1.0
    backend_staging = articulation.data._body_com_pose_b_backend
    if backend_staging is not None:
        backend_staging.timestamp = -1.0
    selected_com = backend_before[0, backend_body_id].clone()
    selected_com[0] += 0.001
    articulation.set_coms_index(
        coms=wp.from_torch(selected_com.reshape(1, 1, 7).to(device), dtype=wp.transformf),
        env_ids=wp.array([0], dtype=wp.int32, device=device),
        body_ids=wp.array([public_body_id], dtype=wp.int32, device=device),
    )
    backend_after = _read_binding_to_torch(articulation, TT.BODY_COM_POSE, "cpu").clone()
    articulation.root_view.set_attribute(TT.BODY_COM_POSE, wp.from_torch(backend_before))
    noop_after = _read_binding_to_torch(articulation, TT.BODY_COM_POSE, "cpu").clone()
    unselected_body_mask = torch.ones(backend_before.shape[1], dtype=torch.bool)
    unselected_body_mask[backend_body_id] = False
    assert torch.equal(noop_after[..., :3], backend_before[..., :3])
    assert torch.equal(backend_after[0, backend_body_id, :3], selected_com[:3])
    assert torch.equal(backend_after[0, unselected_body_mask, :3], backend_before[0, unselected_body_mask, :3])
    # Bound semantic orientation equality by the native setter's float32 no-op normalization.
    native_orientation_atol = torch.max(
        torch.abs(noop_after[0, unselected_body_mask, 3:7] - backend_before[0, unselected_body_mask, 3:7])
    ).item()
    assert native_orientation_atol <= torch.finfo(backend_before.dtype).eps
    torch.testing.assert_close(
        backend_after[0, unselected_body_mask, 3:7],
        backend_before[0, unselected_body_mask, 3:7],
        rtol=0.0,
        atol=native_orientation_atol,
    )
    assert torch.equal(backend_after[..., 3:7], noop_after[..., 3:7])

    # A COM rotation refreshes the static inertia cache.
    coms = articulation.data.body_com_pose_b.torch[:, public_body_id : public_body_id + 1].clone()
    coms[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    articulation.set_coms_index(coms=wp.from_torch(coms, dtype=wp.transformf), body_ids=[public_body_id])

    principal_inertia = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]]], device=device)
    articulation.set_inertias_index(inertias=principal_inertia, body_ids=[public_body_id])
    torch.testing.assert_close(
        articulation.data.body_inertia.torch[:, public_body_id : public_body_id + 1], principal_inertia
    )

    coms[..., 3:7] = torch.tensor([0.0, 0.0, 0.70710677, 0.70710677], device=device)
    articulation.set_coms_index(coms=wp.from_torch(coms, dtype=wp.transformf), body_ids=[public_body_id])
    scene.step(articulation)
    expected_rotated_inertia = torch.tensor([[[2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0]]], device=device)
    torch.testing.assert_close(
        articulation.data.body_inertia.torch[:, public_body_id : public_body_id + 1],
        expected_rotated_inertia,
    )


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_loading_gains_from_usd(scene: _ArticulationScene) -> None:
    """Adopt the authored drive gains when the actuator config leaves stiffness and damping unset."""
    articulation = scene.islands["usd_gains"]
    actuator = articulation.actuators["joints"]
    # Angular drive gains are authored per degree and exposed per radian.
    expected_stiffness = torch.full_like(actuator.stiffness, math.degrees(_STIFFNESS))
    expected_damping = torch.full_like(actuator.damping, math.degrees(_DAMPING))
    torch.testing.assert_close(actuator.stiffness, expected_stiffness)
    torch.testing.assert_close(actuator.damping, expected_damping)
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.DOF_STIFFNESS, scene.device), expected_stiffness)
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.DOF_DAMPING, scene.device), expected_damping)


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_explicit_articulation_root_on_a_floating_base_body(scene: _ArticulationScene) -> None:
    """Initialize and simulate a floating articulation from an explicit root path to a rigid body below the asset."""
    articulation = scene.islands["non_root"]
    num_articulations = articulation.num_instances
    assert articulation.is_initialized
    assert not articulation.is_fixed_base
    assert articulation.root_view.prim_paths == [
        f"/World/non_root/Env_{index}/Robot/base" for index in range(num_articulations)
    ]
    assert articulation.data.root_pos_w.torch.shape == (num_articulations, 3)
    assert articulation.data.root_quat_w.torch.shape == (num_articulations, 4)
    assert articulation.data.joint_pos.torch.shape == (num_articulations, articulation.num_joints)
    for tt in (TT.DOF_POSITION, TT.DOF_VELOCITY, TT.DOF_STIFFNESS):
        assert articulation.root_view.binding_for(tt).shape[1] == articulation.num_joints
    for tt in (TT.BODY_MASS, TT.BODY_COM_POSE):
        assert articulation.root_view.binding_for(tt).shape[1] == articulation.num_bodies
    for actuator in articulation.actuators.values():
        assert actuator.is_implicit_model
        assert actuator.joint_indices == slice(None)
    scene.step(articulation, count=10)
    assert torch.isfinite(articulation.data.root_link_pose_w.torch).all()


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_implicit_drive_targets_submit_feedforward_effort_and_track(scene: _ArticulationScene) -> None:
    """Write nonidentity-ordered implicit targets to their backend columns and track them in the solver."""
    articulation = scene.islands["drive"]
    device = scene.device
    ordering = articulation.joint_ordering
    assert ordering is not None
    user_to_backend = torch.as_tensor(ordering.user_to_backend_indices, dtype=torch.long, device=device)
    backend_to_user = torch.as_tensor(ordering.backend_to_user_indices, dtype=torch.long, device=device)
    assert not torch.equal(user_to_backend, backend_to_user)

    joint_index = torch.arange(articulation.num_joints, dtype=torch.float32, device=device).unsqueeze(0)
    position_target = (-0.25 + 0.031 * joint_index).repeat(articulation.num_instances, 1)
    velocity_target = (0.07 + 0.017 * joint_index).repeat(articulation.num_instances, 1)
    effort_target = torch.where(joint_index.remainder(2) == 0, 0.0, 0.13 * joint_index).repeat(
        articulation.num_instances, 1
    )
    joint_pos = articulation.data.joint_pos.torch.clone()
    joint_vel = articulation.data.joint_vel.torch.clone()
    articulation.set_joint_position_target_index(target=position_target)
    articulation.set_joint_velocity_target_index(target=velocity_target)
    articulation.actuators.target_command.set_effort_index(value=effort_target)
    articulation.write_data_to_sim()

    backend_position_target = _read_binding_to_torch(articulation, TT.DOF_POSITION_TARGET, device)
    backend_velocity_target = _read_binding_to_torch(articulation, TT.DOF_VELOCITY_TARGET, device)
    torch.testing.assert_close(backend_position_target, position_target[:, backend_to_user])
    torch.testing.assert_close(backend_velocity_target, velocity_target[:, backend_to_user])
    torch.testing.assert_close(
        _read_binding_to_torch(articulation, TT.DOF_ACTUATION_FORCE, device), effort_target[:, backend_to_user]
    )
    computed_effort = (
        _STIFFNESS * (position_target - joint_pos) + _DAMPING * (velocity_target - joint_vel) + effort_target
    )
    limits = articulation.data.joint_effort_limits.torch
    expected_telemetry = torch.clamp(computed_effort, min=-limits, max=limits)
    torch.testing.assert_close(articulation.actuators.applied_effort.torch, expected_telemetry)
    assert not torch.allclose(expected_telemetry, effort_target)

    # A partial position target lands on the selected backend cell; the other environment keeps its targets.
    articulation.set_joint_velocity_target_index(target=torch.zeros_like(velocity_target))
    articulation.actuators.target_command.set_effort_index(value=torch.zeros_like(effort_target))
    raw_target_before = _read_binding_to_torch(articulation, TT.DOF_POSITION_TARGET, device).clone()
    drive_target = position_target[1:, :1] + 0.4
    articulation.actuators.target_command.set_position_index(
        value=drive_target,
        env_ids=torch.tensor([1], dtype=torch.int32, device=device),
        joint_ids=torch.tensor([0], dtype=torch.int32, device=device),
    )
    articulation.write_data_to_sim()
    backend_target = _read_binding_to_torch(articulation, TT.DOF_POSITION_TARGET, device)
    torch.testing.assert_close(backend_target[1, user_to_backend[0]], drive_target[0, 0])
    torch.testing.assert_close(backend_target[0], raw_target_before[0])

    # The drives track the commanded targets. Without the target write, they would hold the old ones.
    expected_position = position_target.clone()
    expected_position[1, 0] = drive_target[0, 0]
    scene.step(articulation, count=100)
    torch.testing.assert_close(articulation.data.joint_pos.torch, expected_position, atol=0.1, rtol=0.0)


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_computed_dynamics_follow_public_order_and_model_writes(scene: _ArticulationScene) -> None:
    """Resolve depth-first MJWarp order for ``"mjwarp"`` on OVPhysX and gather computed dynamics into it.

    OVPhysX is a PhysX-family backend (native breadth-first order), so requesting ``mjwarp`` triggers a
    temporary Newton USD discovery of the depth-first order and reorders the public joint/body axes to it.
    Joint-state and model-property writes refresh the gathered dynamics without advancing simulation time.
    """
    articulation = scene.islands["dynamics"]
    device = scene.device
    num_envs = articulation.num_instances

    # OVPhysX exposes the native breadth-first PhysX order on the backend axis.
    assert tuple(articulation.backend_joint_names) == BRANCHING_PHYSX_JOINT_NAMES
    assert tuple(articulation.backend_body_names) == BRANCHING_PHYSX_BODY_NAMES

    # Cross-backend Newton discovery resolves the depth-first MJWarp order and reorders the public axis.
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="joint") == BRANCHING_MJWARP_JOINT_NAMES
    assert get_articulation_name_ordering(articulation, "mjwarp", kind="body") == BRANCHING_MJWARP_BODY_NAMES
    assert tuple(articulation.joint_names) == BRANCHING_MJWARP_JOINT_NAMES
    assert tuple(articulation.body_names) == BRANCHING_MJWARP_BODY_NAMES

    joint_ordering = articulation.joint_ordering
    body_ordering = articulation.body_ordering
    assert joint_ordering is not None
    assert body_ordering is not None
    joint_user_to_backend = torch.as_tensor(joint_ordering.user_to_backend_indices, dtype=torch.long, device=device)
    body_offset = 1 if articulation.is_fixed_base else 0
    body_user_to_backend = torch.as_tensor(
        [
            backend_body_id - body_offset
            for backend_body_id in body_ordering.user_to_backend_indices
            if not body_offset or backend_body_id != 0
        ],
        dtype=torch.long,
        device=device,
    )
    generalized_user_to_backend = torch.cat(
        (
            torch.arange(articulation.num_base_dofs, device=device),
            articulation.num_base_dofs + joint_user_to_backend,
        )
    )

    def expected_dynamics() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_jacobian = _read_binding_to_torch(articulation, TT.JACOBIAN, device).reshape(
            num_envs,
            articulation.num_bodies - body_offset,
            6,
            articulation.num_joints + articulation.num_base_dofs,
        )
        raw_mass_matrix = _read_binding_to_torch(articulation, TT.MASS_MATRIX, device)
        raw_gravity = _read_binding_to_torch(articulation, TT.GRAVITY_FORCE, device)
        return (
            raw_jacobian[:, body_user_to_backend, :, :][:, :, :, generalized_user_to_backend],
            raw_mass_matrix[:, generalized_user_to_backend, :][:, :, generalized_user_to_backend],
            raw_gravity[:, generalized_user_to_backend],
        )

    def assert_dynamics_match(jacobian: bool = True, gravity: bool = True) -> None:
        expected_jacobian, expected_mass_matrix, expected_gravity = expected_dynamics()
        if jacobian:
            torch.testing.assert_close(articulation.data.body_com_jacobian_w.torch, expected_jacobian)
        torch.testing.assert_close(articulation.data.mass_matrix.torch, expected_mass_matrix)
        if gravity:
            torch.testing.assert_close(articulation.data.gravity_compensation_forces.torch, expected_gravity)

    data = articulation.data
    body_com_jacobian = data.body_com_jacobian_w
    mass_matrix = data.mass_matrix
    gravity = data.gravity_compensation_forces
    assert body_com_jacobian is data.body_com_jacobian_w
    assert mass_matrix is data.mass_matrix
    assert gravity is data.gravity_compensation_forces
    assert_dynamics_match()
    assert data.body_link_jacobian_w.torch.shape == body_com_jacobian.torch.shape
    assert torch.isfinite(data.body_link_jacobian_w.torch).all()
    torch.testing.assert_close(mass_matrix.torch, mass_matrix.torch.transpose(-1, -2), rtol=1e-5, atol=1e-5)

    raw_jacobian_before = expected_dynamics()[0]
    joint_position = data.joint_pos.torch.clone()
    joint_position[:, 1] += 0.2
    articulation.write_joint_position_to_sim_index(position=joint_position)
    assert not torch.allclose(expected_dynamics()[0], raw_jacobian_before)
    assert_dynamics_match()

    joint_velocity = torch.linspace(-0.2, 0.2, articulation.num_joints, dtype=torch.float32, device=device)
    joint_velocity = joint_velocity.repeat(num_envs, 1)
    articulation.write_joint_velocity_to_sim_index(velocity=joint_velocity)
    expected_com_velocity = torch.einsum("nbij,nj->nbi", data.body_com_jacobian_w.torch, joint_velocity)
    expected_link_velocity = torch.einsum("nbij,nj->nbi", data.body_link_jacobian_w.torch, joint_velocity)
    torch.testing.assert_close(expected_com_velocity, data.body_com_vel_w.torch[:, 1:], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(expected_link_velocity, data.body_link_vel_w.torch[:, 1:], atol=1e-5, rtol=1e-4)

    # Model-property writes refresh the computed dynamics without advancing simulation time.
    # COM offsets affect COM Jacobians, mass matrices, and gravity forces.
    data.body_com_jacobian_w
    data.mass_matrix
    data.gravity_compensation_forces
    coms = data.body_com_pose_b.torch.clone()
    coms[:, -1, 0] += 0.01
    articulation.set_coms_index(coms=wp.from_torch(coms, dtype=wp.transformf))
    assert data._body_com_jacobian_w.timestamp < data._sim_timestamp
    assert data._mass_matrix.timestamp < data._sim_timestamp
    assert data._gravity_compensation_forces.timestamp < data._sim_timestamp
    assert_dynamics_match()

    # Mass affects the mass matrix and gravity forces, but not the kinematic Jacobian.
    masses = data.body_mass.torch.clone()
    masses[:, -1] *= 1.1
    articulation.set_masses_index(masses=masses)
    assert data._mass_matrix.timestamp < data._sim_timestamp
    assert data._gravity_compensation_forces.timestamp < data._sim_timestamp
    assert_dynamics_match(jacobian=False)

    # Inertia and armature each affect only the generalized mass matrix.
    inertias = data.body_inertia.torch.clone()
    inertias[:, -1, [0, 4, 8]] *= 1.1
    articulation.set_inertias_index(inertias=inertias)
    assert data._mass_matrix.timestamp < data._sim_timestamp
    assert_dynamics_match(jacobian=False, gravity=False)

    armature = data.joint_armature.torch.clone()
    armature[:, -1] += 0.01
    articulation.write_joint_armature_to_sim_index(armature=armature)
    assert data._mass_matrix.timestamp < data._sim_timestamp
    assert_dynamics_match(jacobian=False, gravity=False)


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
@pytest.mark.parametrize("island", ["reversed", "reversed_ordered"])
def test_reversed_joint_dynamics_use_public_joint_basis(scene: _ArticulationScene, island: str) -> None:
    """Check velocity, kinetic energy and gravity in the public joint basis."""
    articulation = scene.islands[island]
    device = scene.device
    assert (articulation.joint_ordering is not None) == (island == "reversed_ordered")

    articulation.write_joint_position_to_sim_index(position=torch.zeros_like(articulation.data.joint_pos.torch))
    velocity = torch.zeros((1, articulation.num_joints), device=device)
    velocity[:, articulation.find_joints("left_shoulder")[0][0]] = 0.4
    velocity[:, articulation.find_joints("left_elbow")[0][0]] = 0.7
    articulation.write_joint_velocity_to_sim_index(velocity=velocity)

    joint_velocity = articulation.data.joint_vel.torch
    predicted_velocity = torch.einsum("nbij,nj->nbi", articulation.data.body_com_jacobian_w.torch, joint_velocity)
    torch.testing.assert_close(predicted_velocity, articulation.data.body_com_vel_w.torch[:, 1:], atol=1e-5, rtol=1e-5)

    generalized_energy = 0.5 * torch.einsum(
        "ni,nij,nj->n", joint_velocity, articulation.data.mass_matrix.torch, joint_velocity
    )
    body_velocity = articulation.data.body_com_vel_w.torch
    body_inertia = articulation.data.body_inertia.torch.reshape(1, articulation.num_bodies, 3, 3)
    body_energy = 0.5 * (
        (articulation.data.body_mass.torch.unsqueeze(-1) * body_velocity[..., :3].square()).sum((-1, -2))
        + torch.einsum("nbi,nbij,nbj->n", body_velocity[..., 3:], body_inertia, body_velocity[..., 3:])
    )
    torch.testing.assert_close(generalized_energy, body_energy, atol=1e-5, rtol=1e-5)

    gravity = torch.tensor(scene.sim.cfg.gravity, device=device)
    body_weight = articulation.data.body_mass.torch[:, 1:, None] * gravity
    expected_compensation = -torch.einsum(
        "nbij,nbi->nj", articulation.data.body_com_jacobian_w.torch[:, :, :3], body_weight
    )
    assert torch.all(expected_compensation.abs() > 0.1)
    torch.testing.assert_close(
        articulation.data.gravity_compensation_forces.torch, expected_compensation, atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_floating_articulation_root_state_dynamics_wrench_and_reset(scene: _ArticulationScene) -> None:
    """Prove floating-root state, generalized dynamics, reset, and a real OVPhysX external-wrench response."""
    articulation = scene.islands["floating"]
    device = scene.device
    num_articulations = articulation.num_instances
    num_bodies = articulation.num_bodies
    num_joints = articulation.num_joints
    assert articulation.is_initialized
    assert not articulation.is_fixed_base
    assert articulation.data.root_link_pose_w.torch.shape == (num_articulations, 7)
    assert articulation.data.root_com_pose_w.torch.shape == (num_articulations, 7)
    assert articulation.data.body_link_pose_w.torch.shape == (num_articulations, num_bodies, 7)
    assert articulation.data.body_com_pose_w.torch.shape == (num_articulations, num_bodies, 7)

    # Floating-base columns stay leading while the actuated-joint axes are gathered.
    user_to_backend = torch.as_tensor(articulation.joint_ordering.user_to_backend_indices, device=device)
    generalized_user_to_backend = torch.cat((torch.arange(6, device=device), 6 + user_to_backend))
    raw_jacobian = _read_binding_to_torch(articulation, TT.JACOBIAN, device).reshape(
        num_articulations, num_bodies, 6, num_joints + 6
    )
    raw_mass_matrix = _read_binding_to_torch(articulation, TT.MASS_MATRIX, device)
    raw_gravity = _read_binding_to_torch(articulation, TT.GRAVITY_FORCE, device)
    torch.testing.assert_close(
        articulation.data.body_com_jacobian_w.torch,
        raw_jacobian[:, :, :, generalized_user_to_backend],
    )
    torch.testing.assert_close(
        articulation.data.mass_matrix.torch,
        raw_mass_matrix[:, generalized_user_to_backend, :][:, :, generalized_user_to_backend],
    )
    torch.testing.assert_close(
        articulation.data.gravity_compensation_forces.torch,
        raw_gravity[:, generalized_user_to_backend],
    )

    # A partial reset clears the wrenches of the selected environment only; a full reset clears them all.
    composers = (articulation.instantaneous_wrench_composer, articulation.permanent_wrench_composer)
    ones = torch.ones((num_articulations, num_bodies, 3), device=device)
    articulation.permanent_wrench_composer.set_forces_and_torques_index(forces=ones, torques=ones)
    articulation.instantaneous_wrench_composer.add_forces_and_torques_index(forces=ones, torques=ones)
    articulation.reset(env_ids=torch.tensor([0], device=device))
    for composer in composers:
        assert composer.active
        assert torch.count_nonzero(composer.composed_force.torch) == num_bodies * 3
        assert torch.count_nonzero(composer.composed_torque.torch) == num_bodies * 3
    articulation.reset()
    for composer in composers:
        assert not composer.active
        assert torch.count_nonzero(composer.composed_force.torch) == 0
        assert torch.count_nonzero(composer.composed_torque.torch) == 0

    # A written root frame moves the other frame through the root-body COM offset.
    com = _read_binding_to_torch(articulation, TT.BODY_COM_POSE, device)
    com[:, 0, :3] = torch.tensor([1.0, 0.0, 0.0], device=device)
    articulation.set_coms_index(coms=wp.from_torch(com, dtype=wp.transformf))
    com_pos_b, com_quat_b = com[:, 0, :3], com[:, 0, 3:7]
    link_pos_in_com, link_quat_in_com = math_utils.subtract_frame_transforms(com_pos_b, com_quat_b)
    rand_state = torch.zeros(num_articulations, 13, device=device)
    rand_state[..., :3] = articulation.data.root_link_pos_w.torch + torch.tensor([0.1, 0.2, 0.3], device=device)
    rand_state[..., 3:7] = torch.nn.functional.normalize(torch.tensor([0.1, 0.2, 0.3, 0.9], device=device), dim=-1)
    for env_ids in (None, torch.arange(num_articulations, dtype=torch.int32, device=device)):
        articulation.write_root_com_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_ids)
        articulation.write_root_com_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_ids)
        torch.testing.assert_close(rand_state[..., :7], articulation.data.root_com_pose_w.torch)
        torch.testing.assert_close(rand_state[..., 7:], articulation.data.root_com_vel_w.torch)
        # The link frame is the written COM frame composed with the inverse COM offset.
        expected_link_pos, expected_link_quat = math_utils.combine_frame_transforms(
            rand_state[..., :3], rand_state[..., 3:7], link_pos_in_com, link_quat_in_com
        )
        torch.testing.assert_close(articulation.data.root_link_pos_w.torch, expected_link_pos)
        torch.testing.assert_close(articulation.data.root_link_quat_w.torch, expected_link_quat)

        articulation.write_root_link_pose_to_sim_index(root_pose=rand_state[..., :7], env_ids=env_ids)
        articulation.write_root_link_velocity_to_sim_index(root_velocity=rand_state[..., 7:], env_ids=env_ids)
        torch.testing.assert_close(rand_state[..., :7], articulation.data.root_link_pose_w.torch)
        torch.testing.assert_close(rand_state[..., 7:], articulation.data.root_link_vel_w.torch)
        # The COM frame is the written link frame composed with the COM offset.
        expected_com_pos, expected_com_quat = math_utils.combine_frame_transforms(
            rand_state[..., :3], rand_state[..., 3:7], com_pos_b, com_quat_b
        )
        torch.testing.assert_close(articulation.data.root_com_pos_w.torch, expected_com_pos)
        torch.testing.assert_close(articulation.data.root_com_quat_w.torch, expected_com_quat)

    # A partial root write keeps the other environment, and a wrench on it accelerates only that one.
    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    initial_pose = articulation.data.root_link_pose_w.torch.clone()
    target_pose = initial_pose[env_ids].clone()
    target_pose[:, :3] += torch.tensor([0.2, -0.1, 0.3], device=device)
    articulation.write_root_link_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
    torch.testing.assert_close(articulation.data.root_link_pose_w.torch[env_ids], target_pose)
    torch.testing.assert_close(articulation.data.root_link_pose_w.torch[:1], initial_pose[:1])
    articulation.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(rand_state[..., 7:]))
    articulation.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor([[[8.0, 0.0, 0.0]]], device=device),
        torques=torch.zeros((1, 1, 3), device=device),
        env_ids=env_ids,
        body_ids=torch.tensor([0], dtype=torch.int32, device=device),
    )
    scene.step(articulation)
    # Gravity acts along z, so only the pushed environment gains velocity along x.
    assert articulation.data.root_com_lin_vel_w.torch[1, 0] > 1e-3
    torch.testing.assert_close(
        articulation.data.root_com_lin_vel_w.torch[0, :2], torch.zeros(2, device=device), atol=1e-6, rtol=0
    )
    articulation.reset()


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_tendon_properties_and_position_targets(scene: _ArticulationScene) -> None:
    """Prove locally authored spatial and fixed tendons are discovered and writable.

    A fixed-tendon target lands in the simulation as ``rest_length - target``. The index form commands
    every tendon of environment 0; the mask form commands tendon 0 of environment 1. Every other cell must
    keep its initial offset.
    """
    articulation = scene.islands["tendon"]
    device = scene.device
    num_articulations = articulation.num_instances
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.num_spatial_tendons == 1
    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    initial_stiffness = articulation.data.spatial_tendon_stiffness.torch.clone()
    stiffness = torch.tensor([[12.0]], device=device)
    damping = torch.tensor([[1.5]], device=device)
    limit_stiffness = torch.tensor([[3.0]], device=device)
    offset = torch.tensor([[0.1]], device=device)
    articulation.set_spatial_tendon_stiffness_index(stiffness=stiffness, env_ids=env_ids)
    articulation.set_spatial_tendon_damping_index(damping=damping, env_ids=env_ids)
    articulation.set_spatial_tendon_limit_stiffness_index(limit_stiffness=limit_stiffness, env_ids=env_ids)
    articulation.set_spatial_tendon_offset_index(offset=offset, env_ids=env_ids)
    torch.testing.assert_close(articulation.data.spatial_tendon_stiffness.torch[env_ids], stiffness)
    torch.testing.assert_close(articulation.data.spatial_tendon_stiffness.torch[:1], initial_stiffness[:1])
    torch.testing.assert_close(articulation.data.spatial_tendon_damping.torch[env_ids], damping)
    torch.testing.assert_close(articulation.data.spatial_tendon_limit_stiffness.torch[env_ids], limit_stiffness)
    torch.testing.assert_close(articulation.data.spatial_tendon_offset.torch[env_ids], offset)

    num_tendons = articulation.num_fixed_tendons
    assert num_tendons == 2
    rest_length = articulation.data.fixed_tendon_rest_length.torch.clone()
    initial_offset = articulation.data.fixed_tendon_offset.torch.clone()

    index_target = torch.full((1, num_tendons), 0.3, dtype=torch.float32, device=device)
    articulation.set_fixed_tendon_position_target_index(target=index_target, env_ids=[0])
    # Distinct per-cell values: a uniform target cannot catch the mask form reading the wrong
    # cell, because every wrong read returns the same number.
    mask_target = (
        0.7
        + 0.1 * torch.arange(num_articulations, dtype=torch.float32, device=device).unsqueeze(1)
        + 0.01 * torch.arange(num_tendons, dtype=torch.float32, device=device).unsqueeze(0)
    )
    env_mask = wp.array([False, True], dtype=wp.bool, device=device)
    tendon_mask = wp.from_torch(torch.arange(num_tendons, device=device) == 0)
    articulation.set_fixed_tendon_position_target_mask(
        target=mask_target, fixed_tendon_mask=tendon_mask, env_mask=env_mask
    )

    scene.step(articulation)

    expected = initial_offset.clone()
    expected[0] = rest_length[0] - 0.3
    expected[1, 0] = rest_length[1, 0] - mask_target[1, 0]
    torch.testing.assert_close(articulation.data.fixed_tendon_offset.torch, expected)


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_native_actuator_submits_real_effort(scene: _ArticulationScene) -> None:
    """Run a Newton-native explicit actuator through the current OVPhysX state and effort binding."""
    articulation = scene.islands["native"]
    device = scene.device
    assert articulation._actuator_control.native_actuator_path_active
    assert articulation.newton_actuator_adapter is not None
    backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
    user_to_backend = list(articulation.joint_ordering.user_to_backend_indices)

    initial_pos = articulation.data.joint_pos.torch.clone()
    target = initial_pos + 0.2
    articulation.actuators.target_command.set_position_index(value=target)
    articulation.write_data_to_sim()

    assert torch.any(articulation.actuators.computed_effort.torch != 0.0)
    assert torch.any(articulation.actuators.applied_effort.torch != 0.0)
    raw_effort = wp.to_torch(articulation._physx_actuator_wrapper.joint_f_2d).clone()
    assert torch.any(raw_effort != 0.0)
    backend_effort = _read_binding_to_torch(articulation, TT.DOF_ACTUATION_FORCE, device)
    torch.testing.assert_close(backend_effort, raw_effort[:, backend_to_user])
    torch.testing.assert_close(backend_effort, articulation.actuators.applied_effort.torch[:, backend_to_user])

    scene.step(articulation, count=8)
    # Use raw OV bindings so the observation cannot refresh the public state shadow.
    current_pos = _read_binding_to_torch(articulation, TT.DOF_POSITION, device)[:, user_to_backend]
    current_vel = _read_binding_to_torch(articulation, TT.DOF_VELOCITY, device)[:, user_to_backend]
    assert not torch.allclose(current_pos, initial_pos)

    articulation.write_data_to_sim()
    expected_effort = torch.clamp(_STIFFNESS * (target - current_pos) - _DAMPING * current_vel, -_MAX_FORCE, _MAX_FORCE)
    torch.testing.assert_close(articulation.actuators.applied_effort.torch, expected_effort)
    torch.testing.assert_close(
        _read_binding_to_torch(articulation, TT.DOF_ACTUATION_FORCE, device),
        wp.to_torch(articulation._physx_actuator_wrapper.joint_f_2d)[:, backend_to_user],
    )


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_native_actuator_reset_and_gain_event_are_environment_selective(scene: _ArticulationScene) -> None:
    """Reset and randomize only the selected OVPhysX native-controller environment."""
    # Newton is only needed by the native actuator path.
    from isaaclab.actuators.newton import read_group_parameter  # noqa: PLC0415

    class Env:
        def __init__(self, asset):
            self.scene = self
            self.num_envs = asset.num_instances
            self.device = asset.device
            self._asset = asset

        def __getitem__(self, name):
            assert name == "robot"
            return self._asset

    articulation = scene.islands["delayed"]
    device = scene.device
    num_joints = articulation.num_joints
    scene.step(articulation, count=3)

    adapter = articulation.newton_actuator_adapter
    stateful_pairs = [
        state
        for actuator, state in zip(adapter.actuators, adapter._states_a)
        if state is not None and getattr(state, "delay_state", None) is not None
    ]
    assert len(stateful_pairs) == 1
    articulation.reset(env_ids=torch.tensor([0], device=device, dtype=torch.long))
    # The delay state is flattened environment-major over the group's joints.
    assert stateful_pairs[0].delay_state.num_pushes.numpy().tolist() == [0] * num_joints + [1] * num_joints

    env = Env(articulation)
    asset_cfg = SceneEntityCfg("robot")
    event_params = {
        "asset_cfg": asset_cfg,
        "stiffness_distribution_params": (101.0, 101.0),
        "damping_distribution_params": (3.0, 3.0),
        "operation": "abs",
        "distribution": "uniform",
    }
    event = randomize_actuator_gains(EventTermCfg(func=randomize_actuator_gains, params=event_params), env)
    event(env, env_ids=torch.tensor([0], device=device), **event_params)

    stiffness = read_group_parameter(articulation.actuators, "joint", "controller", "kp")
    damping = read_group_parameter(articulation.actuators, "joint", "controller", "kd")
    torch.testing.assert_close(stiffness, torch.tensor([[101.0], [20.0]], device=device).repeat(1, num_joints))
    torch.testing.assert_close(damping, torch.tensor([[3.0], [1.0]], device=device).repeat(1, num_joints))


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_root_link_vel_w_refreshes_fk_before_body_com_vel_w_read(scene: _ArticulationScene, monkeypatch) -> None:
    """Reading ``root_link_vel_w`` must run FK before ``body_com_vel_w`` sees a "fresh" buffer.

    Native tensor reads may refresh internally, so also verify that asset and rendering reads
    share one explicit FK update after each write, regardless of which reader comes first.
    """
    articulation = scene.islands["ordered"]
    articulation.update(scene.sim.cfg.dt)

    # Prime the derived buffers before the write so their TimestampedBuffers are populated; otherwise
    # the reads below would trivially be "first reads" regardless of the cache-invalidation bug.
    articulation.data.root_link_vel_w
    articulation.data.body_com_vel_w

    joint_vel = torch.full((2, articulation.num_joints), 3.0, device=scene.device)
    for render_first in (False, True):
        articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)
        articulation.write_joint_position_to_sim_index(position=articulation.data.joint_pos.torch.clone())
        with monkeypatch.context() as patch:
            physx = Mock(wraps=OvPhysxManager.backend.physx)
            patch.setattr(OvPhysxManager.backend, "physx", physx)
            if render_first:
                OvPhysxManager.pre_render()
            articulation.data.root_link_vel_w
            body_com_vel_w = articulation.data.body_com_vel_w.torch
            OvPhysxManager.pre_render()
            physx.update_articulations_kinematic.assert_called_once()
            assert torch.linalg.norm(body_com_vel_w[:, 1, :]) > 1e-3


@wp.kernel
def _occupy_stream_kernel(iterations: int, out: wp.array(dtype=wp.float32)):
    total = float(0.0)
    for i in range(iterations):
        total += wp.sin(float(i))
    out[0] = total


@pytest.mark.parametrize("scene", _CUDA_SCENES, indirect=True)
def test_cpu_only_property_writes_wait_for_pinned_host_staging(scene: _ArticulationScene) -> None:
    """Land CPU-only property writes while the device stream is busy.

    OVPhysX keeps joint and body properties on the host even for a GPU simulation, so the writers
    stage the device-resident data, environment indices, and masks through pinned host buffers
    before calling the CPU setter. Warp issues that device-to-host copy asynchronously on the
    device stream, so a long kernel queued ahead of the copy must not let the setter consume the
    previous contents of the pinned buffers.
    """
    articulation = scene.islands["staging"]
    device = scene.device
    num_envs, num_joints = articulation.num_instances, articulation.num_joints

    stiffness_before = _read_binding_to_torch(articulation, TT.DOF_STIFFNESS, device)
    damping_before = _read_binding_to_torch(articulation, TT.DOF_DAMPING, device)
    masses_before = _read_binding_to_torch(articulation, TT.BODY_MASS, device)
    joint_values = torch.arange(1, num_envs * num_joints + 1, device=device, dtype=torch.float32)
    joint_values = joint_values.reshape(num_envs, num_joints)
    env_ids = torch.tensor([1, 3], device=device)
    env_mask = wp.array([False, True, False, True], dtype=wp.bool, device=device)
    scratch = wp.zeros(1, dtype=wp.float32, device=device)

    def occupy_device_stream(iterations: int = 500_000):
        wp.launch(_occupy_stream_kernel, dim=1, inputs=[iterations], outputs=[scratch], device=device)

    # Warm up every kernel involved so that module compilation cannot drain the stream
    # between the writes below and the kernel that keeps it busy.
    occupy_device_stream(iterations=1)
    articulation.write_joint_stiffness_to_sim_index(stiffness=stiffness_before[env_ids], env_ids=env_ids)
    articulation.write_joint_damping_to_sim_mask(damping=damping_before, env_mask=env_mask)
    articulation.set_masses_mask(masses=masses_before, env_mask=env_mask)
    wp.synchronize_device(device)

    occupy_device_stream()
    articulation.write_joint_stiffness_to_sim_index(stiffness=joint_values[env_ids], env_ids=env_ids)
    occupy_device_stream()
    articulation.write_joint_damping_to_sim_mask(damping=joint_values, env_mask=env_mask)
    occupy_device_stream()
    articulation.set_masses_mask(masses=masses_before * 2.0, env_mask=env_mask)
    wp.synchronize_device(device)

    expected_stiffness = stiffness_before.clone()
    expected_stiffness[env_ids] = joint_values[env_ids]
    expected_damping = damping_before.clone()
    expected_damping[env_ids] = joint_values[env_ids]
    expected_masses = masses_before.clone()
    expected_masses[env_ids] = masses_before[env_ids] * 2.0
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.DOF_STIFFNESS, device), expected_stiffness)
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.DOF_DAMPING, device), expected_damping)
    torch.testing.assert_close(_read_binding_to_torch(articulation, TT.BODY_MASS, device), expected_masses)


@pytest.mark.parametrize("scene", _ALL_DEVICES, indirect=True)
def test_joint_position_limit_writes_clamp_default_joint_positions(scene: _ArticulationScene, caplog) -> None:
    """Clamp only the selected default joint positions into written limits; logging never changes clamping.

    The logging level controls reporting without reusing a stale violation count.
    """
    articulation = scene.islands["ordered"]
    device = scene.device
    logger = type(articulation).__module__
    original_limits = articulation.data.joint_pos_limits.torch.clone()
    original_defaults = articulation.data.default_joint_pos.torch.clone()
    try:
        # The default positions are zero, so both new ranges exclude them.
        env_ids = torch.tensor([1], dtype=torch.int32, device=device)
        joint_ids = torch.tensor([0, 2], dtype=torch.int32, device=device)
        partial_limits = torch.tensor([[[0.2, 0.3], [0.25, 0.35]]], device=device)
        articulation.write_joint_position_limit_to_sim_index(
            limits=partial_limits, env_ids=env_ids, joint_ids=joint_ids
        )
        torch.testing.assert_close(articulation.data.joint_pos_limits.torch[env_ids][:, joint_ids], partial_limits)
        default_joint_pos = articulation.data.default_joint_pos.torch
        selected = default_joint_pos[env_ids][:, joint_ids]
        assert torch.all((selected >= partial_limits[..., 0]) & (selected <= partial_limits[..., 1]))
        unselected = torch.ones_like(default_joint_pos, dtype=torch.bool)
        unselected[env_ids[:, None].long(), joint_ids.long()] = False
        torch.testing.assert_close(default_joint_pos[unselected], original_defaults[unselected])

        limits = torch.zeros_like(original_limits)
        limits[..., 1] = 0.5
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
