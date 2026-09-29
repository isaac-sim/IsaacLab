# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locally authored scenes shared by the kitless Newton asset and controller tests.

Test modules below ``source/isaaclab_newton/test`` import this module after putting this directory on
``sys.path``. The fixtures are small USD documents authored here and written to a per-process temporary
directory, so the scenes need neither Kit nor Nucleus and the repository carries no generated USD files.
"""

from __future__ import annotations

import atexit
import functools
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import warp as wp
from isaaclab_newton.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, SimulationContext

NUM_ENVS = 2
"""Environments per scene: enough to prove a partial write leaves the other environment untouched."""

WRIST_USD_STIFFNESS = (0.5, 0.4, 0.3)
"""Angular drive stiffness authored on the fixed chain's wrist joints [N·m/deg]; the damping is a tenth of it."""

Vec3 = tuple[float, float, float]

_DATA_DIR = Path(__file__).resolve().parent / "assets" / "data"
"""Directory holding the checked-in USD fixtures."""

_ENV_SPACING = 40.0
"""Distance between environment origins along x [m]."""


def local_usd(name: str, **kwargs: Any) -> sim_utils.UsdFileCfg:
    """Return a spawner for one of the local USD fixtures.

    Args:
        name: Authored fixture name, or the name of a checked-in file in ``test/assets/data``.
        **kwargs: Additional :class:`~isaaclab.sim.UsdFileCfg` fields.

    Returns:
        The spawner configuration.
    """
    if name in _FIXTURES:
        path = _fixture_dir() / name
        if not path.exists():
            path.write_text(_FIXTURES[name](), encoding="utf-8")
    else:
        path = _DATA_DIR / name
    return sim_utils.UsdFileCfg(usd_path=str(path), **kwargs)


def newton_sim_cfg(
    device: str = "cpu",
    *,
    dt: float = 1.0 / 120.0,
    gravity: Vec3 = (0.0, 0.0, 0.0),
    use_newton_actuators: bool = True,
    newton_contacts: bool = False,
    solver_cfg: MJWarpSolverCfg | None = None,
    **newton_kwargs: Any,
) -> SimulationCfg:
    """Return an MJWarp simulation configuration, without CUDA-graph capture unless requested.

    Args:
        device: Simulation device.
        dt: Physics time step [s].
        gravity: Scene gravity [m/s^2]. Defaults to none, so free bodies only move when a test drives them.
        use_newton_actuators: Whether explicit actuator groups execute as Newton-native actuators.
        newton_contacts: Whether to use Newton's collision pipeline instead of MuJoCo contacts.
        solver_cfg: Solver configuration. Defaults to :class:`MJWarpSolverCfg` with the selected contacts.
        **newton_kwargs: Additional :class:`NewtonCfg` fields.

    Returns:
        The simulation configuration.
    """
    if solver_cfg is None:
        solver_cfg = MJWarpSolverCfg(use_mujoco_contacts=not newton_contacts)
    newton_kwargs.setdefault("use_cuda_graph", False)
    return SimulationCfg(
        device=device,
        dt=dt,
        gravity=gravity,
        use_newton_actuators=use_newton_actuators,
        physics=NewtonCfg(solver_cfg=solver_cfg, **newton_kwargs),
    )


def env_origins(device: str, num_envs: int = NUM_ENVS) -> torch.Tensor:
    """Return the world origins :func:`spawn_assets` places the environments at [m]."""
    origins = torch.zeros((num_envs, 3), device=device)
    origins[:, 0] = _ENV_SPACING * torch.arange(num_envs, device=device)
    return origins


def spawn_assets(
    asset_cfgs: Mapping[str, ArticulationCfg | RigidObjectCfg | RigidObjectCollectionCfg], num_envs: int = NUM_ENVS
) -> dict[str, Any]:
    """Plan, construct, and replicate assets into ``/World/Env_*`` of the active simulation.

    Call this inside a simulation context and before :meth:`~isaaclab.sim.SimulationContext.reset`.
    Each configuration's ``prim_path`` must address one asset per environment.

    Args:
        asset_cfgs: Asset configurations keyed by a name the caller uses to look the asset up.
        num_envs: Number of environments.

    Returns:
        The Newton assets keyed like ``asset_cfgs``.
    """
    positions = env_origins("cpu", num_envs).numpy()
    # a collection is planned as its member rigid objects
    planned_cfgs = []
    for cfg in asset_cfgs.values():
        planned_cfgs.extend(cfg.rigid_objects.values() if isinstance(cfg, RigidObjectCollectionCfg) else (cfg,))
    sim_utils.create_prim("/World/Env_0", "Xform")
    clone_plan_from_env_0(
        CloneCfg(clone_template="/World/Env_{}"), planned_cfgs, num_envs, _ENV_SPACING, positions=positions
    )
    assets = {}
    for name, cfg in asset_cfgs.items():
        if isinstance(cfg, ArticulationCfg):
            assets[name] = Articulation(cfg)
        elif isinstance(cfg, RigidObjectCollectionCfg):
            assets[name] = RigidObjectCollection(cfg)
        elif isinstance(cfg, RigidObjectCfg):
            assets[name] = RigidObject(cfg)
        else:
            raise TypeError(f"Unsupported asset configuration for '{name}': {type(cfg).__name__}.")
    replicate(SimulationContext.instance().get_clone_plan())
    return assets


@contextmanager
def world_gravity(gravity: Vec3) -> Iterator[None]:
    """Apply the same gravity to every world of the live Newton model, then restore the previous values.

    Args:
        gravity: Gravity applied to every world [m/s^2].
    """
    model = SimulationManager.get_model()
    world_gravity_view = wp.to_torch(model.gravity[: model.world_count])
    previous = world_gravity_view.clone()
    world_gravity_view.copy_(torch.tensor(gravity, device=previous.device))
    SimulationManager.add_model_change(ModelFlags.MODEL_PROPERTIES)
    try:
        yield
    finally:
        world_gravity_view.copy_(previous)
        SimulationManager.add_model_change(ModelFlags.MODEL_PROPERTIES)


##
# Fixture authoring.
##


def _body(
    name: str,
    *,
    mass: float,
    inertia: Vec3,
    com: Vec3 | None = None,
    translate: Vec3 | None = None,
    articulation_root: bool = False,
    collider_size: float | None = None,
) -> str:
    """Return a rigid body with explicit mass properties and, optionally, a child collider cube [m]."""
    applied = '"PhysicsRigidBodyAPI", "PhysicsMassAPI"'
    if articulation_root:
        applied += ', "PhysicsArticulationRootAPI"'
    lines = []
    if translate is not None:
        lines += [f"double3 xformOp:translate = {translate}", 'uniform token[] xformOpOrder = ["xformOp:translate"]']
    lines += [f"float physics:mass = {mass}", f"float3 physics:diagonalInertia = {inertia}"]
    if com is not None:
        lines.append(f"point3f physics:centerOfMass = {com}")
    content = "\n".join(f"        {line}" for line in lines)
    if collider_size is not None:
        content += (
            f'\n\n        def Cube "Collision" (\n            prepend apiSchemas = ["PhysicsCollisionAPI"]\n        )\n'
            f"        {{\n            double size = {collider_size}\n        }}"
        )
    return f'    def Xform "{name}" (\n        prepend apiSchemas = [{applied}]\n    )\n    {{\n{content}\n    }}\n'


def _joint(
    root: str,
    kind: str,
    name: str,
    body0: str | None,
    body1: str,
    *,
    axis: str | None = None,
    local_pos0: Vec3 | None = None,
    local_pos1: Vec3 | None = None,
    limits: tuple[float, float] | None = None,
    drive: tuple[str, float, float] | None = None,
) -> str:
    """Return a joint.

    ``body0=None`` attaches ``body1`` to the world, and ``drive`` is ``(kind, stiffness, damping)``.
    """
    schemas = f' (\n        prepend apiSchemas = ["PhysicsDriveAPI:{drive[0]}"]\n    )' if drive else ""
    lines = []
    if axis is not None:
        lines.append(f'uniform token physics:axis = "{axis}"')
    if body0 is not None:
        lines.append(f"rel physics:body0 = </{root}/{body0}>")
    lines.append(f"rel physics:body1 = </{root}/{body1}>")
    if local_pos0 is not None:
        lines.append(f"point3f physics:localPos0 = {local_pos0}")
    if local_pos1 is not None:
        lines.append(f"point3f physics:localPos1 = {local_pos1}")
    if limits is not None:
        lines += [f"float physics:lowerLimit = {limits[0]}", f"float physics:upperLimit = {limits[1]}"]
    if drive is not None:
        lines += [
            f"float drive:{drive[0]}:physics:stiffness = {drive[1]}",
            f"float drive:{drive[0]}:physics:damping = {drive[2]}",
        ]
    content = "\n".join(f"        {line}" for line in lines)
    return f'    def Physics{kind}Joint "{name}"{schemas}\n    {{\n{content}\n    }}\n'


def _document(root: str, prims: Sequence[str], *, articulation_root: bool = True) -> str:
    """Return a USD document whose default prim is an ``Xform`` holding ``prims``.

    ``articulation_root=False`` leaves the articulation root to one of the bodies.
    """
    header = f'#usda 1.0\n(\n    defaultPrim = "{root}"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'
    metadata = ' (\n    prepend apiSchemas = ["PhysicsArticulationRootAPI"]\n)' if articulation_root else ""
    return header + f'def Xform "{root}"{metadata}\n{{\n' + "\n".join(prims) + "}\n"


def _floating_two_link() -> str:
    """Two colliding links joined about z; the articulation root sits on the ``Root`` body."""
    return _document(
        "Robot",
        [
            _body(
                "Root",
                mass=1.0,
                inertia=(0.01, 0.02, 0.02),
                com=(0.0, 0.0, 0.0),
                articulation_root=True,
                collider_size=0.1,
            ),
            _body(
                "Child",
                mass=1.0,
                inertia=(0.01, 0.02, 0.02),
                com=(0.0, 0.0, 0.0),
                translate=(0.5, 0.0, 0.0),
                collider_size=0.1,
            ),
            _joint(
                "Robot",
                "Revolute",
                "Joint",
                "Root",
                "Child",
                axis="Z",
                local_pos0=(0.25, 0.0, 0.0),
                local_pos1=(-0.25, 0.0, 0.0),
                limits=(-90.0, 90.0),
            ),
        ],
        articulation_root=False,
    )


def _fixed_spatial_chain() -> str:
    """A fixed-base chain of three slides and a three-axis wrist whose links carry center-of-mass offsets.

    Every joint has a USD drive; the wrist's angular gains are per degree.
    """
    links = [
        _body("Root", mass=1.0, inertia=(0.01, 0.01, 0.01)),
        _joint("Robot", "Fixed", "RootJoint", None, "Root"),
    ]
    link_specs = [
        (0.5, (0.02, 0.02, 0.02), (0.02, 0.0, 0.0), None),
        (0.5, (0.02, 0.02, 0.02), (0.0, 0.02, 0.0), None),
        (0.5, (0.02, 0.02, 0.02), (0.0, 0.0, 0.02), None),
        (0.3, (0.01, 0.01, 0.01), (0.03, 0.0, 0.02), (0.0, 0.0, 0.1)),
        (0.3, (0.01, 0.01, 0.01), (0.0, 0.03, 0.02), (0.0, 0.0, 0.2)),
        (0.2, (0.01, 0.01, 0.01), (0.05, 0.02, 0.0), (0.0, 0.0, 0.3)),
    ]
    for index, (mass, inertia, com, translate) in enumerate(link_specs):
        links.append(_body(f"Link_{index}", mass=mass, inertia=inertia, com=com, translate=translate))
    parents = ["Root", "Link_0", "Link_1", "Link_2", "Link_3", "Link_4"]
    for index, axis in enumerate("XYZ"):
        links.append(
            _joint(
                "Robot",
                "Prismatic",
                f"Joint_{index}",
                parents[index],
                f"Link_{index}",
                axis=axis,
                limits=(-0.2, 0.2),
                drive=("linear", 200.0, 20.0),
            )
        )
    for index, (axis, stiffness) in enumerate(zip("XYZ", WRIST_USD_STIFFNESS), start=3):
        links.append(
            _joint(
                "Robot",
                "Revolute",
                f"Joint_{index}",
                parents[index],
                f"Link_{index}",
                axis=axis,
                local_pos0=(0.0, 0.0, 0.1),
                local_pos1=(0.0, 0.0, 0.0),
                limits=(-60.0, 60.0),
                drive=("angular", stiffness, round(0.1 * stiffness, 6)),
            )
        )
    return _document("Robot", links)


def _revolute_pendulum() -> str:
    """A fixed pivot and a unit arm turning about z."""
    return _document(
        "Robot",
        [
            _body("CenterPivot", mass=1.0, inertia=(0.01, 0.01, 0.01)),
            _joint("Robot", "Fixed", "FixedJoint", None, "CenterPivot"),
            _body("Arm", mass=1.0, inertia=(0.01, 0.01, 0.01), translate=(1.0, 0.0, 0.0)),
            _joint(
                "Robot",
                "Revolute",
                "RevoluteJoint",
                "CenterPivot",
                "Arm",
                axis="Z",
                local_pos0=(0.0, 0.0, 0.0),
                local_pos1=(-1.0, 0.0, 0.0),
            ),
        ],
    )


def _fixed_tendon_hand() -> str:
    """A fixed palm with two two-joint fingers; each fixed tendon ``XJ0`` sums ``XJ1`` and ``XJ2``.

    A MuJoCo position actuator drives each tendon.
    """
    prims = [
        _body("Palm", mass=0.5, inertia=(0.001, 0.001, 0.001)),
        _joint("Hand", "Fixed", "PalmJoint", None, "Palm"),
    ]
    for finger, y in (("FF", 0.02), ("MF", -0.02)):
        prims += [
            _body(
                f"{finger}_proximal",
                mass=0.05,
                inertia=(1e-5, 1e-5, 1e-5),
                com=(0.025, 0.0, 0.0),
                translate=(0.05, y, 0.0),
            ),
            _body(
                f"{finger}_distal", mass=0.03, inertia=(1e-5, 1e-5, 1e-5), com=(0.02, 0.0, 0.0), translate=(0.1, y, 0.0)
            ),
            _joint(
                "Hand",
                "Revolute",
                f"{finger}J2",
                "Palm",
                f"{finger}_proximal",
                axis="Y",
                local_pos0=(0.05, y, 0.0),
                local_pos1=(0.0, 0.0, 0.0),
                limits=(0.0, 90.0),
            ),
            _joint(
                "Hand",
                "Revolute",
                f"{finger}J1",
                f"{finger}_proximal",
                f"{finger}_distal",
                axis="Y",
                local_pos0=(0.05, 0.0, 0.0),
                local_pos1=(0.0, 0.0, 0.0),
                limits=(0.0, 90.0),
            ),
            f'    def MjcTendon "{finger}J0"\n    {{\n        uniform token mjc:type = "fixed"\n'
            f"        rel mjc:path = [</Hand/{finger}J1>, </Hand/{finger}J2>]\n"
            "        uniform int[] mjc:path:indices = [0, 1]\n        uniform double[] mjc:path:coef = [1, 1]\n    }\n",
            f'    def MjcActuator "{finger}_tendon_actuator"\n    {{\n        uniform token mjc:biasType = "affine"\n'
            "        uniform double[] mjc:gainPrm = [0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
            "        uniform double[] mjc:biasPrm = [0, -0.05, -0.005, 0, 0, 0, 0, 0, 0, 0]\n"
            f"        rel mjc:target = </Hand/{finger}J0>\n    }}\n",
        ]
    return _document("Hand", prims)


def _floating_two_leg() -> str:
    """A floating base with two three-joint legs named like ANYmal's (``HAA``, ``HFE``, ``KFE``)."""
    prims = [_body("base", mass=10.0, inertia=(0.5, 0.5, 0.5), com=(0.0, 0.0, 0.0))]
    joints = []
    for leg, x, y in (("LF", 0.3, 0.1), ("RH", -0.3, -0.1)):
        prims += [
            _body(f"{leg}_HIP", mass=1.0, inertia=(0.05, 0.05, 0.05), com=(0.0, 0.02, 0.0), translate=(x, y, 0.0)),
            _body(f"{leg}_THIGH", mass=1.0, inertia=(0.05, 0.05, 0.05), com=(0.0, 0.0, -0.1), translate=(x, y, -0.05)),
            _body(f"{leg}_SHANK", mass=0.5, inertia=(0.05, 0.05, 0.05), com=(0.0, 0.0, -0.1), translate=(x, y, -0.3)),
        ]
        joints += [
            _joint(
                "Robot",
                "Revolute",
                f"{leg}_HAA",
                "base",
                f"{leg}_HIP",
                axis="X",
                local_pos0=(x, y, 0.0),
                local_pos1=(0.0, 0.0, 0.0),
                limits=(-45.0, 45.0),
            ),
            _joint(
                "Robot",
                "Revolute",
                f"{leg}_HFE",
                f"{leg}_HIP",
                f"{leg}_THIGH",
                axis="Y",
                local_pos0=(0.0, 0.0, -0.05),
                local_pos1=(0.0, 0.0, 0.0),
                limits=(-90.0, 90.0),
            ),
            _joint(
                "Robot",
                "Revolute",
                f"{leg}_KFE",
                f"{leg}_THIGH",
                f"{leg}_SHANK",
                axis="Y",
                local_pos0=(0.0, 0.0, -0.25),
                local_pos1=(0.0, 0.0, 0.0),
                limits=(-160.0, 0.0),
            ),
        ]
    return _document("Robot", prims + joints)


def _fixed_cartpole() -> str:
    """A cart sliding on a fixed rail with a pole on a revolute joint."""
    return _document(
        "Robot",
        [
            _body("rail", mass=1.0, inertia=(0.01, 0.01, 0.01)),
            _joint("Robot", "Fixed", "rail_joint", None, "rail"),
            _body("cart", mass=1.0, inertia=(0.01, 0.01, 0.01)),
            _body("pole", mass=0.2, inertia=(0.01, 0.01, 0.01), com=(0.0, 0.0, 0.5)),
            _joint("Robot", "Prismatic", "slider_to_cart", "rail", "cart", axis="X", limits=(-2.0, 2.0)),
            _joint("Robot", "Revolute", "cart_to_pole", "cart", "pole", axis="Y"),
        ],
    )


def _rigid_body_with_articulation_root() -> str:
    """A rigid body carrying an articulation root, with a second body fixed to it."""
    return """#usda 1.0
(
    defaultPrim = "Object"
    metersPerUnit = 1
    upAxis = "Z"
)

def Cube "Object" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI", "PhysicsArticulationRootAPI"]
)
{
    double size = 0.1
    float physics:mass = 1

    def Cube "Child" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]
    )
    {
        double size = 0.1
        double3 xformOp:translate = (0, 0, 2)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 1
    }

    def PhysicsFixedJoint "Joint"
    {
        rel physics:body0 = </Object>
        rel physics:body1 = </Object/Child>
    }
}
"""


_FIXTURES: dict[str, Callable[[], str]] = {
    "fixed_cartpole.usda": _fixed_cartpole,
    "fixed_spatial_chain.usda": _fixed_spatial_chain,
    "fixed_tendon_hand.usda": _fixed_tendon_hand,
    "floating_two_leg.usda": _floating_two_leg,
    "floating_two_link.usda": _floating_two_link,
    "revolute_pendulum.usda": _revolute_pendulum,
    "rigid_body_with_articulation_root.usda": _rigid_body_with_articulation_root,
}
"""Authored fixtures by file name."""


@functools.cache
def _fixture_dir() -> Path:
    """Return this process's directory for the authored fixtures, removed when the process exits."""
    directory = Path(tempfile.mkdtemp(prefix="isaaclab_newton_test_fixtures_"))
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    return directory
