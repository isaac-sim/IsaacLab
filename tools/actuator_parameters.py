# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script compares actuator parameters on a procedurally-authored pendulum.

The scene is a row of five identical single-joint pendulums. Each pendulum is
driven by an actuator model that differs only in the parameter under study
(stiffness, damping, armature, joint friction, effort limit, velocity limit,
command delay, or implicit-vs-explicit control). Sweeping the parameter across
the five instances and driving them with the same command profile makes the
effect of that parameter directly visible side by side.

The pendulum is authored procedurally with USD APIs (no external asset files),
so the demo is self-contained and doubles as documentation for the actuator
runtime API. Commands are always issued through the actuator collection
(``articulation.actuators.command.set_*_index``).

.. code-block:: bash

    # List the available parameter comparisons.
    ./isaaclab.sh -p tools/actuator_parameters.py --list_parameters

    # Interactive comparison of the stiffness sweep with the Newton visualizer.
    ./isaaclab.sh -p tools/actuator_parameters.py --parameter stiffness

    # Compare the effort-limit sweep.
    ./isaaclab.sh -p tools/actuator_parameters.py --parameter effort-limit

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="Compare actuator parameters on a procedurally-authored pendulum.",
    conflict_handler="resolve",
)
parser.add_argument(
    "--parameter",
    default="stiffness",
    help="Actuator parameter to compare. Use --list_parameters to see the available keys.",
)
parser.add_argument(
    "--list_parameters",
    action="store_true",
    help="List the available parameter comparisons and exit.",
)
parser.add_argument(
    "--physics",
    default="newton_mjwarp",
    choices=["physx", "newton_mjwarp"],
    help="Physics backend.",
)
parser.add_argument(
    "--record",
    action="store_true",
    help="Render an animated clip and plot the comparison curves into the docs media directory.",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="With --record, generate media for every comparison row instead of just --parameter.",
)
# Hidden flag used by CI: run a fixed number of steps then exit cleanly.
parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton"])
args_cli = parser.parse_args()

import contextlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.actuators import (
    ActuatorBaseCfg,
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuatorCfg,
    ImplicitActuatorCfg,
)
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.physics import PhysicsCfg

##
# Simulation constants (pinned for determinism and reproducible media).
##

SEED = 0
"""Random seed applied before scene construction."""

DT = 1.0 / 360.0
"""Physics step size [s].

Finer than the usual 1/120 so fast transients (e.g. the effort-limit swing-up
overshoot) are resolved rather than numerically flattened.
"""

DECIMATION = 6
"""Number of physics steps between successive control commands (60 Hz commands)."""

CAPSULE_LENGTH = 0.6
"""Length of the pendulum capsule link [m]."""

CAPSULE_RADIUS = 0.05
"""Radius of the pendulum capsule link [m]."""

CAPSULE_MASS = 1.0
"""Mass of the pendulum capsule link [kg]."""

PIVOT_HEIGHT = 2.0
"""Height of the revolute pivot above the ground plane [m]."""

INSTANCE_SPACING = 1.5
"""Spacing between neighbouring pendulum instances along the x-axis [m]."""

SMOKE_STEPS = 20
"""Number of physics steps executed in ``--smoke`` mode before a clean exit."""

DRIVE_STIFFNESS = 100.0
"""Placeholder USD drive stiffness so the joint imports in position-actuation mode [N*m/rad]."""

DRIVE_DAMPING = 5.0
"""Placeholder USD drive damping so the joint imports in position-actuation mode [N*m*s/rad]."""


##
# Recording constants (media generation for the documentation).
##

MEDIA_DIR = Path(__file__).resolve().parents[1] / "docs" / "source" / "_static" / "actuators"
"""Output directory for the generated clips and curves."""

CAMERA_WIDTH = 1280
"""Width of the captured RGB frames [px]."""

CAMERA_HEIGHT = 400
"""Height of the captured RGB frames [px].

The pendulums live in a narrow horizontal band, so the clips use a wide, short
aspect ratio; a 16:9 frame would waste most of its pixels on sky and ground.
"""

CAPTURE_STRIDE = 12
"""Capture one frame every this many physics steps (30 fps real-time playback)."""

CAMERA_EYE = (0.0, -9.0, 1.7)
"""World-space eye position of the recording camera [m]."""

CAMERA_TARGET = (0.0, 0.0, 1.7)
"""World-space look-at target of the recording camera [m]."""

CLIP_MAX_BYTES = 1_200_000
"""Per-clip webp size budget [bytes]."""

RECORD_HFOV_DEG = 47.2
"""Horizontal field of view of the recording camera [deg]."""

PIVOT_MARKER_COLOR = (1.0, 0.127, 0.847)
"""Display color of the joint pivot marker spheres: ``#ff63ed`` decoded to linear RGB."""

PIVOT_MARKER_RADIUS = 0.09
"""Radius of the joint pivot marker spheres [m]."""

SVG_HASHSALT = "isaaclab-actuator-parameters"
"""Fixed matplotlib SVG hash salt so generated curves are byte-reproducible."""

TRACE_COLORS = ("#3B82C4", "#E8833A", "#3EA96B", "#D14B57", "#8E63C4")
"""Mid-tone categorical palette that reads on both light and dark backgrounds."""

VALUE_UNITS = {
    "stiffness": "N·m/rad",
    "damping": "N·m·s/rad",
    "armature": "kg·m²",
    "friction": "N·m",
    "effort-limit": "N·m",
    "velocity-limit": "rad/s",
    "delay": "",
    "implicit-vs-explicit": "",
}
"""Per-row SI unit appended to numeric legend labels (empty for label-valued rows)."""


##
# Comparison matrix definitions.
##


@dataclass
class PlotSpec:
    """Plotting metadata consumed by the recording tooling.

    Attributes:
        quantity: Logged quantity to plot. One of ``joint_pos``, ``joint_vel``,
            or ``applied_torque``.
        title: Human-readable plot title.
        ylabel: Label for the plot's y-axis.
    """

    quantity: str
    title: str
    ylabel: str


@dataclass
class RowSpec:
    """A single actuator-parameter comparison.

    Attributes:
        name: Human-readable description of the comparison.
        values: Per-instance parameter values (five entries). ``None`` entries
            are padded placeholders for comparisons that use fewer than five live
            instances; non-numeric entries act as labels.
        make_actuator_cfg: Factory turning one entry of :attr:`values` into an
            actuator configuration.
        command: Command profile mapping simulation time [s] to per-step position
            [rad], velocity [rad/s], and effort [N*m] targets. ``None`` components
            are not commanded.
        initial_joint_pos: Initial joint position applied on reset [rad].
        initial_joint_vel: Initial joint velocity applied on reset [rad/s].
        duration: Duration of the finite comparison run [s].
        plot: Plotting metadata.
        record_clip: Whether record mode captures an animated clip for this row.
            Curve-only rows (behaviors identical by design, or analytic plots)
            set this to ``False``.
    """

    name: str
    values: list[float | str | None]
    make_actuator_cfg: Callable[[float | str], ActuatorBaseCfg]
    command: Callable[[float], tuple[float | None, float | None, float | None]]
    initial_joint_pos: float
    initial_joint_vel: float
    duration: float
    plot: PlotSpec
    record_clip: bool = True


# Common command profiles. ``t`` is the elapsed simulation time in seconds.


def _step_to_horizontal(t: float) -> tuple[float | None, float | None, float | None]:
    """Position step from 0 to pi/2 [rad] at t = 0.5 s."""
    return (0.0 if t < 0.5 else math.pi / 2.0, None, None)


def _hold_horizontal(t: float) -> tuple[float | None, float | None, float | None]:
    """Constant position target at pi/2 [rad] (horizontal hold against gravity)."""
    return (math.pi / 2.0, None, None)


def _square_wave(t: float) -> tuple[float | None, float | None, float | None]:
    """Position square wave between 0 and pi/2 [rad] with a 2 s period."""
    return (0.0 if (t % 2.0) < 1.0 else math.pi / 2.0, None, None)


def _free(t: float) -> tuple[float | None, float | None, float | None]:
    """No command; the joint swings freely."""
    return (None, None, None)


def _implicit_pendulum_cfg(**kwargs) -> ImplicitActuatorCfg:
    """Build an implicit actuator config for the pendulum joint."""
    return ImplicitActuatorCfg(joint_names_expr=[".*"], **kwargs)


def _use_implicit_integrator(physics_cfg) -> None:
    """Switch the MJWarp solver to its implicit-derivative integrator.

    MJWarp's default ``euler`` integrator applies actuator velocity gains
    explicitly, which goes unstable once ``damping > 2 * inertia / dt`` (about
    87 N·m·s/rad for this pendulum at 360 Hz), which the damping sweep's top
    value crosses. The ``implicitfast`` integrator folds the actuator force
    derivatives into the velocity update, making the implicit PD gain sweeps
    unconditionally stable without changing their values. No-op for other
    physics backends.

    Args:
        physics_cfg: Resolved physics configuration from ``launch_simulation``.
    """
    solver_cfg = getattr(physics_cfg, "solver_cfg", None)
    if solver_cfg is not None and hasattr(solver_cfg, "integrator"):
        solver_cfg.integrator = "implicitfast"


ROWS: dict[str, RowSpec] = {
    "stiffness": RowSpec(
        name="Implicit PD stiffness sweep (damping fixed at 5)",
        values=[10.0, 40.0, 100.0, 400.0, 1000.0],
        make_actuator_cfg=lambda v: _implicit_pendulum_cfg(stiffness=v, damping=5.0),
        command=_step_to_horizontal,
        initial_joint_pos=0.0,
        initial_joint_vel=0.0,
        duration=4.0,
        plot=PlotSpec("joint_pos", "Stiffness sweep", "Joint position [rad]"),
    ),
    "damping": RowSpec(
        name="Implicit PD damping sweep (stiffness fixed at 100)",
        values=[0.5, 2.0, 10.0, 40.0, 100.0],
        make_actuator_cfg=lambda v: _implicit_pendulum_cfg(stiffness=100.0, damping=v),
        command=_step_to_horizontal,
        initial_joint_pos=0.0,
        initial_joint_vel=0.0,
        duration=4.0,
        plot=PlotSpec("joint_pos", "Damping sweep", "Joint position [rad]"),
    ),
    "armature": RowSpec(
        name="Joint armature sweep (stiffness 100, damping 5)",
        values=[0.0, 0.01, 0.05, 0.2, 1.0],
        make_actuator_cfg=lambda v: _implicit_pendulum_cfg(stiffness=100.0, damping=5.0, armature=v),
        command=_step_to_horizontal,
        initial_joint_pos=0.0,
        initial_joint_vel=0.0,
        duration=4.0,
        plot=PlotSpec("joint_pos", "Armature sweep", "Joint position [rad]"),
    ),
    "friction": RowSpec(
        name="Joint friction sweep on a free-spinning joint",
        values=[0.0, 0.05, 0.2, 0.5, 1.0],
        make_actuator_cfg=lambda v: _implicit_pendulum_cfg(stiffness=0.0, damping=0.0, friction=v),
        command=_free,
        initial_joint_pos=0.0,
        initial_joint_vel=4.0 * math.pi,
        duration=6.0,
        plot=PlotSpec("joint_vel", "Joint friction sweep", "Joint velocity [rad/s]"),
    ),
    "effort-limit": RowSpec(
        name="Ideal PD effort-limit sweep (step from hanging to horizontal)",
        # Holding horizontal against gravity takes ~2.9 N·m. Stepping up from
        # the hanging pose, limits below that saturate and sag where gravity
        # torque (~2.9 sin(theta)) matches the limit, oscillating while clipped
        # (a saturated actuator has no torque headroom left for its damping
        # term); limits above it reach horizontal, faster with more headroom.
        values=[1.0, 2.0, 3.0, 4.0, 6.0],
        make_actuator_cfg=lambda v: IdealPDActuatorCfg(
            joint_names_expr=[".*"], stiffness=400.0, damping=20.0, effort_limit=v
        ),
        command=_step_to_horizontal,
        initial_joint_pos=0.0,
        initial_joint_vel=0.0,
        duration=4.0,
        plot=PlotSpec("applied_torque", "Effort-limit sweep", "Applied joint torque [N·m]"),
    ),
    "velocity-limit": RowSpec(
        name="DC motor velocity-limit sweep (torque-speed envelope)",
        values=[2.0, 4.0, 8.0, 16.0, 32.0],
        make_actuator_cfg=lambda v: DCMotorCfg(
            joint_names_expr=[".*"],
            stiffness=100.0,
            damping=5.0,
            saturation_effort=12.0,
            effort_limit=6.0,
            velocity_limit=v,
        ),
        command=_step_to_horizontal,
        initial_joint_pos=0.0,
        initial_joint_vel=0.0,
        duration=4.0,
        plot=PlotSpec("joint_vel", "Velocity-limit sweep", "Joint velocity [rad/s]"),
    ),
    "delay": RowSpec(
        name="Command delay sweep (delayed PD, 0 to 133 ms)",
        # Delays are counted in physics steps (1/360 s); these match the
        # 0/17/33/67/133 ms sweep of the original 120 Hz parametrization.
        values=[0, 6, 12, 24, 48],
        make_actuator_cfg=lambda v: DelayedPDActuatorCfg(
            joint_names_expr=[".*"], stiffness=100.0, damping=5.0, min_delay=int(v), max_delay=int(v)
        ),
        command=_square_wave,
        initial_joint_pos=0.0,
        initial_joint_vel=0.0,
        duration=6.0,
        plot=PlotSpec("joint_pos", "Command delay sweep", "Joint position [rad]"),
    ),
    "implicit-vs-explicit": RowSpec(
        name="Implicit vs explicit PD (identical stiffness 100, damping 5)",
        values=["implicit", "explicit", None, None, None],
        make_actuator_cfg=lambda v: (
            ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=5.0)
            if v == "implicit"
            else IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=5.0)
        ),
        command=_step_to_horizontal,
        initial_joint_pos=0.0,
        initial_joint_vel=0.0,
        duration=4.0,
        plot=PlotSpec("joint_pos", "Implicit vs explicit PD", "Joint position [rad]"),
        # The two responses are identical by design, so a clip adds nothing; the
        # overlaid curve is the teaching artifact.
        record_clip=False,
    ),
}


##
# Procedural pendulum authoring.
##


def _author_pendulum(stage, prim_path: str, x_offset: float) -> None:
    """Author a single-joint pendulum directly on the USD stage.

    The pendulum is a fixed-base articulation. A massless base link is welded to
    the world with a fixed joint at the pivot, and the swinging capsule link is
    attached to the base with a revolute joint about the world y-axis. A position
    target of pi/2 [rad] therefore raises the capsule from its resting (downward)
    pose to horizontal.

    The explicit base-plus-fixed-joint structure (rather than a bare revolute
    joint to the world frame) is what the physics importers recognise as a
    fixed-base articulation rooted at :paramref:`prim_path`.

    Args:
        stage: USD stage to author on.
        prim_path: Articulation root prim path.
        x_offset: World-space x-offset of the pivot [m].
    """
    pivot = Gf.Vec3d(x_offset, 0.0, PIVOT_HEIGHT)
    identity_rot = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

    # Articulation root.
    root = UsdGeom.Xform.Define(stage, prim_path)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    # Base link, welded to the world frame at the pivot.
    base_path = f"{prim_path}/base"
    base = UsdGeom.Xform.Define(stage, base_path)
    base.AddTranslateOp().Set(pivot)
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    base_mass = UsdPhysics.MassAPI.Apply(base.GetPrim())
    base_mass.CreateMassAttr(CAPSULE_MASS)
    # The base is welded to the world, so its inertia is irrelevant; author an
    # explicit diagonal to avoid the physics engine's invalid-inertia warning.
    base_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.01, 0.01))
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/fixed")
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(base_path)])
    fixed_joint.CreateLocalPos0Attr(Gf.Vec3f(x_offset, 0.0, PIVOT_HEIGHT))
    fixed_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))

    # Swinging capsule link (rigid body) placed at the pivot.
    link_path = f"{prim_path}/link"
    link = UsdGeom.Xform.Define(stage, link_path)
    link.AddTranslateOp().Set(pivot)
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(CAPSULE_MASS)

    # Capsule geometry offset so its centre of mass hangs below the pivot.
    capsule = UsdGeom.Capsule.Define(stage, f"{link_path}/geom")
    capsule.CreateAxisAttr("Z")
    capsule.CreateRadiusAttr(CAPSULE_RADIUS)
    capsule.CreateHeightAttr(CAPSULE_LENGTH)
    capsule.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -CAPSULE_LENGTH / 2.0))
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())

    # Revolute joint connecting the base to the swinging link.
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/joint")
    joint.CreateAxisAttr("Y")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(base_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(link_path)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(identity_rot)
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(identity_rot)

    # Angular drive so the actuator runtime can push gains into the solver. The
    # non-zero stiffness/damping put the joint into position-actuation mode at
    # import time; the actual gains are overwritten per instance by the actuator
    # configuration, but a zero-gain drive would import as an unactuated joint.
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateMaxForceAttr(1.0e9)
    drive.CreateStiffnessAttr(DRIVE_STIFFNESS)
    drive.CreateDampingAttr(DRIVE_DAMPING)


def _live_values(row: RowSpec) -> list[float | str]:
    """Return the non-``None`` entries of :attr:`RowSpec.values`."""
    return [value for value in row.values if value is not None]


def build_scene(sim: sim_utils.SimulationContext, row: RowSpec) -> list[Articulation]:
    """Author the pendulum scene for a comparison row.

    A ground plane and dome light are added for visualization, then one pendulum
    is authored per live entry of :attr:`RowSpec.values`. Each pendulum receives
    its own actuator configuration built from that entry.

    Args:
        sim: Active simulation context.
        row: Comparison row to build.

    Returns:
        The spawned articulations, one per live parameter value, in value order.
    """
    stage = sim_utils.get_current_stage()

    # Ground plane and lighting for interactive viewing.
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    # Give the ground a neutral color in the Newton viewer. The grid asset's
    # own material is not translated to Newton shape colors (it is not OmniPBR),
    # which leaves Newton's default green; unbinding it and authoring a display
    # color routes through the supported color path instead.
    ground_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
    for prim in Usd.PrimRange(ground_prim):
        # Block the asset's material binding on every prim (bindings inherit
        # from ancestors, so blocking only the geometry prims is not enough).
        UsdShade.MaterialBindingAPI.Apply(prim).UnbindAllBindings()
        if prim.IsA(UsdGeom.Gprim):
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.27, 0.31)])
            # The asset's collision plane is a guide-purpose prim, which the
            # Newton shape-color translation skips; make it a default-purpose
            # prim so the display color above takes effect.
            gprim.CreatePurposeAttr(UsdGeom.Tokens.default_)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    values = _live_values(row)
    num_instances = len(values)
    articulations: list[Articulation] = []
    for index, value in enumerate(values):
        x_offset = (index - (num_instances - 1) / 2.0) * INSTANCE_SPACING
        prim_path = f"/World/Pendulum_{index}"
        _author_pendulum(stage, prim_path, x_offset)

        cfg = ArticulationCfg(
            prim_path=prim_path,
            spawn=None,
            actuators={"joint": row.make_actuator_cfg(value)},
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={".*": row.initial_joint_pos},
                joint_vel={".*": row.initial_joint_vel},
            ),
        )
        articulations.append(cfg.class_type(cfg))

    return articulations


##
# Simulation loop and logging.
##


def _reset_pendulums(articulations: list[Articulation], row: RowSpec) -> None:
    """Reset every pendulum to the row's initial joint state."""
    for articulation in articulations:
        shape = (articulation.num_instances, articulation.num_joints)
        joint_pos = torch.full(shape, row.initial_joint_pos, dtype=torch.float32, device=articulation.device)
        joint_vel = torch.full(shape, row.initial_joint_vel, dtype=torch.float32, device=articulation.device)
        articulation.write_joint_position_to_sim_index(position=joint_pos)
        articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)
        articulation.reset()


def _apply_command(articulation: Articulation, command: tuple[float | None, float | None, float | None]) -> None:
    """Issue a command tuple to one pendulum through the actuator collection."""
    position, velocity, effort = command
    shape = (articulation.num_instances, articulation.num_joints)
    if position is not None:
        target = torch.full(shape, position, dtype=torch.float32, device=articulation.device)
        articulation.actuators.command.set_position_index(value=target)
    if velocity is not None:
        target = torch.full(shape, velocity, dtype=torch.float32, device=articulation.device)
        articulation.actuators.command.set_velocity_index(value=target)
    if effort is not None:
        target = torch.full(shape, effort, dtype=torch.float32, device=articulation.device)
        articulation.actuators.command.set_effort_index(value=target)


def run_row(
    sim: sim_utils.SimulationContext,
    articulations: list[Articulation],
    row: RowSpec,
    on_frame: Callable[[int], None] | None = None,
    num_steps: int | None = None,
) -> dict[str, torch.Tensor]:
    """Step the scene for one comparison row and log per-step telemetry.

    The row's command profile is issued through the actuator collection every
    :data:`DECIMATION` physics steps. Joint state and actuator torques are logged
    every physics step. Logged tensors always have five columns; columns for
    ``None`` entries of :attr:`RowSpec.values` are filled with ``NaN``.

    Args:
        sim: Active simulation context.
        articulations: Pendulums returned by :func:`build_scene`, in value order.
        row: Comparison row being run.
        on_frame: Optional callback invoked with the physics step index after each
            step (used by the recording tooling for camera capture).
        num_steps: Number of physics steps to run. Defaults to the number of steps
            spanning :attr:`RowSpec.duration`.

    Returns:
        Mapping with keys ``joint_pos``, ``joint_vel``, ``applied_torque``, and
        ``computed_torque``. Each value is a tensor of shape ``[num_steps, 5]``.
    """
    sim_dt = sim.get_physics_dt()
    if num_steps is None:
        num_steps = int(round(row.duration / sim_dt))

    live_columns = [index for index, value in enumerate(row.values) if value is not None]

    logs: dict[str, list[torch.Tensor]] = {
        "joint_pos": [],
        "joint_vel": [],
        "applied_torque": [],
        "computed_torque": [],
    }

    _reset_pendulums(articulations, row)
    sim_time = 0.0
    for step in range(num_steps):
        if step % DECIMATION == 0:
            command = row.command(sim_time)
            for articulation in articulations:
                _apply_command(articulation, command)
        for articulation in articulations:
            articulation.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        for articulation in articulations:
            articulation.update(sim_dt)

        _record_step(logs, articulations, live_columns)
        if on_frame is not None:
            on_frame(step)

    return {key: torch.stack(values) for key, values in logs.items()}


def _record_step(
    logs: dict[str, list[torch.Tensor]],
    articulations: list[Articulation],
    live_columns: list[int],
) -> None:
    """Append one row of telemetry (five columns, NaN-padded) to *logs*."""
    sources = {
        "joint_pos": lambda a: a.data.joint_pos.torch,
        "joint_vel": lambda a: a.data.joint_vel.torch,
        "applied_torque": lambda a: a.actuators.applied_torque.torch,
        "computed_torque": lambda a: a.actuators.computed_torque.torch,
    }
    for key, read in sources.items():
        frame = torch.full((5,), float("nan"))
        for articulation, column in zip(articulations, live_columns):
            frame[column] = read(articulation).reshape(-1)[0].detach().cpu()
        logs[key].append(frame)


def _run_interactive(
    sim: sim_utils.SimulationContext,
    articulations: list[Articulation],
    row: RowSpec,
) -> None:
    """Loop the row's command profile forever while a visualizer is open."""
    sim_dt = sim.get_physics_dt()
    _reset_pendulums(articulations, row)
    sim_time = 0.0
    step = 0
    while sim.is_headless_or_exist_active_visualizer():
        if sim_time >= row.duration:
            _reset_pendulums(articulations, row)
            sim_time = 0.0
            step = 0
        if step % DECIMATION == 0:
            command = row.command(sim_time)
            for articulation in articulations:
                _apply_command(articulation, command)
        for articulation in articulations:
            articulation.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        step += 1
        for articulation in articulations:
            articulation.update(sim_dt)


def _print_parameters() -> None:
    """Print the available comparison keys and their descriptions."""
    print("Available actuator parameter comparisons:")
    width = max(len(key) for key in ROWS)
    for key, row in ROWS.items():
        print(f"  {key:<{width}}  {row.name}")


##
# Media recording (clips) and plotting (curves).
##


def _author_row_markers(stage, prim_path: str) -> None:
    """Author a marker sphere at a pendulum's pivot (rotation center).

    The sphere goes under the welded base link so the Newton importer includes
    it with that body. It is visual-only: it carries no collision API, and the
    base and swinging link are adjacent bodies whose contacts are excluded.

    Args:
        stage: USD stage to author on.
        prim_path: Articulation root prim path of the pendulum.
    """
    pivot = UsdGeom.Sphere.Define(stage, f"{prim_path}/base/pivot_marker")
    pivot.CreateRadiusAttr(PIVOT_MARKER_RADIUS)
    pivot.CreateDisplayColorAttr([Gf.Vec3f(*PIVOT_MARKER_COLOR)])


_RECORDER = None
"""Shared clip recorder; one headless GL viewer reused across every recorded row."""


def _get_record_recorder() -> "NewtonGlPerspectiveVideo":  # noqa: F821
    """Return the headless Newton GL viewer used to capture the clips.

    :class:`~isaaclab_newton.video_recording.NewtonGlPerspectiveVideo` wraps
    ``newton.viewer.ViewerGL`` in headless (EGL) mode, so the clips get the
    Newton viewer's default look — sky background, ground grid, and house
    lighting — without any hand-authored backdrop. Call after
    :meth:`SimulationContext.reset` so the Newton model exists.

    A single viewer is created for the whole process and re-bound to the
    current row's Newton model on each call: recreating the GL context after
    closing a previous viewer renders blank frames, so ``--record --all`` must
    reuse one context across rows.

    Returns:
        The recorder, aimed at the pendulum row and bound to its model.
    """
    global _RECORDER
    from isaaclab_newton.video_recording import NewtonGlPerspectiveVideo, NewtonGlPerspectiveVideoCfg

    if _RECORDER is None:
        _RECORDER = NewtonGlPerspectiveVideo(
            NewtonGlPerspectiveVideoCfg(
                window_width=CAMERA_WIDTH,
                window_height=CAMERA_HEIGHT,
                eye=CAMERA_EYE,
                lookat=CAMERA_TARGET,
                horiz_fov_deg=RECORD_HFOV_DEG,
            )
        )
        # Creates the viewer bound to the current Newton model.
        _RECORDER.update_camera(CAMERA_EYE, CAMERA_TARGET)
    else:
        from isaaclab_newton.physics import NewtonManager

        _RECORDER._viewer.set_model(NewtonManager.get_model())
        _RECORDER.update_camera(CAMERA_EYE, CAMERA_TARGET)
    return _RECORDER


def _strip_webp_dispose_flags(path: Path) -> None:
    """Clear the dispose-to-background flag on every animation frame of *path*.

    libwebp marks long-held frames (such as the merged initial rest pose) with
    dispose-to-background, which players render as a washed-out flash: the
    canvas is cleared and the next frame only paints its changed sub-rectangle.
    In these clips a frame's unchanged pixels always match the previous canvas
    content, so keeping the canvas (dispose "none") renders identically without
    the flash.
    """
    data = bytearray(path.read_bytes())
    pos = 12  # skip the RIFF header and the WEBP fourcc
    while pos + 8 <= len(data):
        tag = bytes(data[pos : pos + 4])
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        if tag == b"ANMF":
            data[pos + 8 + 15] &= 0xFD
        pos += 8 + size + (size & 1)
    path.write_bytes(bytes(data))


def _save_webp(path: Path, frames: list[Image.Image], duration_ms: int, quality: int) -> int:
    """Write *frames* as an animated webp and return the resulting file size [bytes]."""
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        method=6,
        quality=quality,
    )
    _strip_webp_dispose_flags(path)
    return path.stat().st_size


def _write_clip(frames: list[Image.Image], key: str) -> Path:
    """Encode *frames* to ``<key>-clip.webp``, shrinking until under the size budget.

    Applies the fallback ladder from most to least fidelity: quality 80, then
    70, then 55, then half the frame rate (dropping every other frame), then a
    960x540 downscale. Playback stays real-time because the per-frame duration
    is doubled whenever the frame rate is halved.

    Args:
        frames: Captured RGB frames in playback order.
        key: Comparison-row key used for the output filename.

    Returns:
        The path to the written clip.
    """
    path = MEDIA_DIR / f"{key}-clip.webp"
    duration_ms = int(round(1000.0 * CAPTURE_STRIDE * DT))

    size = _save_webp(path, frames, duration_ms, quality=80)
    for quality in (70, 55):
        if size > CLIP_MAX_BYTES:
            size = _save_webp(path, frames, duration_ms, quality=quality)
    if size > CLIP_MAX_BYTES:
        frames = frames[::2]
        duration_ms *= 2
        size = _save_webp(path, frames, duration_ms, quality=55)
    if size > CLIP_MAX_BYTES:
        frames = [frame.resize((960, 540), Image.LANCZOS) for frame in frames]
        size = _save_webp(path, frames, duration_ms, quality=55)
    if size > CLIP_MAX_BYTES:
        print(f"[WARN]: {path.name} is {size / 1000:.1f} KB, still above the {CLIP_MAX_BYTES // 1000} KB budget.")
    return path


def _clip_label(key: str, value: float | str) -> str:
    """Build a short, unit-less caption for one pendulum column (e.g. ``stiffness = 40``)."""
    if isinstance(value, str):
        return value
    if key == "delay":
        return f"delay = {int(value)} steps"
    return f"{key} = {_format_value(value)}"


def _annotate_frames(frames: list[Image.Image], row: RowSpec, key: str) -> list[Image.Image]:
    """Draw each pendulum's parameter value below its column on every frame.

    Captions sit in the bottom band of the frame (over the static ground, clear
    of the swing plane), white with a dark outline, sized proportionally to the
    frame width, and centered under each pendulum's column using the pinhole
    projection of its world x-offset.

    Args:
        frames: Captured RGB frames (modified in place).
        row: Comparison row being recorded.
        key: Comparison-row key used for the caption text.

    Returns:
        The annotated frames.
    """
    from matplotlib import font_manager
    from PIL import ImageDraw, ImageFont

    height, width = frames[0].height, frames[0].width
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size=max(12, round(0.02 * width)))

    # Project each pendulum's pivot x-offset through the recording pinhole camera
    # (eye x = target x = 0, so the image x only depends on the world x-offset).
    live = [(column, value) for column, value in enumerate(row.values) if value is not None]
    num_instances = len(live)
    distance = abs(CAMERA_EYE[1])
    half_width_at_pivots = distance * math.tan(math.radians(RECORD_HFOV_DEG) / 2.0)
    y_text = height - round(0.06 * height)

    for image in frames:
        draw = ImageDraw.Draw(image)
        for slot, (_, value) in enumerate(live):
            x_offset = (slot - (num_instances - 1) / 2.0) * INSTANCE_SPACING
            x_frac = 0.5 + 0.5 * x_offset / half_width_at_pivots
            draw.text(
                (round(x_frac * width), y_text),
                _clip_label(key, value),
                font=font,
                fill=(255, 255, 255),
                anchor="ms",
                stroke_width=2,
                stroke_fill=(25, 30, 40),
            )
    return frames


@contextlib.contextmanager
def _mpl_style(dark: bool) -> Iterator[None]:
    """Apply a transparent-background matplotlib style for a light or dark curve.

    Args:
        dark: Whether to use foreground colors legible on a dark page background.

    Yields:
        None; the style is active for the duration of the context.
    """
    foreground = "#E6E6E6" if dark else "#1A1A1A"
    grid = "#6E6E6E" if dark else "#BFBFBF"
    rc = {
        "svg.hashsalt": SVG_HASHSALT,
        "svg.fonttype": "path",
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": foreground,
        "axes.labelcolor": foreground,
        "axes.edgecolor": foreground,
        "axes.titlecolor": foreground,
        "xtick.color": foreground,
        "ytick.color": foreground,
        "grid.color": grid,
        "legend.framealpha": 0.0,
        "font.size": 11,
    }
    with plt.rc_context(rc):
        yield


def _format_value(value: float) -> str:
    """Format a numeric parameter value without a trailing ``.0``."""
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _trace_label(key: str, value: float | str) -> str:
    """Build a legend label such as ``stiffness=100 N·m/rad`` (or the raw string label)."""
    if isinstance(value, str):
        return value
    if key == "delay":
        steps = int(value)
        return f"delay={steps} steps ({round(steps * DT * 1000.0)} ms)"
    unit = VALUE_UNITS.get(key, "")
    return f"{key}={_format_value(value)} {unit}".rstrip()


def _save_curve(fig, key: str, dark: bool) -> Path:
    """Write *fig* as a light or dark SVG with reproducible (timestamp-free) metadata.

    Trailing whitespace is stripped from every line (and a final newline is
    ensured) so the output is byte-stable under the repository's pre-commit
    hooks: matplotlib emits trailing spaces that the hooks would otherwise trim,
    making committed files differ from freshly generated ones.
    """
    path = MEDIA_DIR / f"{key}-curve-{'dark' if dark else 'light'}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    lines = [line.rstrip() for line in path.read_text().splitlines()]
    path.write_text("\n".join(lines) + "\n")
    return path


def _plot_velocity_limit(row: RowSpec, key: str) -> list[Path]:
    """Plot the analytic DC-motor torque-speed envelope for each velocity limit.

    The envelope follows the linear four-quadrant model
    ``clip(saturation_effort * (1 - q_dot / velocity_limit), -effort_limit, effort_limit)``,
    matching :class:`isaaclab.actuators.DCMotor`.
    """
    cfg = row.make_actuator_cfg(row.values[0])
    stall = float(cfg.saturation_effort)
    continuous = float(cfg.effort_limit)
    max_limit = max(float(value) for value in row.values if value is not None)
    q_max = max_limit * (1.0 + continuous / stall) * 1.1
    speed = np.linspace(-q_max, q_max, 512)

    paths = []
    for dark in (False, True):
        with _mpl_style(dark):
            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            for column, value in enumerate(row.values):
                if value is None:
                    continue
                envelope = np.clip(stall * (1.0 - speed / float(value)), -continuous, continuous)
                ax.plot(speed, envelope, color=TRACE_COLORS[column % len(TRACE_COLORS)], label=_trace_label(key, value))
            ax.axhline(continuous, color="0.5", ls="--", lw=0.8)
            ax.axhline(-continuous, color="0.5", ls="--", lw=0.8)
            ax.axhline(0.0, color="0.6", lw=0.6)
            ax.axvline(0.0, color="0.6", lw=0.6)
            ax.set_xlabel("Joint velocity [rad/s]")
            ax.set_ylabel("Joint torque [N·m]")
            ax.set_title("DC-motor torque-speed envelope")
            ax.grid(True, alpha=0.4)
            ax.legend(loc="upper right", fontsize=9)
            paths.append(_save_curve(fig, key, dark))
    return paths


def plot_row(row: RowSpec, logs: dict[str, torch.Tensor] | None, key: str) -> list[Path]:
    """Plot a row's comparison curves as a light and a dark SVG.

    Args:
        row: Comparison row being plotted.
        logs: Per-step telemetry from :func:`run_row`, or ``None`` for the
            analytic ``velocity-limit`` row.
        key: Comparison-row key used for the output filenames.

    Returns:
        The written SVG paths, light variant first.
    """
    if key == "velocity-limit":
        return _plot_velocity_limit(row, key)

    assert logs is not None, f"logs are required to plot {key!r}"
    quantity = row.plot.quantity
    live = [(column, value) for column, value in enumerate(row.values) if value is not None]
    times = np.arange(logs["joint_pos"].shape[0]) * DT

    paths = []
    for dark in (False, True):
        with _mpl_style(dark):
            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            data = logs[quantity].numpy()
            for column, value in live:
                color = TRACE_COLORS[column % len(TRACE_COLORS)]
                ax.plot(times, data[:, column], color=color, label=_trace_label(key, value))
            if key == "delay":
                reference = np.array([row.command(t)[0] for t in times])
                ax.plot(times, reference, color="0.5", ls="--", lw=1.0, label="command")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(row.plot.ylabel)
            ax.set_title(row.plot.title)
            ax.grid(True, alpha=0.4)
            ax.legend(loc="best", fontsize=9)
            paths.append(_save_curve(fig, key, dark))
    return paths


def record_row(sim: sim_utils.SimulationContext, row: RowSpec, key: str) -> list[Path]:
    """Capture one row's clip (when the row has one) and plot its curves.

    Builds the scene, runs the row while capturing every :data:`CAPTURE_STRIDE`
    physics frames from a fixed camera, writes the animated clip, then plots the
    comparison curves from the logged telemetry. Rows with
    :attr:`RowSpec.record_clip` disabled skip the camera and clip entirely and
    only produce curves.

    Args:
        sim: Active simulation context (fresh stage for this row).
        row: Comparison row to record.
        key: Comparison-row key used for the output filenames.

    Returns:
        Every written media path (clip first when captured, then the light and
        dark curves).
    """
    articulations = build_scene(sim, row)
    if row.record_clip:
        stage = sim_utils.get_current_stage()
        for index in range(len(_live_values(row))):
            _author_row_markers(stage, f"/World/Pendulum_{index}")
    sim.reset()
    recorder = _get_record_recorder() if row.record_clip else None

    frames: list[Image.Image] = []

    def on_frame(step: int) -> None:
        if recorder is not None and step % CAPTURE_STRIDE == 0:
            frames.append(Image.fromarray(recorder.render_rgb_array()))

    logs = run_row(sim, articulations, row, on_frame=on_frame)
    paths = []
    if frames:
        paths.append(_write_clip(_annotate_frames(frames, row, key), key))
    paths += plot_row(row, logs, key)
    return paths


def _print_size_table(written: dict[str, list[Path]]) -> None:
    """Print a ``file -> KB`` table plus the directory total."""
    print("\nGenerated media:")
    total = 0
    for paths in written.values():
        for path in paths:
            size = path.stat().st_size
            total += size
            print(f"  {path.name:<28s} {size / 1000:8.1f} KB")
    print(f"  {'TOTAL':<28s} {total / 1000:8.1f} KB")


def _run_record() -> None:
    """Generate clips and curves for the selected comparison rows."""
    if args_cli.all:
        keys = list(ROWS)
    elif args_cli.parameter in ROWS:
        keys = [args_cli.parameter]
    else:
        raise SystemExit(f"Unknown parameter {args_cli.parameter!r}. Use --list_parameters to see the available keys.")

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {}

    # ``velocity-limit`` is an analytic curve with no clip, so it needs no simulation.
    sim_keys = [key for key in keys if key != "velocity-limit"]
    analytic_keys = [key for key in keys if key == "velocity-limit"]

    if sim_keys:
        with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
            _use_implicit_integrator(physics_cfg)
            for index, key in enumerate(sim_keys):
                # Rebuild a fresh stage per row so scenes do not accumulate.
                if index > 0:
                    sim_utils.SimulationContext.clear_instance()
                torch.manual_seed(SEED)
                sim_cfg = sim_utils.SimulationCfg(dt=DT, device=args_cli.device, physics=physics_cfg)
                sim = sim_utils.SimulationContext(sim_cfg)
                sim.set_camera_view(eye=list(CAMERA_EYE), target=list(CAMERA_TARGET))
                row = ROWS[key]
                print(f"[INFO]: Recording '{key}': {row.name}")
                written[key] = record_row(sim, row, key)

    for key in analytic_keys:
        row = ROWS[key]
        print(f"[INFO]: Plotting analytic curves for '{key}': {row.name}")
        written[key] = plot_row(row, None, key)

    _print_size_table(written)


def main() -> None:
    """Author the requested comparison scene and run it."""
    if args_cli.list_parameters:
        _print_parameters()
        return

    if args_cli.record:
        _run_record()
        return

    if args_cli.parameter not in ROWS:
        raise SystemExit(f"Unknown parameter {args_cli.parameter!r}. Use --list_parameters to see the available keys.")
    row = ROWS[args_cli.parameter]

    torch.manual_seed(SEED)

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        _use_implicit_integrator(physics_cfg)
        sim_cfg = sim_utils.SimulationCfg(dt=DT, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[0.0, -6.0, 2.0], target=[0.0, 0.0, 1.6])

        articulations = build_scene(sim, row)
        sim.reset()
        print(f"[INFO]: Comparing actuator parameter '{args_cli.parameter}': {row.name}")

        if args_cli.smoke:
            run_row(sim, articulations, row, num_steps=SMOKE_STEPS)
            print("[INFO]: Smoke run complete.")
            return

        _run_interactive(sim, articulations, row)


if __name__ == "__main__":
    main()
