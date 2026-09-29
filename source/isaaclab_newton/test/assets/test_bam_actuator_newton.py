# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end checks of Newton-native BAM on live MJWarp articulations.

The temporary pendulum assets have closed-form gravity loads. Their resting states are
checked against independently calculated stiction-band bounds for the vendored XL330 fit.
No Isaac Sim runtime or downloaded asset is needed.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    MjWarpActuatorBridge,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonManager,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators import BamActuatorCfg
from isaaclab.actuators.newton import ControllerBam, read_group_parameter, write_group_parameter
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import test_devices

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

PENDULUM_USDA = """\
#usda 1.0
(
    defaultPrim = "Robot"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Pivot" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.1
        float3 physics:diagonalInertia = (0.001, 0.001, 0.001)
    }

    def Xform "Arm" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.02
        point3f physics:centerOfMass = (0.05, 0, 0)
        float3 physics:diagonalInertia = (0.002, 0.002, 0.002)
    }

    def PhysicsFixedJoint "anchor"
    {
        rel physics:body1 = </Robot/Pivot>
    }

    def PhysicsRevoluteJoint "joint"
    {
        uniform token physics:axis = "Y"
        rel physics:body0 = </Robot/Pivot>
        rel physics:body1 = </Robot/Arm>
    }
}
"""
"""Fixed-base single-degree-of-freedom pendulum.

``Pivot`` is welded to the world by ``anchor``, a fixed joint whose ``physics:body0`` is left
unset. That weld is what makes the USD physics parser report an articulation at all: a lone
body hanging off a world-anchored revolute joint is imported as an orphan joint and no
articulation is created, so :class:`~isaaclab.assets.Articulation` finds nothing to bind to.
``Arm`` is the only degree of freedom.

``Arm``'s centre of mass is offset along +X and the joint spins about +Y, so rotating the
joint by ``theta`` swings the centre of mass down and gravity applies a pure
``m * g * L * cos(theta)`` load about the joint axis.

``Arm``'s inertia is authored independently of its mass distribution and is deliberately much
larger than ``m * L^2``: it stands in for the rotor inertia a geared servo reflects to its
output shaft (0.0018 kg m^2 for the Dynamixel XL330 the BAM parameters were fitted on).
Without it the joint inertia would sit far below the actuator's electrical damping times the
timestep, and a back-EMF torque applied explicitly by the actuator could not be integrated
stably.
"""

DT = 1.0 / 120.0
"""Physics timestep [s]."""

NUM_ENVS = 2
"""Environments simulated side by side, so a per-environment randomization is observable."""

NUM_STEPS = 200
"""Steps each settling phase runs for [-]. The joint is at rest well before this."""

VIN = 7.4
"""Supply voltage the actuator is configured with [V]."""

KP_FW = 200.0
"""Firmware proportional gain the actuator is configured with [-]."""

INITIAL_ANGLE = 0.3
"""Angle the arm is released from [rad]. The commanded target is always 0."""


def _make_sim_cfg(device: str, use_newton_actuators: bool = False, use_cuda_graph: bool = True) -> SimulationCfg:
    """Build the MJWarp configuration used by every test in this module."""
    return SimulationCfg(
        dt=DT,
        device=device,
        use_newton_actuators=use_newton_actuators,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=20, nconmax=20, ls_iterations=20, integrator="implicitfast", impratio=1),
            num_substeps=2,
            debug_mode=False,
            use_cuda_graph=use_cuda_graph,
        ),
    )


@pytest.fixture(scope="module")
def pendulum_usd(tmp_path_factory) -> str:
    """Write :data:`PENDULUM_USDA` to a temporary file and return its path."""
    path = tmp_path_factory.mktemp("bam_pendulum") / "single_joint_pendulum.usda"
    path.write_text(PENDULUM_USDA)
    return str(path)


@pytest.fixture
def sim(device):
    """Newton simulation context running the Isaac Lab-executed actuator path."""
    with build_simulation_context(
        device=device,
        gravity_enabled=True,
        add_ground_plane=False,
        sim_cfg=_make_sim_cfg(device),
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        yield sim_ctx


@pytest.fixture
def native_sim(device):
    """Newton simulation context running the Newton-native actuator path."""
    with build_simulation_context(
        device=device,
        gravity_enabled=True,
        add_ground_plane=False,
        sim_cfg=_make_sim_cfg(device, use_newton_actuators=True),
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        yield sim_ctx


@pytest.fixture
def native_sim_eager(device):
    """Newton-native actuator path with CUDA graph capture off.

    A replayed graph runs its recorded kernels without re-entering Python, so a test that
    observes the hooks from Python has to step eagerly to see every iteration on both devices.
    """
    with build_simulation_context(
        device=device,
        gravity_enabled=True,
        add_ground_plane=False,
        sim_cfg=_make_sim_cfg(device, use_newton_actuators=True, use_cuda_graph=False),
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        yield sim_ctx


def _gravity_load(robot: Articulation, sim) -> float:
    """Return the peak gravity torque ``m * g * L`` of the arm [N.m], read from the sim.

    Deriving the load from the live articulation rather than from the authored numbers keeps
    the reference calculation honest about what the importer actually built (a stage-unit
    misreading, for instance, would show up here rather than silently shifting the
    prediction).
    """
    arm = robot.body_names.index("Arm")
    mass = float(robot.data.body_mass.torch[0, arm])
    lever = float(robot.data.body_com_pos_b.torch[0, arm, 0])
    return mass * abs(sim.cfg.gravity[2]) * lever


STICTION_BANDS = {
    0.5: (0.007463133433448432, 0.031064942414470192),
    1.0: (-0.0010575354059803614, 0.04929333851780336),
    2.0: (-0.04128957832843019, 0.11684233352917653),
}
"""Static angle bounds [rad] for the pendulum at each friction multiplier.

Independently solved from ``|motor + gravity| = scale * friction`` at zero speed,
using the vendored XL330 m6 fit, 200 firmware gain, 7.4 V, and a 0.00981 N.m peak
load. Motor torque is linear throughout these intervals (neither PWM nor current
saturates). Scalar bisection of each boundary gives the recorded values; the
reference is deliberately fixed rather than recomputed with the controller.
"""


def _stiction_band(load: float, friction_scale: float = 1.0) -> tuple[float, float]:
    """Return the recorded band after checking this is the reference pendulum load."""
    assert load == pytest.approx(0.00981, rel=1e-6)
    return STICTION_BANDS[friction_scale]


"""
Rollout helpers.
"""


def _release(robot: Articulation) -> None:
    """Put every arm back at :data:`INITIAL_ANGLE` at rest and clear the actuator state."""
    robot.write_joint_position_to_sim_index(position=torch.full_like(robot.data.joint_pos.torch, INITIAL_ANGLE))
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.joint_vel.torch))
    robot.actuators.reset()


def _settle(robot: Articulation, sim) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Command a zero position target for :data:`NUM_STEPS` steps and record the rollout.

    Returns:
        Joint positions [rad], joint velocities [rad/s] and applied efforts [N.m], each of
        shape ``(NUM_STEPS, NUM_ENVS, 1)``.
    """
    robot.actuators.target_command.set_position_index(
        value=torch.zeros(NUM_ENVS, robot.num_joints, device=robot.device)
    )
    positions, velocities, efforts = [], [], []
    for _ in range(NUM_STEPS):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        positions.append(robot.data.joint_pos.torch.clone())
        velocities.append(robot.data.joint_vel.torch.clone())
        efforts.append(robot.actuators.applied_effort.torch.clone())
    return torch.stack(positions), torch.stack(velocities), torch.stack(efforts)


def _assert_rest(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    efforts: torch.Tensor,
    velocity_tolerance: float = 1e-3,
) -> torch.Tensor:
    """Assert the rollout is finite and has come to rest, and return the final angles.

    Args:
        positions: Recorded joint positions [rad].
        velocities: Recorded joint velocities [rad/s].
        efforts: Recorded applied efforts [N.m].
        velocity_tolerance: Largest final speed that still counts as at rest [rad/s].

    Returns:
        The final joint angle of each environment [rad].
    """
    for name, trace in (("position", positions), ("velocity", velocities), ("effort", efforts)):
        assert torch.isfinite(trace).all(), f"non-finite joint {name} in the rollout"
    assert velocities[-1].abs().max() < velocity_tolerance, "the pendulum has not come to rest"
    return positions[-1].reshape(NUM_ENVS)


NATIVE_REST_TOLERANCE = 5e-3
"""Largest final speed the Newton-native path counts as at rest [rad/s].

MuJoCo's friction-loss constraint is compliant, not a hard stop: even with the stiffened
solver reference the reference implementation uses, a held joint keeps creeping at order
1e-3 rad/s.
That is three orders of magnitude below the 0.6 rad/s the arm is released with, and the
residual drift is bounded separately by :func:`_assert_creep_is_bounded`. Tightening this
threshold would not measure a better actuator, only a stiffer constraint.
"""

NATIVE_MAX_CREEP = math.radians(0.2)
"""Largest angle the native path may drift over the last :data:`NATIVE_CREEP_WINDOW` steps [rad]."""

NATIVE_CREEP_WINDOW = 50
"""Trailing window the creep bound is measured over [physics steps]."""

GRAPH_DECIMATION = 2
"""Decimation the graph-capture test runs at [physics steps per environment step].

Even by necessity, not by taste: see the test's docstring.
"""


NATIVE_FRICTION_SEPARATION = math.radians(2.0)
"""Smallest hanging-error gap the 0.5 / 2.0 friction scales must open up [rad].

The scales are 4x apart, which measures as roughly 4 degrees on this fixture. Requiring a
real separation, not just an ordering, is what makes the assertion fail if the budget never
reaches the solver: two environments running the same published friction settle together and
their order is then decided by rounding.
"""

FRICTION_SCALES = (0.5, 2.0)
"""Per-environment friction-budget scales the randomization tests write [-]."""


def _assert_creep_is_bounded(positions: torch.Tensor) -> None:
    """Assert the settled joint is not sliding away under its own friction constraint."""
    drift = (positions[-1] - positions[-1 - NATIVE_CREEP_WINDOW]).abs().max()
    assert float(drift) < NATIVE_MAX_CREEP, f"the held joint drifted {float(drift):.2e} rad while nominally at rest"


def _settle_with_friction_scales(robot: Articulation, sim, load: float) -> torch.Tensor:
    """Write :data:`FRICTION_SCALES` per environment, settle, and assert the effect.

    The write goes through the group-parameter API -- the path an environment's
    domain-randomization event uses -- and lands in the controller array the in-graph friction
    publish reads, so this exercises the whole chain from the event down to MuJoCo's
    constraint.

    Returns:
        The settled joint angle of each environment [rad].
    """
    expected = torch.tensor([[FRICTION_SCALES[0]], [FRICTION_SCALES[1]]], device=robot.device)
    write_group_parameter(robot.actuators, "servo", "controller", "friction_scale", expected)
    torch.testing.assert_close(read_group_parameter(robot.actuators, "servo", "controller", "friction_scale"), expected)

    _release(robot)
    rollout = _settle(robot, sim)
    settled = _assert_rest(*rollout, velocity_tolerance=NATIVE_REST_TOLERANCE)
    _assert_creep_is_bounded(rollout[0])

    # More friction, more hanging error: the arm is released above the target and stops
    # earlier the wider its stiction band is.
    separation = float(settled[1]) - float(settled[0])
    assert separation > NATIVE_FRICTION_SEPARATION, f"the friction scales barely separated ({separation:.2e} rad)"
    for env, scale in enumerate(FRICTION_SCALES):
        band_low, band_high = _stiction_band(load, friction_scale=scale)
        assert band_low <= float(settled[env]) <= band_high
    return settled


def _build_native_pendulum(sim, pendulum_usd: str, actuator_cfg: BamActuatorCfg | None = None) -> Articulation:
    """Spawn :data:`NUM_ENVS` BAM-driven pendulums on the Newton-native actuator path.

    Args:
        sim: Simulation context to spawn into.
        pendulum_usd: Path of the pendulum asset.
        actuator_cfg: Servo group to drive them with. Defaults to the plain BAM configuration.
    """
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Env_[^/]*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
            actuators={"servo": actuator_cfg or BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)},
        )
    )
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg], NUM_ENVS, 1.0)
    replicate(sim.get_clone_plan())
    sim.reset()
    assert robot.is_initialized
    assert "servo" in robot.actuators._native_group_names, "the BAM group must run on the Newton path"
    return robot


def _native_controller(robot: Articulation) -> ControllerBam:
    """Return the BAM controller the articulation's native group is executed by."""
    controllers = [
        actuator.controller
        for actuator in NewtonManager._adapter.actuators
        if isinstance(actuator.controller, ControllerBam)
    ]
    assert len(controllers) == 1, "the fixture has exactly one BAM actuator"
    return controllers[0]


def _assert_recorded_trajectory(robot: Articulation, sim, controller: ControllerBam) -> None:
    """Replay the pre-cleanup native trajectory, including command reversals.

    The fixture was recorded from isolated revision 3691a0bf04 on CUDA with the same
    pendulum and deterministic configuration. It stores positions [rad], velocities
    [rad/s], motor efforts [N.m], friction budgets [N.m], targets [rad], timestep [s]
    and dependency versions. This guards native behavior during refactoring; the
    static-band checks independently constrain the physical resting state.
    """
    _release(robot)
    with np.load(Path(__file__).parent / "data" / "bam_pendulum_trajectory.npz") as golden:
        assert sim.get_physics_dt() == float(golden["dt"])
        traces = {name: [] for name in ("position", "velocity", "effort", "friction_budget")}
        for target in golden["target"]:
            robot.actuators.target_command.set_position_index(
                value=torch.full_like(robot.data.joint_pos.torch, float(target))
            )
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            traces["position"].append(robot.data.joint_pos.torch.cpu().numpy().copy())
            traces["velocity"].append(robot.data.joint_vel.torch.cpu().numpy().copy())
            traces["effort"].append(robot.actuators.applied_effort.torch.cpu().numpy().copy())
            traces["friction_budget"].append(controller.friction_budget.numpy().copy())
        for name, values in traces.items():
            # CPU and CUDA solver reductions differ slightly; preserve a tight physical tolerance.
            np.testing.assert_allclose(np.stack(values), golden[name], atol=2e-5, rtol=2e-4, err_msg=name)


@pytest.mark.parametrize("device", test_devices())
def test_native_pendulum_settles_inside_the_stiction_band(native_sim, device, pendulum_usd):
    """Check a settled native pendulum against the recorded static-friction bounds."""
    robot = _build_native_pendulum(native_sim, pendulum_usd)
    load = _gravity_load(robot, native_sim)
    controller = _native_controller(robot)
    assert controller.solver_applies_friction, "the MuJoCo solver must own the friction budget"
    # Authoring seeds a positive joint friction on the driven joints. MuJoCo only assembles a
    # friction-loss constraint row where the frictionloss is positive, and it sizes its
    # constraint budget from the model as spawned, so the row has to exist before the first
    # solve; the per-step budget then overwrites the value.
    assert (NewtonManager.backend.model.joint_friction.numpy() > 0.0).all()

    _assert_recorded_trajectory(robot, native_sim, controller)
    _release(robot)
    rollout = _settle(robot, native_sim)
    final_angle = _assert_rest(*rollout, velocity_tolerance=NATIVE_REST_TOLERANCE)
    _assert_creep_is_bounded(rollout[0])

    band_low, band_high = _stiction_band(load)
    for env in range(NUM_ENVS):
        assert band_low <= float(final_angle[env]) <= band_high

    # The solver-side contract: the actuator applies the motor torque and nothing else; the
    # load is cancelled by the friction-loss constraint, which is invisible to this telemetry.
    holding_effort = robot.actuators.applied_effort.torch.reshape(NUM_ENVS)
    motor_torque = torch.as_tensor(controller.motor_torque.numpy(), device=holding_effort.device)
    torch.testing.assert_close(holding_effort, motor_torque.reshape(NUM_ENVS), atol=1e-6, rtol=0.0)
    # ... and it is a real torque, not a dead actuator sitting at zero.
    assert holding_effort.abs().min() > 1e-3

    # The published budget is what MuJoCo is clipping with.
    frictionloss = NewtonManager._solver.mjw_model.dof_frictionloss.numpy().reshape(-1)
    np.testing.assert_allclose(frictionloss, controller.friction_budget.numpy(), atol=1e-7, rtol=0.0)

    # The load the gearbox works against is read from the solver rather than estimated, and at
    # rest that read is exact: it is the gravity torque to the last digit. It also proves the
    # actuator's own friction rows are stripped out of the constraint force -- leaving them in
    # would report roughly twice this value here and feed the friction back on itself.
    external_torque = torch.as_tensor(controller.external_torque.numpy(), device=final_angle.device)
    expected_load = (load * torch.cos(final_angle)).to(external_torque.dtype)
    torch.testing.assert_close(external_torque.reshape(NUM_ENVS), expected_load, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_native_friction_randomization_changes_the_hanging_error(native_sim, device, pendulum_usd):
    """Randomize ``friction_scale`` through the API used by environment events."""
    robot = _build_native_pendulum(native_sim, pendulum_usd)

    for attr in ("vin", "sag_gain", "friction_scale", "kp_scale", "kd_scale"):
        values = read_group_parameter(robot.actuators, "servo", "controller", attr)
        assert values.shape == (NUM_ENVS, robot.num_joints)

    _settle_with_friction_scales(robot, native_sim, _gravity_load(robot, native_sim))


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("decimation", [1, GRAPH_DECIMATION])
def test_native_bam_actuators_are_captured_in_the_cuda_graph(native_sim, device, pendulum_usd, decimation):
    """Captured BAM updates stay live with both single-step and even decimation."""
    robot = _build_native_pendulum(native_sim, pendulum_usd)
    load = _gravity_load(robot, native_sim)
    assert NewtonManager._adapter.is_all_graphable
    assert NewtonManager._is_all_graphable()
    assert NewtonManager._pre_actuator_callbacks, "the external-torque gather must be registered"

    NewtonManager.set_decimation(decimation)
    native_sim.step()
    if device.startswith("cuda"):
        assert NewtonManager._graph is not None, "the decimation loop was not captured"

    # A graph that baked a stale friction budget would settle both environments together, so
    # replaying it under a per-environment randomization is the real capture evidence: the
    # in-graph publish has to read the controller array a host-side write just changed.
    _settle_with_friction_scales(robot, native_sim, load)


@pytest.mark.parametrize("device", test_devices())
def test_the_friction_budget_is_refreshed_on_every_physics_step(native_sim_eager, device, pendulum_usd):
    """Load gather, actuator step and friction publish must run once per *physics* step.

    The BAM friction budget is sized from the previous solve's generalized load, so a budget
    computed once per *control* step would face a load up to ``decimation`` solves old. The
    one-step lag is deliberate -- it is what the reference implementation carries -- but a
    ``decimation``-step lag is not, and nothing else in this suite distinguishes the two: both
    leave the same value in ``dof_frictionloss`` when the step returns.

    The order within an iteration matters just as much and is asserted with the cadence: the
    gather has to read the previous solve before the actuators run, and the publish has to land
    after them and before the substeps consume the row.
    """
    robot = _build_native_pendulum(native_sim_eager, pendulum_usd)
    events: list[str] = []
    gather, publish = MjWarpActuatorBridge.gather_external_torque, MjWarpActuatorBridge.publish_dof_friction
    step = NewtonManager._adapter.step

    def spy(name, wrapped):
        def wrapper(*args, **kwargs):
            events.append(name)
            return wrapped(*args, **kwargs)

        return wrapper

    MjWarpActuatorBridge.gather_external_torque = spy("gather", gather)
    MjWarpActuatorBridge.publish_dof_friction = spy("publish", publish)
    NewtonManager._adapter.step = spy("actuators", step)
    try:
        NewtonManager.set_decimation(GRAPH_DECIMATION)
        assert NewtonManager._graph is None, "the eager fixture must not capture a graph"
        events.clear()
        _release(robot)
        native_sim_eager.step()
    finally:
        MjWarpActuatorBridge.gather_external_torque = gather
        MjWarpActuatorBridge.publish_dof_friction = publish
        NewtonManager._adapter.step = step

    assert events == ["gather", "actuators", "publish"] * GRAPH_DECIMATION


def _build_two_native_pendulums(sim, pendulum_usd: str, second_cfg: BamActuatorCfg) -> tuple:
    """Spawn two BAM articulations per environment and initialize the simulation.

    Both robots are the same asset, so Newton merges their BAM joints into *one* actuator
    whenever the configurations agree on every grouping-key field -- which is exactly the
    multi-robot case the per-articulation binding has to get right.
    """
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robots = []
    for name, cfg in (
        ("RobotA", BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)),
        ("RobotB", second_cfg),
    ):
        robots.append(
            Articulation(
                ArticulationCfg(
                    prim_path=f"/World/Env_[^/]*/{name}",
                    spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
                    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
                    actuators={"servo": cfg},
                )
            )
        )
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg for robot in robots], NUM_ENVS, 1.0)
    replicate(sim.get_clone_plan())
    sim.reset()
    return tuple(robots)


@pytest.mark.parametrize("device", test_devices())
def test_two_articulations_sharing_one_actuator_must_agree(native_sim, device, pendulum_usd):
    """Two robots merged into one Newton actuator cannot carry different BAM settings.

    ``vin_range``, ``vin_drop_gain_range``, ``friction_scale_range`` and ``stiff_frictionloss``
    are not part of Newton's actuator-grouping key, so structurally identical robots share one
    actuator and one set of parameter arrays. Applying the second articulation's ranges would
    silently discard the first's randomization, and skipping them would silently ignore the
    second's configuration, so the conflict has to be refused.
    """
    conflicting = BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW, friction_scale_range=(0.5, 2.0))
    with pytest.raises(ValueError, match="share one Newton actuator"):
        _build_two_native_pendulums(native_sim, pendulum_usd, conflicting)


@pytest.mark.parametrize("device", test_devices())
def test_two_articulations_with_matching_settings_bind_once(native_sim, device, pendulum_usd):
    """Agreeing robots share the actuator, and neither is left unbound or bound twice."""
    matching = BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)
    robot_a, robot_b = _build_two_native_pendulums(native_sim, pendulum_usd, matching)

    bam_actuators = [
        actuator for actuator in NewtonManager._adapter.actuators if isinstance(actuator.controller, ControllerBam)
    ]
    assert len(bam_actuators) == 1, "the identical robots must merge into one Newton actuator"
    controller = bam_actuators[0].controller
    assert controller.solver_applies_friction
    assert controller.external_torque is not None

    # One binding, not one per articulation: a second registration would write the same budget
    # twice per step and, worse, hide a scoping mistake. The gather hook is the countable half
    # of the pair -- the publish hook is registered as a lambda, so it cannot be told apart from
    # any other post-actuator callback -- and both are registered together or not at all.
    assert len(NewtonManager._pre_actuator_callbacks) == 1

    for robot in (robot_a, robot_b):
        assert "servo" in robot.actuators._native_group_names


@pytest.mark.parametrize("device", test_devices())
def test_startup_ranges_are_sampled_per_environment(native_sim, device, pendulum_usd):
    """The config's start-up ranges must be drawn even though no solver exists yet.

    Sampling happens while the model is being built, before
    :meth:`~isaaclab_newton.physics.NewtonManager.initialize_solver` runs, precisely so that it
    does not depend on which solver the scene uses -- the values feed the controller's kernels,
    not the solver. Implementation A samples them on every backend and so must this one.
    """
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Env_[^/]*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
            actuators={
                "servo": BamActuatorCfg(
                    joint_names_expr=[".*"],
                    kp_fw=KP_FW,
                    vin_range=(6.0, 8.0),
                    friction_scale_range=(0.5, 1.5),
                )
            },
        )
    )
    clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg], NUM_ENVS, 1.0)
    replicate(native_sim.get_clone_plan())
    native_sim.reset()
    assert robot.is_initialized

    for attr, (low, high) in (("vin", (6.0, 8.0)), ("friction_scale", (0.5, 1.5))):
        values = read_group_parameter(robot.actuators, "servo", "controller", attr)
        assert bool(((values >= low) & (values <= high)).all()), f"{attr} outside its configured range"
        assert len(torch.unique(values)) > 1, f"{attr} drew the same value for every environment"
    # An unset range keeps the authored nominal.
    torch.testing.assert_close(
        read_group_parameter(robot.actuators, "servo", "controller", "sag_gain"),
        torch.zeros(NUM_ENVS, robot.num_joints, device=robot.device),
    )


@pytest.mark.parametrize("device", test_devices())
def test_each_articulation_configures_only_its_own_actuator(native_sim, device, pendulum_usd):
    """Two robots that do *not* merge must each get their own configuration.

    The actuator adapter is simulation-global, so an articulation that walked the whole
    adapter would apply its own start-up ranges to another robot's actuator -- silently, and
    with whichever articulation initialized first winning. Differing ``max_delay`` puts the two
    robots in different Newton actuators; only the scoping decides which one each configures.
    """
    delayed = BamActuatorCfg(
        joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW, max_delay=2, friction_scale_range=(3.0, 3.0)
    )
    robot_a, robot_b = _build_two_native_pendulums(native_sim, pendulum_usd, delayed)
    assert len({id(actuator) for actuator in NewtonManager._adapter.actuators}) == 2, (
        "the two robots must not merge, or the test cannot tell the configurations apart"
    )

    # robot_a keeps ``_build_two_native_pendulums``'s default cfg (no start-up range, so 1.0).
    for robot, expected in ((robot_a, 1.0), (robot_b, 3.0)):
        friction_scale = read_group_parameter(robot.actuators, "servo", "controller", "friction_scale")
        torch.testing.assert_close(friction_scale, torch.full_like(friction_scale, expected), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_startup_ranges_are_sampled_without_a_mujoco_solver(device, pendulum_usd):
    """Start-up randomization must not depend on which solver the scene runs.

    The ranges feed the native controller's kernels. On a solver that cannot apply joint
    dry friction the BAM controller falls
    back to its own stiction clip -- but the randomization is unaffected, which is only true
    because the sampling happens while the model is built, before any solver exists.
    """
    sim_cfg = SimulationCfg(
        dt=DT,
        device=device,
        use_newton_actuators=True,
        physics=NewtonCfg(solver_cfg=FeatherstoneSolverCfg(), num_substeps=2, debug_mode=False),
    )
    with build_simulation_context(
        device=device, gravity_enabled=True, add_ground_plane=False, sim_cfg=sim_cfg
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        for index in range(NUM_ENVS):
            sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/Env_[^/]*/Robot",
                spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
                actuators={"servo": BamActuatorCfg(joint_names_expr=[".*"], kp_fw=KP_FW, vin_range=(6.0, 8.0))},
            )
        )
        clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [robot.cfg], NUM_ENVS, 1.0)
        replicate(sim_ctx.get_clone_plan())
        sim_ctx.reset()
        assert robot.is_initialized

        controller = _native_controller(robot)
        # No MuJoCo model, so the controller keeps the torque-level clip ...
        assert not controller.solver_applies_friction
        assert controller.external_torque is None
        # ... and the start-up randomization happened anyway.
        vin = read_group_parameter(robot.actuators, "servo", "controller", "vin")
        assert bool(((vin >= 6.0) & (vin <= 8.0)).all())
        assert len(torch.unique(vin)) > 1


@pytest.mark.parametrize("device", test_devices())
def test_bam_cfg_is_refused_on_the_isaac_lab_executed_path(sim, device, pendulum_usd):
    """BAM requires the native actuator loop."""
    with pytest.raises(ValueError, match="use_newton_actuators"):
        _build_native_pendulum(sim, pendulum_usd, BamActuatorCfg(joint_names_expr=[".*"], kp_fw=KP_FW))
