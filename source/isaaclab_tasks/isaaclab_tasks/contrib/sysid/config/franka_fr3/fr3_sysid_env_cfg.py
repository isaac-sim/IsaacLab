# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CMA-ES stiffness/damping sysid env for the Franka FR3 (fixed base, 7 joints).

Ported from the agile G1 sysid stack: same replay + CMA-ES loop, but the fitted
parameters are the solver-side PD gains of the implicit actuator ``{stiffness,
damping}`` instead of armature/friction/delay. Used exclusively by
``scripts/sysid/fit.py``.

The robot is spawned from a USD generated out of the mesh-stripped FR3 URDF
(``assets/fr3_nomesh.urdf`` — inertials and joint limits are inline, meshes are
irrelevant for free-air joint sysid). Generate it once with
``scripts/sysid/prepare_fr3_asset.py``.
"""

from __future__ import annotations

import os

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.sysid.sysid_env_cfg import (
    SysidActionsCfg,
    SysIdCfg,
    SysidEventCfg,
    SysidObservationsCfg,
    SysidRewardsCfg,
    SysidTerminationsCfg,
)
from isaaclab_tasks.utils import PresetCfg

##
# Robot definition
##

# convert_urdf.py treats the output as a directory: the root layer lands at
# <out>/<name>/<name>.usda with the multi-physics payloads beside it.
FR3_USD_PATH = os.path.join(os.path.dirname(__file__), "assets", "fr3.usd", "fr3_nomesh", "fr3_nomesh.usda")

# Column order of the dataset produced by isaac_ros_sysid (franka_fr3.yaml).
FR3_SYSID_JOINT_ORDER: list[str] = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

# Franka "ready" pose [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4] — same home pose the
# data collection chirps around (config/robots/franka_fr3.yaml).
FR3_READY_POSE: dict[str, float] = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -0.7853981633974483,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.356194490192345,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.5707963267948966,
    "fr3_joint7": 0.7853981633974483,
}

FR3_SYSID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FR3_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        # The USD is converted with --fix-base already; re-asserting
        # fix_root_link here would try to create a fixed joint on the URDF's
        # massless dummy root link and hit a NotImplementedError.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(FR3_READY_POSE)),
    actuators={
        # Single implicit group over the whole arm. stiffness/damping default to
        # the real controller's host-PD gains (dex/robot-control
        # franka_pd_params.yaml); the optimizer overwrites them per env every
        # generation. Effort/velocity limits from fr3.urdf.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint.*"],
            effort_limit_sim={"fr3_joint[1-4]": 87.0, "fr3_joint[5-7]": 12.0},
            velocity_limit_sim={
                "fr3_joint[1-4]": 2.62,
                "fr3_joint5": 5.26,
                "fr3_joint6": 4.18,
                "fr3_joint7": 5.26,
            },
            stiffness={"fr3_joint[1-4]": 600.0, "fr3_joint5": 250.0, "fr3_joint6": 150.0, "fr3_joint7": 50.0},
            damping={"fr3_joint[1-4]": 30.0, "fr3_joint5": 10.0, "fr3_joint6": 10.0, "fr3_joint7": 5.0},
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

##
# CMA-ES parameter bounds: [stiffness, damping] per joint, layout
# [stiffness x N, damping x N] — see scripts/sysid/cma_es.py.
##

_WRIST_JOINTS = {"fr3_joint5", "fr3_joint6", "fr3_joint7"}
# [low, high] per parameter. Deliberately stack-agnostic: candidate real
# controllers range from a 600-block host PD (dex/robot-control) to libfranka's
# internal joint impedance around 1000-1500 Nm/rad (franka_fr3 ros2_control,
# safe ceiling 5000). Gain provenance is a dataset input (kp_used/kd_used) via
# --warmstart_from_data, not something the bounds should hard-code.
_MAIN_STIFFNESS_BOUNDS = torch.tensor([50.0, 5000.0])
_WRIST_STIFFNESS_BOUNDS = torch.tensor([20.0, 3000.0])
_MAIN_DAMPING_BOUNDS = torch.tensor([0.5, 300.0])
_WRIST_DAMPING_BOUNDS = torch.tensor([0.1, 150.0])


def build_bounds(joint_order: list[str]) -> torch.Tensor:
    """Build the CMA-ES bounds tensor of shape (2*N, 2) for the given joint list."""
    rows: list[torch.Tensor] = []
    for j in joint_order:  # stiffness row per joint
        rows.append(_WRIST_STIFFNESS_BOUNDS if j in _WRIST_JOINTS else _MAIN_STIFFNESS_BOUNDS)
    for j in joint_order:  # damping row per joint
        rows.append(_WRIST_DAMPING_BOUNDS if j in _WRIST_JOINTS else _MAIN_DAMPING_BOUNDS)
    assert len(rows) == 2 * len(joint_order)
    return torch.stack(rows, dim=0)


##
# Physics preset — mjwarp/Newton is the default backend for this task.
##


@configclass
class FR3SysidPhysicsCfg(PresetCfg):
    physx: PhysxCfg = PhysxCfg()

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=50,
            nconmax=20,
            integrator="implicitfast",
            # Free-air chirps on a fixed-base arm never make contact.
            disable_contacts=True,
        ),
        num_substeps=1,
    )

    default = newton_mjwarp


##
# Scene
##


@configclass
class FR3SysidSceneCfg(InteractiveSceneCfg):
    """Minimal scene: fixed-base FR3 and a light. No ground — nothing touches it."""

    robot: ArticulationCfg = FR3_SYSID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


##
# Environment
##


@configclass
class FR3SysIdCfg(SysIdCfg):
    robot_name: str = "franka_fr3"
    joint_order: list[str] = FR3_SYSID_JOINT_ORDER
    # The FR3 firmware adds gravity (and Coriolis) compensation on top of its
    # internal impedance loop; a zero-g sim approximates the compensated plant so
    # the fitted gains don't have to absorb gravity-holding torque. Set False to
    # fit against the raw dynamics instead.
    zero_gravity: bool = True


@configclass
class FR3SysIdEnvCfg(ManagerBasedRLEnvCfg):
    """CMA-ES sysid env for FR3 implicit-actuator gains; driven by scripts/sysid/fit.py."""

    scene: FR3SysidSceneCfg = FR3SysidSceneCfg(num_envs=256, env_spacing=1.5)
    observations: SysidObservationsCfg = SysidObservationsCfg()
    actions: SysidActionsCfg = SysidActionsCfg()
    rewards: SysidRewardsCfg = SysidRewardsCfg()
    events: SysidEventCfg = SysidEventCfg()
    terminations: SysidTerminationsCfg = SysidTerminationsCfg()

    sysid: FR3SysIdCfg = FR3SysIdCfg()

    def __post_init__(self) -> None:
        # The real FR3 impedance loop is FCI-locked at 1 kHz while the collector
        # streams targets at 200 Hz (zero-order-held for ~5 firmware cycles).
        # decimation reproduces that hold exactly: the action target is applied
        # unchanged for `decimation` solver substeps. 1 kHz physics also
        # resolves the stiff low-inertia wrist joints that 200 Hz would not.
        # fit.py re-derives both from the dataset (controller_update_rate_hint).
        self.decimation = 5
        self.sim.dt = 0.001  # 1 kHz; env step rate = sim.dt * decimation = 200 Hz
        self.sim.render_interval = self.decimation
        # fit.py overrides this from the trajectory length; the default is high
        # enough that a time_out reset can never silently corrupt a replay.
        self.episode_length_s = 120.0
        self.sim.physics = FR3SysidPhysicsCfg()
        if self.sysid.zero_gravity:
            self.sim.gravity = (0.0, 0.0, 0.0)
