# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare one-way and two-way G1 coupling across three shallow MPM strips.

Both variants use the published ``Isaac-Velocity-Flat-G1`` Newton-MJWarp
checkpoint, a fixed forward command, and the same rigid runway. Three adjacent
16 cm particle layers cover equal thirds of the course: dry sand, packed snow,
and clay. In the one-way variant the robot moves the particles without receiving
their reaction forces; the two-way variant returns those forces to the robot.

.. code-block:: bash

    uv run --extra rsl-rl python scripts/demos/mpm/tuning/g1_coupling.py \
      --coupling two_way --visualizer kit
"""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation

TASK = "Isaac-Velocity-Flat-G1"
ROBOT_PATTERN = r"/World/envs/env_.*/Robot"
FOOT_PATTERN = r"/World/envs/env_.*/Robot/(left|right)_ankle_roll_link"
LOWER_LEG_PATTERN = r"/World/envs/env_.*/Robot/(left|right)_(knee|ankle_pitch|ankle_roll)_link"
RUNWAY_PATTERN = r"/World/envs/env_.*/RigidRunway"
ROBOT_START_X = -2.10
STRIP_START_X = -0.90
STRIP_LENGTH = 1.20
STRIP_HALF_WIDTH = 0.75
STRIP_THICKNESS = 0.16
STRIP_NAMES = ("sand", "snow", "clay")
CAMERA_EYE = (0.0, -6.2, 2.10)
CAMERA_TARGET = (0.0, 0.0, 0.52)
CAMERA_FOCAL_LENGTH = 34.0
CAMERA_LEAD = 0.70
CAMERA_TRACK_MIN_X = ROBOT_START_X + CAMERA_LEAD
CAMERA_TRACK_MAX_X = STRIP_START_X + len(STRIP_NAMES) * STRIP_LENGTH + 1.0
PARTICLES_PER_VOXEL_AXIS = 2.0
SHIN_PROXY_RADIUS = 0.05
SHIN_PROXY_HEIGHT = 0.22
SHIN_PROXY_OFFSET = (0.0, 0.0, -0.14)
STRIP_COLORS = {
    "sand": (0.76, 0.48, 0.20),
    "snow": (0.76, 0.89, 1.00),
    "clay": (0.62, 0.28, 0.16),
}


class StripMaterialPreset(NamedTuple):
    """Material values for one particle strip."""

    label: str
    density: float
    young_modulus: float
    friction: float
    yield_pressure: float
    tensile_yield_ratio: float
    yield_stress: float
    hardening: float
    dilatancy: float
    viscosity: float


STRIP_MATERIAL_PRESETS = {
    # Values start from the sand, snow, and mud rows of Table 5 in Daviet's
    # mixed-MPM paper. A finite 1 PPa value represents the tabulated rigid
    # elastic limit while remaining valid input to Newton's schema.
    "sand": StripMaterialPreset("dry sand", 1600.0, 1.0e15, 0.48, 1.0e15, 0.0, 0.0, 0.0, 0.0, 0.0),
    "snow": StripMaterialPreset(
        "snow",
        250.0,
        1.0e15,
        0.30,
        2.0e6,
        0.05,
        0.0,
        1.0,
        1.0,
        0.0,
    ),
    "clay": StripMaterialPreset(
        "cohesive clay",
        1500.0,
        1.0e15,
        0.0,
        1.0e15,
        1.0,
        200.0,
        0.0,
        0.1,
        100.0,
    ),
}

parser = argparse.ArgumentParser(description="G1 one-way versus two-way Newton MPM walking comparison.")
parser.add_argument(
    "--coupling",
    choices=("one_way", "two_way"),
    default="two_way",
    help="Whether particle reaction forces are returned to the robot.",
)
parser.add_argument("--checkpoint", type=str, default=None, help="G1 RSL-RL checkpoint; defaults to published.")
parser.add_argument("--command_speed", type=float, default=0.6, help="Fixed forward velocity command [m/s].")
parser.add_argument("--seed", type=int, default=42, help="Environment and policy seed.")
parser.add_argument("--voxel_size", type=float, default=0.040, help="MPM grid voxel size [m].")
parser.add_argument(
    "--particle_jitter_fraction",
    type=float,
    default=0.30,
    help="Maximum per-axis particle offset as a fraction of particle spacing.",
)
parser.add_argument("--mpm_substeps", type=int, default=4, help="MPM substeps per coupled simulation tick.")
parser.add_argument("--proxy_mass_scale", type=float, default=1.0, help="Robot proxy inertia multiplier.")
parser.add_argument(
    "--proxy_relaxation",
    type=float,
    default=0.1,
    help="Under-relaxation applied to the two-way proxy feedback force.",
)
parser.add_argument(
    "--proxy_mode",
    choices=("lagged", "staggered"),
    default="staggered",
    help="Newton virtual-proxy scheduling mode; both choices preserve two-way feedback.",
)
parser.add_argument(
    "--proxy_bodies",
    choices=("feet", "lower_legs", "robot"),
    default="robot",
    help=(
        "Robot bodies exposed as MPM proxies; lower legs includes explicit shin colliders and both feet, while "
        "robot loads G1's full collision asset and includes its torso, head, arms, hands, and legs."
    ),
)
parser.add_argument(
    "--max_steps", type=int, default=-1, help="Stop after this many policy steps; negative runs forever."
)
parser.add_argument("--disable_cuda_graph", action="store_true", help="Disable Newton CUDA graphs for debugging.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

if not np.isfinite(args_cli.command_speed) or args_cli.command_speed <= 0.0:
    parser.error("--command_speed must be finite and positive")
if not np.isfinite(args_cli.voxel_size) or args_cli.voxel_size <= 0.0:
    parser.error("--voxel_size must be finite and positive")
if not np.isfinite(args_cli.particle_jitter_fraction) or not 0.0 <= args_cli.particle_jitter_fraction < 0.5:
    parser.error("--particle_jitter_fraction must be finite and in [0, 0.5)")
if not np.isfinite(args_cli.proxy_mass_scale) or args_cli.proxy_mass_scale <= 0.0:
    parser.error("--proxy_mass_scale must be finite and positive")
if not np.isfinite(args_cli.proxy_relaxation) or args_cli.proxy_relaxation < 0.0:
    parser.error("--proxy_relaxation must be finite and nonnegative")
if args_cli.mpm_substeps <= 0:
    parser.error("--mpm_substeps must be positive")


def _spawn_hidden_proxy_capsule(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Any:
    """Spawn an invisible capsule that remains active for collision."""
    from isaaclab.sim.spawners.shapes import spawn_capsule
    from isaaclab.sim.utils import set_prim_visibility

    prim = spawn_capsule(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    set_prim_visibility(prim, False)
    return prim


def _kinematic_box(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    contact_margin: float,
) -> Any:
    """Create one visible kinematic collider."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.NewtonRigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=contact_margin,
                contact_gap=0.0,
            ),
            physics_material=sim_utils.NewtonMaterialPropertiesCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.48,
                metallic=0.08,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def _configure_common_environment(env_cfg: Any) -> None:
    """Apply deterministic playback settings shared by both coupling modes."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from isaaclab_assets import G1_CFG

    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 0.0
    env_cfg.scene.terrain = None
    env_cfg.scene.height_scanner = None
    env_cfg.scene.contact_forces = None
    env_cfg.observations.policy.height_scan = None
    env_cfg.scene.robot.init_state.pos = (ROBOT_START_X, 0.0, 0.74)
    if args_cli.proxy_bodies == "robot":
        # The velocity task normally uses g1_minimal.usd, whose only collision
        # bodies are the feet and torso. The full asset keeps the same body and
        # joint topology while adding authored, link-local collision geometry
        # for the pelvis, head, limbs, palms, and fingers.
        env_cfg.scene.robot.spawn.usd_path = G1_CFG.spawn.usd_path
        # Those collision meshes are USD instances. Make them editable so the
        # recursive Newton contact-margin override below reaches every full-body
        # collider instead of only the two non-instanced foot boxes.
        env_cfg.scene.robot.spawn.make_uninstanceable = True
    particle_spacing = args_cli.voxel_size / PARTICLES_PER_VOXEL_AXIS
    env_cfg.scene.robot.spawn.collision_props = sim_utils.NewtonCollisionPropertiesCfg(
        contact_margin=particle_spacing,
        contact_gap=0.0,
    )
    env_cfg.episode_length_s = 100.0
    env_cfg.curriculum.terrain_levels = None
    env_cfg.ui_window_class_type = None

    command = env_cfg.commands.base_velocity
    command.resampling_time_range = (100.0, 100.0)
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = True
    command.debug_vis = False
    command.ranges.lin_vel_x = (args_cli.command_speed, args_cli.command_speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)

    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_com = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.terminations.base_contact = None
    env_cfg.rewards.feet_air_time = None
    env_cfg.rewards.feet_slide = None

    requested_visualizers = args_cli.visualizer or []
    visualizer_cfgs = []
    if "kit" in requested_visualizers:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        visualizer_cfgs.append(KitVisualizerCfg(eye=CAMERA_EYE, lookat=CAMERA_TARGET, focal_length=CAMERA_FOCAL_LENGTH))
    if {"newton", "newton_gl", "newton_rtx"}.intersection(requested_visualizers):
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

        cfg_type = NewtonRTXVisualizerCfg if requested_visualizers == ["newton_rtx"] else NewtonGLVisualizerCfg
        visualizer_kwargs = {}
        if cfg_type is NewtonRTXVisualizerCfg:
            # Keep the standalone comparison independent of optional HDR assets.
            visualizer_kwargs = {"rtx_environment": "studio", "dome_texture_file": None}
        visualizer_cfgs.append(
            cfg_type(
                eye=CAMERA_EYE,
                lookat=CAMERA_TARGET,
                streaming_view=False,
                show_particles=True,
                **visualizer_kwargs,
            )
        )
    env_cfg.sim.visualizer_cfgs = visualizer_cfgs
    env_cfg.scene.backdrop = AssetBaseCfg(
        prim_path="/World/PresentationGround",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 16.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.17)),
    )
    if args_cli.proxy_bodies == "lower_legs":
        for side in ("left", "right"):
            setattr(
                env_cfg.scene,
                f"{side}_shin_mpm_proxy",
                AssetBaseCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{side}_knee_link/MPMProxyCollider",
                    spawn=sim_utils.CapsuleCfg(
                        func=_spawn_hidden_proxy_capsule,
                        radius=SHIN_PROXY_RADIUS,
                        height=SHIN_PROXY_HEIGHT,
                        axis="Z",
                        collision_props=sim_utils.NewtonCollisionPropertiesCfg(
                            collision_enabled=True,
                            contact_margin=particle_spacing,
                            contact_gap=0.0,
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(pos=SHIN_PROXY_OFFSET),
                ),
            )


def _configure_particle_runway(env_cfg: Any) -> None:
    """Add the rigid runway, three particle strips, and directed coupler."""
    from isaaclab_newton.assets import MPMObjectCfg
    from isaaclab_newton.physics import MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
    from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

    from isaaclab_contrib.coupling import (
        CouplerEntryCfg,
        CouplerProxyCfg,
        CouplerProxyMappingCfg,
    )

    previous_physics = env_cfg.sim.physics
    rigid_solver_cfg = previous_physics.solver_cfg
    particle_spacing = args_cli.voxel_size / PARTICLES_PER_VOXEL_AXIS
    jitter_width = 2.0 * args_cli.particle_jitter_fraction * particle_spacing

    env_cfg.scene.rigid_runway = _kinematic_box(
        "{ENV_REGEX_NS}/RigidRunway",
        size=(12.0, 2.40, 0.16),
        position=(1.35, 0.0, -0.08),
        color=(0.20, 0.26, 0.34),
        contact_margin=particle_spacing,
    )

    for index, name in enumerate(STRIP_NAMES):
        material = STRIP_MATERIAL_PRESETS[name]
        lower_x = STRIP_START_X + index * STRIP_LENGTH
        setattr(
            env_cfg.scene,
            f"{name}_strip",
            MPMObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{name.title()}Strip",
                spawn=MPMGridCfg(
                    lower=(lower_x, -STRIP_HALF_WIDTH, 0.0),
                    upper=(lower_x + STRIP_LENGTH, STRIP_HALF_WIDTH, STRIP_THICKNESS),
                    voxel_size=args_cli.voxel_size,
                    particles_per_cell=PARTICLES_PER_VOXEL_AXIS,
                    particle_placement="cell_center",
                    jitter=jitter_width,
                    material=MPMParticleMaterialCfg(
                        density=material.density,
                        young_modulus=material.young_modulus,
                        poisson_ratio=0.3,
                        viscosity=material.viscosity,
                        friction=material.friction,
                        damping=0.01,
                        yield_pressure=material.yield_pressure,
                        tensile_yield_ratio=material.tensile_yield_ratio,
                        yield_stress=material.yield_stress,
                        hardening=material.hardening,
                        dilatancy=material.dilatancy,
                    ),
                    visual_color=STRIP_COLORS[name],
                ),
            ),
        )

    robot_proxy_patterns = {
        "feet": FOOT_PATTERN,
        "lower_legs": LOWER_LEG_PATTERN,
        "robot": ROBOT_PATTERN,
    }
    proxy_selectors = [robot_proxy_patterns[args_cli.proxy_bodies], RUNWAY_PATTERN]
    feedback_relaxation = args_cli.proxy_relaxation if args_cli.coupling == "two_way" else 0.0

    coupler_cfg = CouplerProxyCfg(
        entries=[
            CouplerEntryCfg(
                name="rigid",
                solver_cfg=rigid_solver_cfg,
                bodies=[ROBOT_PATTERN, RUNWAY_PATTERN],
                include_static_shapes=False,
                substeps=2,
            ),
            CouplerEntryCfg(
                name="mpm",
                solver_cfg=MPMSolverCfg(
                    voxel_size=args_cli.voxel_size,
                    grid_type="sparse",
                    grid_padding=0,
                    max_active_cell_count=1 << 18,
                    max_leaf_node_count=1 << 15,
                    max_lower_node_count=1 << 11,
                    max_upper_node_count=1 << 9,
                    max_iterations=36,
                    tolerance=1.0e-4,
                    solver="auto",
                    warmstart_mode="auto",
                    transfer_scheme="pic",
                    strain_basis="P0",
                    velocity_basis="Q1",
                    collider_basis="pic27",
                    collider_velocity_mode="forward",
                    air_drag=1.0,
                    project_outside_colliders=False,
                    separate_worlds=True,
                ),
                all_particles=True,
                include_static_shapes=False,
                include_child_joints=False,
                substeps=args_cli.mpm_substeps,
                in_place=True,
            ),
        ],
        proxies=[
            CouplerProxyMappingCfg(
                source="rigid",
                destination="mpm",
                bodies=proxy_selectors,
                mode=args_cli.proxy_mode,
                mass_scale=args_cli.proxy_mass_scale,
                proxy_relaxation=feedback_relaxation,
                collision_pipeline=None,
            )
        ],
        iterations=1,
    )
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=coupler_cfg,
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        soft_contact_cfg=previous_physics.soft_contact_cfg,
        default_shape_cfg=previous_physics.default_shape_cfg,
        num_substeps=1,
        use_cuda_graph=not args_cli.disable_cuda_graph,
        load_visual_shapes=previous_physics.load_visual_shapes,
    )


def configure_environment(env_cfg: Any) -> None:
    """Configure the deterministic three-material G1 comparison."""
    _configure_common_environment(env_cfg)
    _configure_particle_runway(env_cfg)


def _tracking_camera_pose(base_x: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return a side-on dolly camera that keeps G1 and the next strip visible."""
    camera_x = float(np.clip(base_x + CAMERA_LEAD, CAMERA_TRACK_MIN_X, CAMERA_TRACK_MAX_X))
    return (camera_x, CAMERA_EYE[1], CAMERA_EYE[2]), (camera_x, CAMERA_TARGET[1], CAMERA_TARGET[2])


def main() -> None:
    """Load the published policy and play the selected coupling variant."""
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, agent_cfg = resolve_task_config(
        TASK,
        "rsl_rl_cfg_entry_point",
        play_mode=True,
        overrides=("physics=newton_mjwarp",),
    )

    if args_cli.checkpoint:
        checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"G1 checkpoint not found: {checkpoint}")
    else:
        from isaaclab_rl.entrypoints.common import resolve_play_checkpoint

        # Resolve against the published task preset before replacing its physics
        # selector with the coupled rigid/MPM solver configuration below.
        checkpoint = Path(resolve_play_checkpoint(None, "rsl_rl", TASK, env_cfg)).resolve()
    print(f"[INFO]: Resolved G1 checkpoint: {checkpoint}", flush=True)
    configure_environment(env_cfg)
    print("[INFO]: Configured G1 rigid/MPM environment; launching visualizer.", flush=True)
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        import torch
        from rsl_rl.runners import OnPolicyRunner

        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.seed import configure_seed

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        if args_cli.deterministic:
            configure_seed(args_cli.seed, torch_deterministic=True)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()

        strip_particle_counts = {
            name: env.unwrapped.scene[f"{name}_strip"].particles_per_object for name in STRIP_NAMES
        }
        print(
            f"[INFO]: G1 {args_cli.coupling} playback ready with {sum(strip_particle_counts.values())} particles; "
            f"{1000 * args_cli.voxel_size:g} mm voxels and {100 * STRIP_THICKNESS:g} cm strips.",
            flush=True,
        )

        step = 0
        try:
            while env.unwrapped.sim.is_headless_or_exist_active_visualizer() and (
                args_cli.max_steps < 0 or step < args_cli.max_steps
            ):
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    if hasattr(policy, "reset"):
                        policy.reset(dones)
                step += 1
                base_x = float(env.unwrapped.scene["robot"].data.root_pos_w.torch[0, 0])
                eye, target = _tracking_camera_pose(base_x)
                env.unwrapped.sim.set_camera_view(eye=eye, target=target)
        finally:
            robot = env.unwrapped.scene["robot"]
            root_pos = robot.data.root_pos_w.torch[0]
            root_velocity = robot.data.root_lin_vel_w.torch[0]
            print(
                f"[INFO]: G1 stopped after {step} steps at x={float(root_pos[0]):.3f} m, "
                f"height={float(root_pos[2]):.3f} m, forward speed={float(root_velocity[0]):.3f} m/s.",
                flush=True,
            )
            env.close()


if __name__ == "__main__":
    main()
