# G1 locomotion on a fixed sand bed

This manager-based task starts a G1 robot on a rigid approach platform next to a fixed MPM sand bed. MJWarp simulates the robot and rigid colliders. Implicit MPM simulates the sand, and proxy coupling transfers sand reaction forces to the robot.

The task uses the contributed G1 box-foot asset. Observations and rewards include robot foot contact forces from the rigid platform and the MPM sand. Reset events sample particle material parameters with uniform or log-uniform distributions. Symmetry augmentation uses the shared `isaaclab_contrib.mdp.symmetry` API and task-local G1 joint mappings.

```bash
uv run isaaclab train --rl_library rsl_rl --task IsaacContrib-Velocity-Sand-G1-29dof-MPM --num_envs 2
uv run --with wandb isaaclab play --rl_library rsl_rl --task IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play --num_envs 1 --wandb_run <run-id> --viz newton_rtx
```

The environment configuration is `g1_mpm_env_cfg.py`. Scene geometry and particle material defaults are in `env_cfg/scene_cfg.py`; material sampling ranges are in `env_cfg/event_cfg.py`. The robot's initial position is on the rigid approach platform.

Coupled material updates require Newton's particle-material synchronization fix. Multi-environment MPM requires the Warp multi-environment cell-lookup fix. Until compatible upstream dependency pins are available, use the dependency versions described in the draft PR validation notes.

### Tilting testbed

`IsaacContrib-Balance-Sand-G1-29dof-MPM-Tilting` starts G1 in the sand with a zero
velocity command. Its `-Play` variant disables randomization. The stationary
approach platform is replaced by a closed tray attached to roll and pitch joints.

```bash
uv run --extra rsl-rl isaaclab train --rl_library rsl_rl \
  --task IsaacContrib-Balance-Sand-G1-29dof-MPM-Tilting --num_envs 4 --viz none

uv run --extra rsl-rl isaaclab play --rl_library rsl_rl \
  --task IsaacContrib-Balance-Sand-G1-29dof-MPM-Tilting-Play --num_envs 1 \
  --viz newton_gl --checkpoint /path/to/model.pt
```

Configure `testbed_motion` in `G1TiltingMPMEnvCfg`: roll/pitch ranges are in radians,
defaulting to ±15°, with 12 s and 17 s periods. The tray holds level for 2 s after
reset, then ramps into the oscillation over 3 s. Mechanical joint limits are ±45°.
For example, `roll_range=(math.radians(-5), math.radians(10))` sets the roll target
range to −5° through +10°. Each target range must include zero and lie strictly
inside the mechanical limits (`scene.testbed.spawn.joint_limit_deg`).

`testbed_motion.debug_vis=True` draws two arrows at a tray corner: blue follows
the actual tray normal; green points opposite gravity (world-up with default gravity).
The angle between them shows the tray's tilt. Set it to `False` to hide
the arrows. The tray geometry and actuator configuration live in
`env_cfg/scene_cfg.py:G1TiltingMPMSceneCfg`.

Targets update at 50 Hz; MJWarp integrates the driven joints at 400 Hz and MPM
updates at 200 Hz with the default settings. The tray is a dynamic proxy collider
in MPM. Its actual angle can differ from its target under load. No particle or
tray poses are overwritten during stepping; episode resets restore the tray,
robot, particles, and coupling history together. Bounds and base-height rewards
use the rotating tray frame; the height reference is the nominal sand surface,
not the instantaneous deformed particle surface.

### Rough fixed bed

`IsaacContrib-Velocity-Sand-Rough-G1-29dof-MPM` and its `-Play` variant spawn a
particle layer below a terrain-generator surface. The default is a planar ramp
along +X. Configure the generator in `env_cfg/scene_cfg.py:G1RoughMPMSceneCfg`.
Both solvers use the same generated support mesh, lowered by the sand depth.
Nonzero particle jitter is supported when the jittered XY positions remain inside
the mesh. Missing height queries raise an error rather than spawning invalid particles.

The rough variant retains the base locomotion rewards, including flat-ground
height and touchdown references; adapt those references before rough-terrain training.

### Solver settings

The active-cell capacity is derived from the particle grid when
`mpm_active_cell_count_per_world=None`; an explicit value overrides it. Other
sparse-grid capacities remain configurable per environment. Particle material
randomization is configured in `env_cfg/event_cfg.py`.

The tilting task drives physical roll/pitch joints; it does not teleport the tray
or particles during stepping. Its base-height reward and workspace termination
use the tray frame. Scene configuration remains in `env_cfg/scene_cfg.py`.
