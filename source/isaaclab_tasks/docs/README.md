# Isaac Lab: Environment Suite

Using the core framework developed as part of Isaac Lab, we provide various learning environments for robotics research.
These environments follow the `gym.Env` API from OpenAI Gym version `0.21.0`. The environments are registered using
the Gym registry.

Each environment's name is composed of `Isaac-<Task>-<Robot>-v<X>`, where `<Task>` indicates the skill to learn
in the environment, `<Robot>` indicates the embodiment of the acting agent, and `<X>` represents the version of
the environment (which can be used to suggest different observation or action spaces).

The environments are configured using either Python classes (wrapped using `configclass` decorator) or through
YAML files. The template structure of the environment is always put at the same level as the environment file
itself. However, its various instances are included in directories within the environment directory itself.
This looks like as follows:

```tree
isaaclab_tasks/locomotion/
├── __init__.py
└── velocity
    ├── config
    │   └── anymal_c
    │       ├── agent  # <- this is where we store the learning agent configurations
    │       ├── __init__.py  # <- this is where we register the environment and configurations to gym registry
    │       ├── flat_env_cfg.py
    │       └── rough_env_cfg.py
    ├── __init__.py
    └── velocity_env_cfg.py  # <- this is the base task configuration
```

The environments are then registered in the `isaaclab_tasks/locomotion/velocity/config/anymal_c/__init__.py`:

```python
gym.register(
    id="IsaacContrib-Velocity-Rough-AnymalC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AnymalCRoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Flat-AnymalC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalCFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
```

> **Note:** As a practice, we specify all the environments in a single file to avoid name conflicts between different
> tasks or environments. However, this practice is debatable and we are open to suggestions to deal with a large
> scaling in the number of tasks or environments.

## Fixed-base G1 manipulation cameras

For standalone OVPhysX, the task's backend preparation hook installs explicit
tabletop and steering-wheel collision shapes, shared with GR1T2. The bin's
existing primitive colliders and the object's drop termination remain enabled.
The teleoperation launcher invokes this hook when selecting `--physics ovphysx`;
custom launchers must apply `stabilize_for_kitless_ovphysx()` before creating the
environment. Newton uses its separate backend preparation.

`IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs` provides three 640×480 RGB
cameras: `robot_pov_cam` at 20 Hz, and `left_wrist_cam` and `right_wrist_cam` at
10 Hz each. The head uses the calibrated G1 head-camera pose; wrist cameras
are mounted on their respective wrist-yaw links and look along wrist-local +X
with +Z up. Their offsets are configurable in the task's scene configuration.

The same names identify policy image observations and the task's XR camera
feeds, with the head panel centered and wrist panels to its left and right.
Sensor `update_period` controls image sampling in simulation time; feed
`max_update_hz` caps uploads. Actual delivery also depends on simulation and
renderer throughput. Camera sampling rates are configuration targets, not
measured headset frame rates.

The teleoperation launcher's `--disable_external_cameras` option removes these
cameras and their image observations. The kitless XR launcher accepts up to
three camera names with `--ovxr_pip_camera`. Camera feeds are opt-in: without
this option, live OVXR and MCAP replay omit task cameras and PiP layers.
Selected feeds keep their own sampling cadence.
