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

## Standalone physics for GR1T2 and G1 teleoperation tasks

For standalone OVPhysX, the fixed-base G1 task's backend preparation hook moves the
tabletop collision anchor onto its existing meshes, using the same preparation as GR1T2.
OVPhysX cooks the steering wheel's authored collision meshes. The bin's existing
primitive colliders and the object's drop termination remain enabled.
Registered GR1T2 and G1 tasks prepare the selected backend before scene construction,
including when created through `gym.make()` or `instantiate(cfg)` without the
teleoperation launcher. Newton uses its separate backend preparation. Resolve a
fresh task configuration when changing backends after preparation; copying a prepared
configuration preserves its selected backend and caller overrides.

## Fixed-base G1 manipulation cameras

`IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs` provides three 640×480 RGB
cameras: `robot_pov_cam` at 20 Hz, and `left_wrist_cam` and `right_wrist_cam` at
10 Hz each. The head uses the calibrated G1 head-camera pose; wrist cameras
are mounted on their respective wrist-yaw links and look along wrist-local +X
with +Z up. Their offsets are configurable in the task's scene configuration.

The cameras feed the task's XR camera panels. The policy observation space stays
state-only by default. Select `presets=rgb` to add the three images to the policy
group, for example to record demonstrations with them:

```bash
uv run isaaclab teleop record --task IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs \
    --xr presets=rgb
```

`image_obs_list` names the three cameras; image consumers such as robomimic use
only the names present in the policy observations.

The same names identify the policy image observations and the task's XR camera
feeds, with the head panel centered and wrist panels to its left and right.
In Kit, sensor `update_period` controls image sampling in simulation time; feed
`max_update_hz` caps uploads. In OVXR, each panel renders natively at the lower of
`--ovxr_pip_fps` and its feed's nonzero `max_update_hz`; `update_period` does not
apply. Actual delivery also depends on simulation and renderer throughput. Camera
sampling rates are configuration targets, not measured headset frame rates.

With `--xr`, the teleoperation launcher's `--disable_external_cameras` option removes
these cameras and their image observations. The kitless XR launcher accepts up to
three camera names with `--ovxr_pip_camera`. Camera feeds are opt-in: without
this option, live OVXR and MCAP replay omit task cameras and PiP layers. Selected
OVXR cameras feed the panels only: their policy image observations are removed even
with `presets=rgb`, so the application does not request a second camera render.
The RGB recording example above uses Kit `--xr`.
