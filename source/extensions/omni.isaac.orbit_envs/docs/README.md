# Orbit: Environment Suite

Using the core framework developed as part of Orbit, we provide various learning environments for robotics research.
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
omni/isaac/orbit_envs/locomotion/
├── __init__.py
└── velocity
    ├── a1
    │   └── flat_terrain_cfg.py
    ├── anymal_c
    │   └── flat_terrain_cfg.py
    ├── __init__.py
    ├── velocity_cfg.py
    └── velocity_env.py
```

The environments are then registered in the `omni/isaac/orbit_envs/__init__.py`:

```python
gym.register(
    id="Isaac-Velocity-Anymal-C-v0",
    entry_point="omni.isaac.orbit_envs.locomotion.velocity:VelocityEnv",
    kwargs={"cfg_entry_point": "omni.isaac.orbit_envs.locomotion.velocity.anymal_c.flat_terrain_cfg:FlatTerrainCfg"},
)

gym.register(
    id="Isaac-Velocity-A1-v0",
    entry_point="omni.isaac.orbit_envs.locomotion.velocity:VelocityEnv",
    kwargs={"cfg_entry_point": "omni.isaac.orbit_envs.locomotion.velocity.a1.flat_terrain_cfg:FlatTerrainCfg"},
)
```

> **Note:** As a practice, we specify all the environments in a single file to avoid name conflicts between different
> tasks or environments. However, this practice is debatable and we are open to suggestions to deal with a large
> scaling in the number of tasks or environments.
