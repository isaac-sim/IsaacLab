# Eigenbot IsaacLab Extension

This is an external extension for IsaacLab to add the EigenBot robot system and training. All modifications to code should generally stay within this extension.

As with the original IsaacGym implementation, we use `rsl_rl` as the library for RL and robotic learning, a direct workflow to easily port from `legged_gym`, and single-agent control, since EigenBot currently does not need to itneract with other robots.

Once full porting is complete, RL is stress tested, and ROS migration is finished, we can conduct another port to a manager-based workflow to improve modularity, but this is not needed currently.

## Overview
Here is an overview of the extension structure.

```
eigenbot/
├── scripts/
│   ├── list_envs.py
│   ├── random_agent.py
│   ├── zero_agent.py
│   └── rsl_rl/
│       ├── cli_args.py
│       ├── play.py
│       └── train.py
└── source/
    └── eigenbot/
        ├── setup.py
        └── eigenbot/
            ├── ui_extension_example.py
            ├── assets/
            │   ├── eigenbot.py
            │   └── eigenbot/
            │       ├── urdf/
            │       │   └── eigenbot_hexapod.urdf
            │       └── meshes/
            │           └── *.stl
            └── tasks/
                └── direct/
                    └── eigenbot/
                        ├── eigenbot_env.py
                        ├── eigenbot_env_cfg.py
                        └── agents/
                            └── rsl_rl_ppo_cfg.py
```

## Specifics

### Assets
The `source/eigenbot/eigenbot/assets` folder will contain all the model files, URDFs, and supporting USD-style files necessary for rendering the EigenBot. The `eigenbot/meshes` subfolder should contain all `.stl, .obj, .png` files needed for meshes and textures, and the `eigenbot/urdf` file should contain the EigenBot URDF file.

#### Modifying
When modifying assets, add additional mesh, texture, and URDF files to the corresponding folders, being careful that naming is consitent and all URDF dependencies are satisfied and pathed correctly. Then, modify the simulation core to create compatability.

### Simulation Core
The simulation core is contained within two main files.
- `source/eigenbot/eigenbot/assets/eigenbot.py` contains the physics properties, joint properties, actuators, and initialization pose for the EigenBot as an `ArticulationCfg`.
- `source/eigenbot/tasks/direct/eigenbot/eigenbot_env_cfg.py` contains the environment properties, consisting of environment rewards, physics, sensing, interactions, and terrain subconfigs, wrapped within the `EigenbotEnvCfg` main config.

#### Modifying
It is relatively easily to modify environment properties or add/remove properties by modifying the configuration files and treating any downstream effects.

### RL Core
The reinforcement leearning core is contained within two main files.
- `source/eigenbot/tasks/direct/eigenbot/eigenbot_env.py`contains the `EigenbotEnv` class which holds all the rewards functions, computations, actions, reset, etc. functionality for an RL environment that will train the EigenBot. This is the most complex file, but think about it as just serving an RL environment.
- `source/eigenbot/tasks/direct/eigenbot/agents/rsl_rl_ppo_cfg.py` configures the PPO training policy from RSL_RL to train the robot.

#### Modifying
For changing the training hyperparameters, this can be easily done by changing the PPO training config. For changing the RL rewards themselves, more complex programming needs to be done to modify the `EigenbotEnv` class.

### RL Scripting
The `scripts/rsl_rl` folder contains scripts to train, play, and process command line arguments for RL policy training. The `scripts`folder itself also has scripts to list the registered environments in the EigenBot extension and run smoke tests with a random action agent or a no-action agent to test gravity and environment function.

### Extras
The `source/eigenbot/setup.py` file is a metadata file to indicate how the EigenBot extension should be bundled and installed as a Python package in the Docker image. The `source/eigenbot/eigenbot/ui_extension_example.py` is an example for how Omniverse Kit UI extensions should be made to be added to the IsaacSim interface.

## Changes and More Work
- **Terrain curriculum not yet implemented**. The current env uses flat ground. Height observations return constant values.
- **Physical domain randomization (friction/mass/COM applied to physics)** requires explicit torque control mode — currently values are stored for privileged observations only.
- **Depth Camera**. This sensor was omitted from the current port for simplicity. It needs to be added back using a `CameraCfg` sensor and a RSL_RL policy class to support sim and training.
- **Action Delay**. This was a disabled by default feature in the IsaacGym implementation, which simulates latency by adding action frames to a history buffer. Currently only the observation history buffer is ported, but the action history buffer is not.
- **Scan/privileged encoders**. Original IsaacGym uses specialized `scan_encoder` and `priv_encoder` networks. The current config uses a standard MLP. A custom RSL_RL policy class would be needed for the full encoder architecture. This is however, an RL training rewrite, not a sim part.