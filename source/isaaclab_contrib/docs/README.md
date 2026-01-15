# Isaac Lab: MultiRotor Extension

This extension provides comprehensive support for multirotor systems (drones, quadcopters, hexacopters, etc.)
in Isaac Lab. It includes specialized actuator models, asset classes, and MDP components specifically designed
for multirotor simulation.

## Features

The extension provides the following key components:

### Assets

* **`Multirotor`**: A specialized articulation class that extends the base `Articulation` class to support
  multirotor systems with thruster actuators. This class handles the simulation of multirotor dynamics,
  including thrust application at specific body locations and integration with the thruster actuator model.

### Actuators

* **`Thruster`**: A low-level motor/thruster dynamics model with separate rise/fall time constants. This
  actuator model simulates realistic motor response characteristics including:
  - Asymmetric rise and fall time constants
  - Thrust limits (minimum and maximum)
  - Integration schemes (Euler or RK4)
  - Motor spin-up and spin-down dynamics

### MDP Components

* **Thrust Actions**: Action terms specifically designed for multirotor control, allowing direct thrust
  commands to individual thrusters or groups of thrusters. These integrate seamlessly with Isaac Lab's
  MDP framework for reinforcement learning tasks.

## Using the Extension

To use this extension in your code, import the required components:

```python
from isaaclab_contrib.assets import Multirotor, MultirotorCfg
from isaaclab_contrib.actuators import Thruster, ThrusterCfg
from isaaclab_contrib.mdp.actions import ThrustActionCfg
```

### Example: Creating a Multirotor Asset

Here's how to configure and create a multirotor asset:

```python
import isaaclab.sim as sim_utils
from isaaclab_contrib.assets import MultirotorCfg
from isaaclab_contrib.actuators import ThrusterCfg

# Define thruster actuator configuration
thruster_cfg = ThrusterCfg(
    thrust_limit=(0.0, 10.0),  # Min and max thrust in Newtons
    rise_time_constant=0.1,     # Time constant for thrust increase
    fall_time_constant=0.2,     # Time constant for thrust decrease
)

# Create multirotor configuration
multirotor_cfg = MultirotorCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="path/to/your/multirotor.usd",
    ),
    actuators={
        "thrusters": thruster_cfg,
    },
)
```

### Example: Using Thrust Actions in Environments

To use thrust actions in your RL environment:

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_contrib.mdp.actions import ThrustActionCfg

@configclass
class MyMultirotorEnvCfg(ManagerBasedRLEnvCfg):
    # ... other configuration ...

    # Define actions
    actions = ActionsCfg()

    @configclass
    class ActionsCfg:
        thrust = ThrustActionCfg(
            asset_name="robot",
            thruster_names=["motor_.*"],  # regex pattern for thruster names
        )
```

## Extension Structure

The extension follows Isaac Lab's standard structure:

```tree
isaaclab_contrib/
├── actuators/              # Thruster actuator implementations
├── assets/                 # Multirotor asset classes
│   └── multirotor/
├── mdp/                    # MDP components for RL
└── utils/                  # Utility functions and types
```

## Key Concepts

### Thruster Dynamics

The `Thruster` actuator model simulates realistic motor response with asymmetric dynamics:

- **Rise Time**: How quickly thrust increases when commanded
- **Fall Time**: How quickly thrust decreases when commanded
- **Thrust Limits**: Physical constraints on minimum and maximum thrust output

This asymmetry reflects real-world motor behavior where spinning up often takes longer than spinning down.

### Multirotor Asset

The `Multirotor` class extends the standard `Articulation` to provide specialized functionality:

- Manages multiple thruster actuators as a group
- Applies thrust forces at specific body locations
- Integrates with Isaac Lab's physics simulation
- Provides thruster-specific state information through `MultirotorData`

## Testing

The extension includes comprehensive unit tests:

```bash
# Test thruster actuator
python -m pytest source/isaaclab_contrib/test/actuators/test_thruster.py

# Test multirotor asset
python -m pytest source/isaaclab_contrib/test/assets/test_multirotor.py
```
