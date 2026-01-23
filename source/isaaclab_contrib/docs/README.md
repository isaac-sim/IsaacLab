# Isaac Lab: Community Contributions

This extension (`isaaclab_contrib`) provides a collection of community-contributed components for Isaac Lab. These contributions extend the core framework with specialized robot types, actuator models, sensors, and other features that are not yet part of the main Isaac Lab package but are actively maintained and supported by the community.

## Overview

The `isaaclab_contrib` package serves as an incubator for experimental and specialized features that:

- Extend Isaac Lab's capabilities for specific robot types or use cases
- Provide domain-specific actuator models and control interfaces
- Offer specialized MDP components for reinforcement learning tasks
- May eventually be integrated into the core Isaac Lab framework

## Current Contributions

### Multirotor Systems

Comprehensive support for multirotor vehicles (drones, quadcopters, hexacopters, octocopters, etc.) including:

- **Assets**: `Multirotor` articulation class with thruster-based control
- **Actuators**: `Thruster` model with realistic motor dynamics
- **MDP Components**: `ThrustAction` terms for RL control

See the [Multirotor Systems](#multirotor-systems-detailed) section below for detailed documentation.

## Extension Structure

The extension follows Isaac Lab's standard package structure:

```tree
isaaclab_contrib/
├── actuators/              # Contributed actuator models
│   ├── thruster.py         # Thruster actuator for multirotors
│   └── thruster_cfg.py
├── assets/                 # Contributed asset classes
│   └── multirotor/         # Multirotor asset implementation
├── mdp/                    # MDP components for RL
│   └── actions/            # Action terms
└── utils/                  # Utility functions and types
```

## Installation and Usage

The `isaaclab_contrib` package is included with Isaac Lab. To use contributed components:

```python
# Import multirotor components
from isaaclab_contrib.assets import Multirotor, MultirotorCfg
from isaaclab_contrib.actuators import Thruster, ThrusterCfg
from isaaclab_contrib.mdp.actions import ThrustActionCfg
```

---

## Multirotor Systems (Detailed)

This section provides detailed documentation for the multirotor contribution, which enables simulation and control of multirotor aerial vehicles in Isaac Lab.

### Features

The multirotor system includes:

#### Assets

- **`Multirotor`**: A specialized articulation class that extends the base `Articulation` class to support multirotor systems with thruster actuators
  - Manages multiple thruster actuators as a group
  - Applies thrust forces at specific body locations
  - Uses allocation matrices for control allocation
  - Provides thruster-specific state information through `MultirotorData`

#### Actuators

- **`Thruster`**: A low-level motor/thruster dynamics model with realistic response characteristics:
  - **Asymmetric rise and fall time constants**: Models different spin-up/spin-down rates
  - **Thrust limits**: Configurable minimum and maximum thrust constraints
  - **Integration schemes**: Euler or RK4 integration methods
  - **Dynamic response**: Simulates motor transient behavior

#### MDP Components

- **`ThrustAction`**: Action terms specifically designed for multirotor control:
  - Direct thrust commands to individual thrusters or groups
  - Flexible preprocessing (scaling, offsetting, clipping)
  - Automatic hover thrust offset computation
  - Integrates with Isaac Lab's MDP framework for RL tasks

<details>

### Quick Start

#### Creating a Multirotor Asset

```python
import isaaclab.sim as sim_utils
from isaaclab_contrib.assets import MultirotorCfg
from isaaclab_contrib.actuators import ThrusterCfg

# Define thruster actuator configuration
thruster_cfg = ThrusterCfg(
    thruster_names_expr=["rotor_[0-3]"],  # Match rotors 0-3
    thrust_range=(0.0, 10.0),              # Min and max thrust in Newtons
    rise_time_constant=0.12,               # Time constant for thrust increase (120ms)
    fall_time_constant=0.25,               # Time constant for thrust decrease (250ms)
)

# Create multirotor configuration
multirotor_cfg = MultirotorCfg(
    prim_path="/World/envs/env_.*/Quadcopter",
    spawn=sim_utils.UsdFileCfg(
        usd_path="path/to/quadcopter.usd",
    ),
    init_state=MultirotorCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),    # Start 1m above ground
        rps={".*": 110.0},      # All thrusters at 110 RPS (hover)
    ),
    actuators={
        "thrusters": thruster_cfg,
    },
    allocation_matrix=[  # 6x4 matrix for quadcopter
        [1.0, 1.0, 1.0, 1.0],       # Total vertical thrust
        [0.0, 0.0, 0.0, 0.0],       # Lateral force X
        [0.0, 0.0, 0.0, 0.0],       # Lateral force Y
        [0.0, 0.13, 0.0, -0.13],    # Roll torque
        [-0.13, 0.0, 0.13, 0.0],    # Pitch torque
        [0.01, -0.01, 0.01, -0.01], # Yaw torque
    ],
    rotor_directions=[1, -1, 1, -1],  # Alternating CW/CCW
)
```

#### Using Thrust Actions in RL Environments

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab_contrib.mdp.actions import ThrustActionCfg

@configclass
class QuadcopterEnvCfg(ManagerBasedRLEnvCfg):
    # ... scene, observations, rewards, etc. ...

    @configclass
    class ActionsCfg:
        # Normalized thrust control around hover
        thrust = ThrustActionCfg(
            asset_name="robot",
            scale=2.0,                    # Actions in [-1,1] become [-2,2] N deviation
            use_default_offset=True,      # Add hover thrust from config
            clip={".*": (0.0, 10.0)},    # Constrain final thrust to [0, 10] N
        )

    actions = ActionsCfg()
```

### Key Concepts

#### Allocation Matrix

The allocation matrix maps individual thruster forces to a 6D wrench (force + torque) applied to the multirotor's base link:

```
wrench = allocation_matrix @ thrust_vector
```

Where:
- `wrench`: [Fx, Fy, Fz, Tx, Ty, Tz]ᵀ (6D body wrench)
- `allocation_matrix`: 6 × N matrix (6 DOF, N thrusters)
- `thrust_vector`: [T₁, T₂, ..., Tₙ]ᵀ (N thruster forces)

The matrix encodes the geometric configuration of thrusters including positions, orientations, and moment arms.

#### Thruster Dynamics

The `Thruster` actuator model simulates realistic motor response with asymmetric first-order dynamics:

```
dT/dt = (T_target - T_current) / τ
```

Where τ is the time constant (different for rise vs. fall):
- **Rise Time (τ_rise)**: How quickly thrust increases when commanded (typically slower)
- **Fall Time (τ_fall)**: How quickly thrust decreases when commanded (typically faster)
- **Thrust Limits**: Physical constraints [T_min, T_max] enforced after integration

This asymmetry reflects real-world motor behavior primarily caused by ESC (Electronic Speed Controller) response and propeller aerodynamics, which result in slower spin-up (thrust increase) than spin-down. While rotor inertia affects both acceleration and deceleration equally, it is not the main cause of the asymmetric response.

#### Thruster Control Modes

The multirotor system supports different control approaches:

1. **Direct Thrust Control**: Directly command thrust forces/RPS
2. **Normalized Control**: Commands as deviations from hover thrust
3. **Differential Control**: Small adjustments around equilibrium

The `ThrustAction` term provides flexible preprocessing to support all modes through scaling and offsetting.

</details>

### Demo Script

A complete demonstration of quadcopter simulation is available:

```bash
# Run quadcopter demo
./isaaclab.sh -p scripts/demos/quadcopter.py
```

---

## Testing

The extension includes comprehensive unit tests for all contributed components:

```bash
# Test multirotor components
python -m pytest source/isaaclab_contrib/test/assets/test_multirotor.py
python -m pytest source/isaaclab_contrib/test/actuators/test_thruster.py

# Run all contrib tests
python -m pytest source/isaaclab_contrib/test/
```

## Contributing

We welcome community contributions to `isaaclab_contrib`! If you have developed:

- Specialized robot asset classes
- Novel actuator models
- Custom MDP components
- Domain-specific utilities

Please follow the Isaac Lab contribution guidelines and open a pull request. Contributions should:

1. Follow the existing package structure
2. Include comprehensive documentation (docstrings, examples)
3. Provide unit tests
4. Be well-tested with Isaac Lab's simulation framework

For more information, see the [Isaac Lab Contributing Guide](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).

## License

This extension follows the same BSD-3-Clause license as Isaac Lab. See the LICENSE file for details.
