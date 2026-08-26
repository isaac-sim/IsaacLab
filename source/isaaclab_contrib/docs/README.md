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

### TacSL Tactile Sensor

Support for tactile sensor from [Akinola et al., 2025](https://arxiv.org/abs/2408.06506).
It uses the Taxim model from [Si et al., 2022](https://arxiv.org/abs/2109.04027) to render the tactile images.

See the [TacSL Tactile Sensor](#tacsl-tactile-sensor-detailed) section below for detailed documentation.

### Trail Library

The Trail library is a modular framework for generating parameterizable mountain-bike trails. See [this section](#trail-library-detailed) for more information.

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
├── sensors/                # Contributed sensor classes
│   └── tacsl_sensor/       # TacSL tactile sensor implementation
├── terrains/               # Contributed sensor classes
│   └── trail/              # Trail library
└── utils/                  # Utility functions and types
```

## Installation and Usage

The `isaaclab_contrib` package is included with Isaac Lab. To use contributed components:

```python
# Import multirotor components
from isaaclab_contrib.assets import Multirotor, MultirotorCfg
from isaaclab_contrib.actuators import Thruster, ThrusterCfg
from isaaclab_contrib.mdp.actions import ThrustActionCfg
from isaaclab_contrib.sensors import VisuoTactileSensor, VisuoTactileSensorCfg
```

---

## Multirotor Systems (Detailed)

This section provides detailed documentation for the multirotor contribution, which enables simulation and control of multirotor aerial vehicles in Isaac Lab.

<details>

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
from isaaclab.utils.configclass import configclass
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

A complete demonstration of multirotor simulation is available:

```bash
# Run multirotor demo
uv run python scripts/demos/arl_robot_1.py
```

## TacSL Tactile Sensor (Detailed)

This section provides detailed documentation for the TacSL tactile sensor contribution, which enables GPU-based simulation of vision-based tactile sensors in Isaac Lab. The implementation is based on the TacSL framework from [Akinola et al., 2025](https://arxiv.org/abs/2408.06506) and uses the Taxim model from [Si et al., 2022](https://arxiv.org/abs/2109.04027) for rendering tactile images.

<details>

### Features

The TacSL tactile sensor system includes:

#### Sensor Capabilities

- **`VisuoTactileSensor`**: A specialized sensor class that simulates vision-based tactile sensors with elastomer deformation
  - **Camera-based RGB sensing**: Renders realistic tactile images showing surface deformation and contact patterns
  - **Force field sensing**: Computes per-taxel normal and shear forces for contact-rich manipulation
  - **GPU-accelerated rendering**: Leverages GPU for efficient tactile image generation
  - **SDF-based contact detection**: Uses signed distance fields for accurate geometry-elastomer interaction

#### Configuration Options

- **Elastomer Properties**:
  - Configurable tactile array size (rows × columns of taxels)
  - Adjustable tactile margin for sensor boundaries
  - Compliant contact parameters (stiffness, damping)

- **Physics Parameters**:
  - Normal contact stiffness: Controls elastomer compression response
  - Tangential stiffness: Models lateral resistance to sliding
  - Friction coefficient: Defines surface friction properties

- **Visualization & Debug**:
  - Trimesh visualization of tactile contact points
  - SDF closest point visualization
  - Debug rendering of sensor point cloud

### Quick Start

#### Creating a Tactile Sensor

```python
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from isaaclab_assets.sensors import GELSIGHT_R15_CFG

# Define tactile sensor configuration
tactile_sensor_cfg = VisuoTactileSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/elastomer/tactile_sensor",
    history_length=0,
    debug_vis=False,

    # Sensor rendering configuration
    render_cfg=GELSIGHT_R15_CFG,  # Use GelSight R15 sensor parameters

    # Enable RGB and/or force field sensing
    enable_camera_tactile=True,    # RGB tactile images
    enable_force_field=True,        # Force field data

    # Elastomer configuration
    tactile_array_size=(20, 25),   # 20×25 taxel array
    tactile_margin=0.003,           # 3mm sensor margin

    # Contact object configuration
    contact_object_prim_path_expr="{ENV_REGEX_NS}/contact_object",

    # Force field physics parameters
    normal_contact_stiffness=1.0,   # Normal stiffness (N/mm)
    friction_coefficient=2.0,        # Surface friction
    tangential_stiffness=0.1,        # Tangential stiffness

    # Camera configuration (dimensions must match GELSIGHT_R15_CFG which provides the render_cfg)
    camera_cfg=CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/elastomer_tip/cam",
        height=GELSIGHT_R15_CFG.image_height,
        width=GELSIGHT_R15_CFG.image_width,
        data_types=["distance_to_image_plane"],
        spawn=None,  # Camera already exists in USD
    ),
)
```

#### Setting Up the Robot Asset with Compliant Contact

```python
from isaaclab.assets import ArticulationCfg

robot_cfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileWithCompliantContactCfg(
        usd_path="path/to/gelsight_finger.usd",

        # Compliant contact parameters for elastomer
        compliant_contact_stiffness=100.0,    # Elastomer stiffness
        compliant_contact_damping=10.0,       # Elastomer damping
        physics_material_prim_path="elastomer",  # Prim with compliant contact

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=-0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
    ),
    actuators={},
)
```

#### Accessing Tactile Data

```python
# In your simulation loop
scene.update(sim_dt)

# Access tactile sensor data
tactile_data = scene["tactile_sensor"].data

# RGB tactile image (if enabled)
if tactile_data.tactile_rgb_image is not None:
    rgb_images = tactile_data.tactile_rgb_image  # Shape: (num_envs, height, width, 3)

# Force field data (if enabled)
if tactile_data.tactile_normal_force is not None:
    normal_forces = tactile_data.tactile_normal_force  # Shape: (num_envs * rows * cols,)
    shear_forces = tactile_data.tactile_shear_force    # Shape: (num_envs * rows * cols, 2)

    # Reshape to tactile array dimensions
    num_envs = scene.num_envs
    rows, cols = scene["tactile_sensor"].cfg.tactile_array_size
    normal_forces = normal_forces.view(num_envs, rows, cols)
    shear_forces = shear_forces.view(num_envs, rows, cols, 2)
```

### Key Concepts

#### Sensor Modalities

The TacSL sensor supports two complementary sensing modalities:

1. **Camera-Based RGB Sensing** (`enable_camera_tactile=True`):
   - Uses depth information from a camera inside the elastomer
   - Renders realistic tactile images showing contact patterns and deformation
   - Employs the Taxim rendering model for physically-based appearance
   - Outputs RGB images that mimic real GelSight/DIGIT sensors

2. **Force Field Sensing** (`enable_force_field=True`):
   - Computes forces at each taxel (tactile element) in the array
   - Provides normal forces (compression) and shear forces (tangential)
   - Uses SDF-based contact detection with contact objects
   - Enables direct force-based manipulation strategies

#### Compliant Contact Model

The sensor uses PhysX compliant contact for realistic elastomer deformation:

- **Compliant Contact Stiffness**: Controls how much the elastomer compresses under load (higher = stiffer)
- **Compliant Contact Damping**: Controls energy dissipation during contact (affects bounce/settling)
- **Physics Material**: Specified prim (e.g., "elastomer") that has compliant contact enabled

This allows the elastomer surface to deform realistically when contacting objects, which is essential for accurate tactile sensing.

#### Tactile Array Configuration

The sensor discretizes the elastomer surface into a grid of taxels:

```
tactile_array_size = (rows, cols)  # e.g., (20, 25) = 500 taxels
```

- Each taxel corresponds to a point on the elastomer surface
- Forces are computed per-taxel for force field sensing
- The tactile_margin parameter defines the boundary region to exclude from sensing
- Higher resolution (more taxels) provides finer spatial detail but increases computation

#### SDF-Based Contact Detection

For force field sensing, the sensor uses Signed Distance Fields (SDFs):

- Contact objects must have SDF collision meshes
- SDF provides distance and gradient information for force computation
- **Note**: Simple shape primitives (cubes, spheres spawned via `CuboidCfg`) cannot generate SDFs
- Use USD mesh assets for contact objects when force field sensing is required

#### Sensor Rendering Pipeline

The RGB tactile rendering follows this pipeline:

1. **Initial Render**: Captures the reference state (no contact)
2. **Depth Capture**: Camera measures depth to elastomer surface during contact
3. **Deformation Computation**: Compares current depth to reference depth
4. **Taxim Rendering**: Generates RGB image based on deformation field
5. **Output**: Realistic tactile image showing contact geometry and patterns

#### Physics Simulation Parameters

For accurate tactile sensing, configure PhysX parameters:

```python
from isaaclab_physx.physics import PhysxCfg

sim_cfg = sim_utils.SimulationCfg(
    dt=0.005,  # 5ms timestep for stable contact simulation
    physics=PhysxCfg(
        gpu_collision_stack_size=2**30,  # Increase for contact-rich scenarios
    ),
)
```

Also ensure high solver iteration counts for the robot:

```python
solver_position_iteration_count=12  # Higher = more accurate contact resolution
solver_velocity_iteration_count=1
```

### Performance Considerations

- **GPU Acceleration**: Tactile rendering is GPU-accelerated for efficiency
- **Multiple Sensors**: Can simulate multiple tactile sensors across parallel environments
- **Timing Analysis**: Use `sensor.get_timing_summary()` to profile rendering performance
- **SDF Computation**: Initial SDF generation may take time for complex meshes

</details>

### Demo Script

A complete demonstration of TacSL tactile sensor is available:

```bash
# Run TacSL tactile sensor demo with RGB and force field sensing
uv run python scripts/demos/sensors/tacsl_sensor.py \
    --use_tactile_rgb \
    --use_tactile_ff \
    --num_envs 16 \
    --contact_object_type nut

# Save visualization data
uv run python scripts/demos/sensors/tacsl_sensor.py \
    --use_tactile_rgb \
    --use_tactile_ff \
    --save_viz \
    --save_viz_dir tactile_output
```

---

## Trail Library (Detailed)

Each terrain patch comprises four components: a starting platform, a final platform, a connecting trail segment, and distinct skill elements. The resulting terrain is represented as a single mesh using the trimesh library, which models geometry as triangles and stores visual information within the RGB channels.

A trail segment is generated by first defining a cross-section polygon with counter-clockwise ordered vertices. This polygon is then swept along a sweeping path, a sequence of knot-points (x, y, z and yaw) connecting the two platforms. To form a simple ground floor around the trail, the vertices on the left and right borders are expanded outward. Each trail segment may contain one or more skill elements of a unique type, such as ramps, skinnies, roots, rocks, or waves. Each element is parameterized by a uniform distribution over its dimensions.

The generated terrains can easily be embedded into the existing Terrain Generator. The parameterization of the skill elements scales linearly (other maps may be defined as well) with the terrain level, allowing to employ a classical terrain curriculum.

The terrain mesh is systematically randomized by overlaying the vertices with a sequence of geometric disturbances. Additionally, decorative elements such as trees, roots, or stones can be procedurally placed adjacent to the trail. These perturbations break predictable structural patterns, preventing overfitting to specific procedural patterns.

Semantic information is explicitly stored in the vertex colors. The exact RGB values can be chosen arbitrarily, provided the mapping from RGB to semantic label remains bijective. The semantic information can be easily retrieved by a raycaster, which maps the RGB color back into semantic labels. This allows to shape the reward landscape using local semantics surrounding the agent.

<details>

### Module Contents

- `trail_terrains.py`
  Core mesh-generation routines. Main entry point:
  `mesh_trail_segment(difficulty, cfg)`, which returns a generated mesh and its origin.

- `trail_cfg.py`
  Ready-to-use terrain presets (for example `WavesCfg`, `JumpsCfg`, `StonesCfg`,
  `RootsCfg`, `RampsCfg`). Presets extend `TrailBaseCfg` and define profile,
  sweep, and object-parameter ranges.

- `elements/`
  Reusable low-level building blocks:
  - `object_profiles.py`: 2D object profiles for skill elements (waves, ramps, roots, etc.)
  - `sweeping_paths.py`: sweep-path functions that turn 2D profiles into 3D geometry, used to generate the trail and the skill elements.
  - `trail_profiles.py`: trail cross-section profiles
  - `trail_walls.py`: parametrization of the left and right trail-wall
  - `decoration_functions.py`: decorative-object generation used to place trees, roots, rocks, etc in the local surrounding of the trial
  - `terrain_functions.py`: terrain disturbance functions (sine patterns, noise, etc.)
  - `roll_functions.py`: trail roll angle (absolute relative to heading axis) functions

- `utils/`
  Shared utilities used across the terrain generator:
  - `colors.py`: color palettes and color helpers
  - `numpy_arrays.py`: helpers functions based on numpy parametrization
  - `transformations.py`: pose and transform helpers
  - `trimesh_utils.py`: mesh cleaning and repair helpers
  - `math.py`: math helper functions
- `examples/`
  Contains example code:
  - `trails.py`: trail terrain configuration used in the publication, ready for plug-and-play

Decoration functions extract the `decoration_elements.zip` archive and build `.glb` assets within the system temporary directory (under `<tmp_dir>/isaaclab_contrib/trail/`). Depending on the setup, there might exist two `.glb` files: one detailed for deployment/rendering and one simplified for training. If any decoration objects or pre-processing functions are changed, all `.glb` files in the temporary directory need to be manually deleted to force a recomputation.

### Available Terrains

The trail library provides these terrain configuration classes in
`trail_cfg.py`:

- `WavesCfg`: Repeating waves as can be found on pump tracks.
- `JumpsCfg`: Jump features with optional gaps/plateaus, which can be found on mountain-bike parks.
- `StonesCfg`: Rock/grave patches, modeling extended rock beds.
- `RootsCfg`: Root-like patches, modeling forest paths with root networks.
- `RampsCfg`: Single ramp obstacles.
- `MultipleRampsCfg`: Parallel multi-lane ramp sets (this is a typical feature of the Allmend trail in Zurich).
- `SinusoidalCurvesCfg`: S-shaped sinusoidal trail curves.
- `DropCurvesCfg`: Vertical sinusoidal drops/pits.
- `WingCurvesCfg`: Circular wing-style turns.
- `SkinnyCfg`: Narrow beam-style sections.
- `Loops360Cfg`: 360-degree loop structures.
- `StairsCfg`: Stair-step terrain sections as can be found on hiking paths.
- `SlalomCfg`: Pole/slalom obstacle course.

New terrains can be added following the concept implied by the implementation.

### Key Concepts

- `mesh_trail_segment(difficulty, cfg)` -- The core generation function. `difficulty` is a float in `[0.0, 1.0]` that linearly interpolates all curriculum parameters between their `cp0` (easy) and `cp1` (hard) values. It is driven by Isaac Lab's terrain curriculum, which increases `difficulty`
as the agent advances and builds up its skills.

- `TrailBaseCfg` and presets -- All terrain configs extend `TrailBaseCfg` (which itself extends Isaac Lab's `SubTerrainBaseCfg`). The `TrailBaseCfg` class controls surroundings, objects, trail shape, roll angles, decorations, and coloring. Ready-made presets (`WavesCfg`, `JumpsCfg`,
`StonesCfg`, `RootsCfg`, `RampsCfg`, `SlalomCfg`) extend `TrailBaseCfg` with
defaults and expose only the parameters that distinguish each terrain type.

- Control points `cp0` /  `cp1` -- These are `ObjectParameters` instances that define object dimensions at the **easy** and **hard** ends of the curriculum, respectively. Each field may be a fixed value or a `(min, max)` tuple from which a value is sampled per terrain patch:

  | Field | Description |
  |-------|-------------|
  | `length` | Object length along the trail heading direction [m] |
  | `width` | Trail width at this curriculum point [m] |
  | `params` | Dict of additional object-specific parameters (e.g. `height`) |

  By default, at generation time, the interpolated value is computed using the relation `cp0 + difficulty * (cp1 - cp0)`.

#### Semantic color regions

Vertex colors encode region type. The default color assignments are:

| Region | Color | Default |
|--------|-------|---------|
| `col_trail` | Trail surface | Brown (HSV) |
| `col_trail_object` | Objects on trail | Brown (HSV) |
| `col_floor` | Floor beside trail | Green (HSV) |
| `col_start` | Starting platform | Yellow (RGB `1,1,0`) |
| `col_goal` | Ending platform | Magenta (RGB `1,0,1`) |

Override any of these with a `ColorParameters(hsv={...})` or `ColorParameters(hsv={"r":…,"g":…,"b":…})` instance.

### Example Usage

Trail terrain generation integrates directly with the Isaac Lab terrain generator.
Below is an example definition of a terrain-generator config class. `cp0` sets object
dimensions at difficulty 0 (**easiest**) and `cp1` at difficulty 1 (**hardest**). It is not possible to set a difficulty larger than 1 as, depending on the trail objects,m this might cause division by zero.

```py
from isaaclab_contrib.terrains.trail.trail_cfg import ObjectParameters as OP
from isaaclab_contrib.terrains.trail.trail_cfg import StonesCfg

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass

@configclass
class TrailTerrainGeneratorCfg(TerrainGeneratorCfg):
    curriculum = True
    size = (50.0, 40.0)  # length, width
    border_width = 0.0  # set this to zero as it is not used
    border_height: float = 0.0  # set this to zero as it is not used
    num_rows = 10  # terrain difficulty levels
    num_cols = 100  # terrain variants per level
    difficulty_range = (0.0, 1.0)  # 0.0 easiest, 1.0 hardest
    use_cache = False  # set True to reuse generated meshes across runs (safe once cfg is stable)

TRAIL_CFG = TrailTerrainGeneratorCfg(
    sub_terrains={
        "stones": StonesCfg(
            proportion=1.0,
            length_between_objects=(0.0, 0.2),
            length_between_platform_and_object=(2.0, 4.0),
            cp0=OP(length=(0.2, 0.4), width=2.0, params={"height": 0.0}),
            cp1=OP(length=(0.4, 0.8), width=(0.9, 1.5), params={"height": (0.01, 0.1)}),
        )
    },
)
```

A manger based RL environment is set up in isaaclab_tasks/contib/trail and can be run with `./isaaclab.sh -p scripts/environments/zero_agent.py --task IsaacContrib-Velocity-Trail-AnymalC --num_envs 4`

### Semantic Embedding

The colors stored in `visual.vertex_colors` can be visualized by setting `visual_material=None` in the `TerrainImporterCfg`. Those colors can also be read directly with a color-sensitive ray caster. In this case, MDP components can be designed to digest semantic information extracted directly from the mesh. For instance, by coloring the initial platform, the final platform, the trail, and the trail objects uniquely, it is possible to
- span the agent at the initial platform,
- reward agents that follow the trail,
- upgrade agents to the next level that reach the final platform,
- penalize/reward agents that touch/overcome certain objects or reach certain checkpoints, and
- design a terrain curriculum that upgrades the agents upon reaching the final platform.

As an example, a slalom-like environment may be created using the following terrain config
```py
sub_terrains["slalom"] = SlalomCfg(
    proportion=1.0,
    length_between_objects=(1.0, 3.5),
    length_between_platform_and_object=(1.0, 3.0),
    cp0=OP(length=0.4, width=3.5, params={"height": 1.0, "rel_dist_from_center": (1.0, 1.0)}),
    cp1=OP(
        length=(0.05, 0.05), width=(1.5, 2.0), params={"height": (0.8, 1.0), "rel_dist_from_center": (0.0, 1.0)}
    ),
    col_trail_object=ColorParameters(hsv={"r": 0.0, "g": 0.0, "b": 1.0}),
    col_trail=ColorParameters(hsv={"r": 1.0, "g": 0.0, "b": 0.0}),
    col_floor=ColorParameters(hsv={"r": 0.0, "g": 0.0, "b": 0.0}),
    col_start=ColorParameters(hsv={"r": 1.0, "g": 1.0, "b": 0.0}),
    col_goal=ColorParameters(hsv={"r": 1.0, "g": 0.0, "b": 1.0}),
)
```

### Quick Start

Attach the trail terrain generator to a scene of manager-based RL environment as follows
```py
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.terrains.trail.examples.trails import TRAIL_CFG  # import the example trail config class

@configclass
class TrailSceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TRAIL_CFG,
        visual_material=None,  # set to None if you want to see the RGB colors defined in the mesh
        debug_vis=False,
    )
```

</details>

## Testing

The extension includes comprehensive unit tests for all contributed components:

```bash
# Test multirotor components
python -m pytest source/isaaclab_contrib/test/assets/test_multirotor.py
python -m pytest source/isaaclab_contrib/test/actuators/test_thruster.py
python -m pytest source/isaaclab_contrib/test/assets/test_drone_geometric_controllers.py

# Test trail components
python -m pytest source/isaaclab_contrib/test/terrains/test_trail.py

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
