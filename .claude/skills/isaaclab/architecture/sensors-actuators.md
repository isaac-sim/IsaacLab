# Sensors, Actuators & Visualizers

## Actuator Types

### Implicit (Physics Engine Handled)
| Type | Use Case | Config |
|------|----------|--------|
| `ImplicitActuatorCfg` | Simple PD/effort control (default) | `stiffness`, `damping`, `effort_limit_sim` |

### Explicit (Computationally Modeled)
| Type | Use Case | Config |
|------|----------|--------|
| `IdealPDActuatorCfg` | Explicit PD with discrete-time control | `stiffness`, `damping` |
| `DCMotorCfg` | DC motor with saturation limits | Adds `saturation_effort` |
| `DelayedPDActuatorCfg` | PD with command delay | `min_delay`, `max_delay` (physics steps) |
| `RemotizedPDActuatorCfg` | Angle-dependent torque limits | Joint parameter lookup table |
| `ActuatorNetMLPCfg` | Neural network (MLP) actuator | MLP model path |
| `ActuatorNetLSTMCfg` | Neural network (LSTM) actuator | LSTM model path |

**Recommendation**: Start with `ImplicitActuatorCfg`. Use `DCMotor` for more realistic
dynamics. Neural net actuators are for advanced sim-to-real transfer.

```python
actuators={
    "my_motor": ImplicitActuatorCfg(
        joint_names_expr=["joint_.*"],  # Regex for joint selection
        effort_limit_sim=400.0,         # Max force (N) or torque (Nm)
        stiffness=0.0,                  # 0 = pure effort, >0 = position spring
        damping=10.0,                   # Friction/damping coefficient
    ),
}
```

**Control modes by stiffness/damping:**
- `stiffness=0, damping=0`: Pure effort (force/torque) control
- `stiffness>0, damping>0`: PD position control
- `stiffness=0, damping>0`: Velocity-like control with damping

## Action Types

| Config | Control Mode |
|--------|-------------|
| `JointEffortActionCfg` | Direct force/torque on joints |
| `JointPositionActionCfg` | Target joint positions (PD control) |
| `JointVelocityActionCfg` | Target joint velocities |
| `DifferentialInverseKinematicsActionCfg` | Cartesian end-effector control |

## Camera Sensors

**Important**: Camera/sensor config classes can only be deeply imported after `AppLauncher`
starts the Kit runtime. Top-level `import isaaclab.sensors` will fail otherwise.

### Two Camera Types

| Type | Config | Use Case |
|------|--------|----------|
| **TiledCamera** | `TiledCameraCfg` | GPU-accelerated batched rendering. Recommended for RL training with many envs. |
| **Camera** | `CameraCfg` | Standard camera with per-camera rendering. Use for single-env or high-quality capture. |

`TiledCameraCfg` inherits from `CameraCfg` (same API) but uses tiled rendering for
GPU-parallel image generation across environments. Always prefer `TiledCameraCfg` for
RL training with visual observations.

### TiledCamera Example (GPU-accelerated, recommended)
```python
from isaaclab.sensors import TiledCameraCfg

camera = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    data_types=["rgb", "depth"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        clipping_range=(0.1, 20.0),
    ),
    width=100, height=100,
)
```

### Available Data Types
- `rgb` - Color image (H, W, 3)
- `rgba` - Color + alpha (H, W, 4)
- `depth` / `distance_to_image_plane` - Depth map (H, W, 1)
- `distance_to_camera` - Ray distance to optical center
- `albedo` - Fast diffuse-albedo only (optimized, no lighting)
- `simple_shading_constant_diffuse` - Constant diffuse shading
- `simple_shading_diffuse_mdl` - MDL diffuse shading
- `simple_shading_full_mdl` - Full MDL shading
- `normals` - Surface normals (H, W, 3)
- `motion_vectors` - Optical flow
- `semantic_segmentation` - Semantic labels
- `instance_segmentation_fast` / `instance_id_segmentation_fast` - Instance IDs

### Camera Offset Convention
```python
CameraCfg.OffsetCfg(
    pos=(0.0, 0.0, 0.0),
    rot=(0.0, 0.0, 0.0, 1.0),       # Quaternion (x, y, z, w)
    convention="ros",                 # "ros" | "opengl" | "world"
)
```
- `"ros"`: forward +Z, up -Y (default)
- `"opengl"`: forward -Z, up +Y (USD/Omniverse convention)
- `"world"`: forward +X, up +Z

## Other Sensors

| Sensor | Config | Data |
|--------|--------|------|
| Contact | `ContactSensorCfg` | Contact forces, normals |
| IMU | `ImuCfg` | Acceleration, angular velocity |
| Frame Transformer | `FrameTransformerCfg` | Relative body transforms |
| Visuotactile | `VisuotactileSensorCfg` | Visual + tactile (community, `isaaclab_contrib`) |

### Ray Caster Variants

| Config | Use Case |
|--------|----------|
| `RayCasterCfg` | Basic ray casting for height maps and terrain distances |
| `RayCasterCameraCfg` | Ray caster configured as a camera (depth from rays) |
| `MultiMeshRayCasterCfg` | Ray casting against multiple mesh targets |
| `MultiMeshRayCasterCameraCfg` | Multi-mesh ray caster as camera |

Ray casters are faster than rendering-based cameras for depth-only observations
(e.g., terrain height maps for legged locomotion).

## Visualizers

| Visualizer | Config | Rendering | Best For |
|-----------|--------|-----------|----------|
| **Kit** | `KitVisualizerCfg` | RTX ray-tracing (GUI viewport) | Interactive development, debugging |
| **Newton** | `NewtonVisualizerCfg` | OpenGL (standalone window) | Headless training, fast preview |
| **Rerun** | `RerunVisualizerCfg` | Browser (WebGL via gRPC) | Remote monitoring, distributed training |

- **Kit**: Default Isaac Sim viewport, 1280x720, configurable dock position
- **Newton**: Independent OpenGL window (no Isaac Sim GUI needed), joint/contact/COM visualization
- **Rerun**: Streams to `localhost:9090` (web), data on gRPC port 9876, optional `.rrd` recording

> **Known Issue (develop branch / Isaac Sim 6.0)**: `--visualizer newton` and
> `--visualizer rerun` both fail with `cannot import name 'DeviceLike' from 'warp'`.
> This is a warp version mismatch — Kit extscache ships warp 1.10.1 which lacks
> `DeviceLike` (added in 1.11). Training continues but visualizer silently degrades.
> Use `--headless` or `--visualizer kit` (GUI only) as a workaround.
>
> **Note**: `--visualizer omniverse` is NOT valid. Use `kit`, `newton`, or `rerun`.

### Rendering Modes
- **Default**: Viewport window (omit `--headless`)
- **Headless**: `--headless` (no rendering, max performance)
- **Livestream**: `--livestream 2` (remote visualization)

### Viewer Configuration
```python
def __post_init__(self):
    self.viewer.eye = (8.0, 0.0, 5.0)       # Camera position
    self.viewer.lookat = (0.0, 0.0, 0.0)    # Camera target
    self.viewer.resolution = (1280, 720)
    self.viewer.origin_type = "world"         # "world"|"env"|"asset_root"|"asset_body"
```

### Viewport Controls
| Action | Control |
|--------|---------|
| Move camera | RMB + WASD |
| Vertical | RMB + Q/E |
| Rotate | RMB + drag |
| Zoom | Scroll or Alt + RMB |
| Pan | MMB + drag |
