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

In Lab 3.0, cameras work across physics backends. Some camera types (e.g., ray casters)
are fully Kit-independent; rendering-based cameras depend on the active renderer.

### Camera Types

| Type | Config | Use Case |
|------|--------|----------|
| **TiledCamera** | `TiledCameraCfg` | GPU-accelerated batched rendering. Recommended for RL training. |
| **Camera** | `CameraCfg` | Standard camera with per-camera rendering. |
| **RayCasterCamera** | `RayCasterCameraCfg` | Depth from ray casting (Kit-independent, faster for depth-only). |

### Camera Offset Convention
```python
CameraCfg.OffsetCfg(
    pos=(0.0, 0.0, 0.0),
    rot=(0.0, 0.0, 0.0, 1.0),       # Quaternion (x, y, z, w)
    convention="ros",                 # "ros" | "opengl" | "world"
)
```

**Note**: Isaac Lab uses `xyzw` quaternion order throughout.

## Other Sensors

| Sensor | Config | Data |
|--------|--------|------|
| Contact | `ContactSensorCfg` | Contact forces, normals |
| IMU | `ImuCfg` | Acceleration, angular velocity |
| Frame Transformer | `FrameTransformerCfg` | Relative body transforms |
| Visuotactile | `VisuotactileSensorCfg` | Visual + tactile (`isaaclab_contrib`) |

### Ray Caster Variants

| Config | Use Case |
|--------|----------|
| `RayCasterCfg` | Basic ray casting for height maps and terrain distances |
| `RayCasterCameraCfg` | Ray caster configured as a camera (depth from rays) |
| `MultiMeshRayCasterCfg` | Ray casting against multiple mesh targets |
| `MultiMeshRayCasterCameraCfg` | Multi-mesh ray caster as camera |

Ray casters are faster than rendering-based cameras for depth-only observations
(e.g., terrain height maps for legged locomotion).

### Sensors and Physics Backends

Sensors use the factory pattern like other Isaac Lab objects. The core sensor interfaces
are in `isaaclab.sensors`; backend-specific implementations are in `isaaclab_physx.sensors`
and `isaaclab_newton.sensors`. Your code imports from `isaaclab.sensors` and the factory
resolves the correct backend at runtime.

## Visualizers

| Visualizer | Config | Description |
|-----------|--------|-------------|
| **Kit** | `KitVisualizerCfg` | Isaac Sim viewport (RTX, requires display) |
| **Newton** | `NewtonVisualizerCfg` | Native Newton physics renderer |
| **Rerun** | `RerunVisualizerCfg` | Browser-based (WebGL via gRPC), remote/distributed |
| **Viser** | `ViserVisualizerCfg` | Web-based viewer (Newton's ViewerViser) |

For details on backends, renderers, and visualizers: [backends.md](backends.md)
