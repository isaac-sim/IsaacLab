# Isaac Lab Teleop

`isaaclab_teleop` integrates the [IsaacTeleop](https://github.com/isaac-sim/IsaacTeleop) retargeting
framework with Isaac Lab, providing a single teleoperation device class that manages OpenXR sessions,
XR anchor synchronization, retargeting pipelines, and action-tensor generation.

## Key Features

- **`IsaacTeleopDevice`** -- unified device that wraps an IsaacTeleop `TeleopSession` behind a
  context-manager interface. Returns a flat `torch.Tensor` action each frame.
- **`IsaacTeleopCfg` / `XrCfg`** -- declarative configuration for retargeting pipelines, XR anchor
  placement, rotation modes, and tuning UI.
- **Deferred session creation** -- if OpenXR handles are not yet available (e.g. the user has not
  clicked *Start AR*), the session is created transparently on the first `advance()` call once
  handles appear.
- **Teleop commands** -- register callbacks for `START`, `STOP`, and `RESET` commands dispatched via
  XR controller buttons or the Carbonite message bus.
- **XR anchor rotation modes** -- `FIXED`, `FOLLOW_PRIM`, `FOLLOW_PRIM_SMOOTHED`, and `CUSTOM`
  modes for controlling how the anchor orientation tracks the reference prim.
- **Retargeting tuning UI** -- optional ImGui window for real-time adjustment of retargeter
  parameters when `retargeters_to_tune` is provided.

## Architecture

`IsaacTeleopDevice` composes three focused collaborators:

| Component | Responsibility |
|---|---|
| `XrAnchorManager` | XR anchor prim setup, dynamic/static synchronization, coordinate-frame transform computation |
| `TeleopSessionLifecycle` | Pipeline building, OpenXR handle acquisition, session create/destroy, action-tensor extraction |
| `CommandHandler` | Callback registration and XR message-bus command dispatch |

## Usage

### 1. Configure Your Environment

Add an `isaac_teleop` attribute to your environment config:

```python
from isaaclab_teleop import IsaacTeleopCfg, XrCfg

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        pipeline, retargeters = my_pipeline_builder()
        self.isaac_teleop = IsaacTeleopCfg(
            xr_cfg=XrCfg(
                anchor_pos=(0.5, 0.0, 0.5),
                anchor_prim_path="{ENV_REGEX_NS}/Robot/base_link",
            ),
            pipeline_builder=lambda: pipeline,
            retargeters_to_tune=lambda: retargeters,
        )
```

> Both `pipeline_builder` and `retargeters_to_tune` must be **callables** (lambdas or functions)
> because `@configclass` deep-copies mutable attributes and retargeter objects often contain
> non-picklable handles.

### 2. Define a Pipeline Builder

Create a function that builds your IsaacTeleop retargeting pipeline. The builder should return an
`OutputCombiner` with an `"action"` key containing the flattened action tensor (typically via
`TensorReorderer`). Optionally return a list of retargeters to expose in the tuning UI:

```python
from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaacteleop.retargeting_engine.retargeters import (
    GripperRetargeter, Se3AbsRetargeter, TensorReorderer,
)
from isaacteleop.retargeting_engine.interface import OutputCombiner

def my_pipeline_builder():
    controllers = ControllersSource(name="controllers")
    se3 = Se3AbsRetargeter(cfg, name="ee_pose")
    # ... connect retargeters and flatten with TensorReorderer ...
    pipeline = OutputCombiner({"action": reorderer.output("output")})
    return pipeline, [se3]
```

### 3. Run Teleoperation

The existing teleop scripts automatically detect `isaac_teleop` in the environment config:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task My-IsaacTeleop-Env-v0
```

### 4. Programmatic Usage

`IsaacTeleopDevice` supports Python's context-manager protocol:

```python
from isaaclab_teleop import IsaacTeleopCfg, IsaacTeleopDevice

cfg = IsaacTeleopCfg(pipeline_builder=my_pipeline_builder)

with IsaacTeleopDevice(cfg) as device:
    device.add_callback("RESET", env.reset)
    while running:
        action = device.advance()
        if action is not None:
            env.step(action.repeat(num_envs, 1))
```

`advance()` returns `None` while waiting for the OpenXR session, so callers can continue
rendering without blocking.

## Configuration Reference

### `IsaacTeleopCfg`

| Field | Type | Default | Description |
|---|---|---|---|
| `xr_cfg` | `XrCfg` | `XrCfg()` | XR anchor position, rotation, and dynamic-anchoring settings |
| `pipeline_builder` | `Callable[[], OutputCombiner]` | *required* | Builds the retargeting pipeline |
| `retargeters_to_tune` | `Callable[[], list[BaseRetargeter]] \| None` | `None` | Retargeters to expose in the tuning UI |
| `plugins` | `list[PluginConfig]` | `[]` | IsaacTeleop plugin configurations |
| `sim_device` | `str` | `"cuda:0"` | Torch device for output action tensors |
| `teleoperation_active_default` | `bool` | `False` | Whether teleoperation is active on session start |
| `app_name` | `str` | `"IsaacLabTeleop"` | Application name for the IsaacTeleop session |

### `XrCfg`

| Field | Type | Default | Description |
|---|---|---|---|
| `anchor_pos` | `tuple[float, float, float]` | `(0, 0, 0)` | XR anchor position in world frame |
| `anchor_rot` | `tuple[float, float, float, float]` | `(0, 0, 0, 1)` | XR anchor rotation (quaternion xyzw) |
| `anchor_prim_path` | `str \| None` | `None` | Prim to attach anchor to for dynamic positioning |
| `anchor_rotation_mode` | `XrAnchorRotationMode` | `FIXED` | How anchor rotation tracks the reference prim |
| `anchor_rotation_smoothing_time` | `float` | `1.0` | Slerp time constant (seconds) for `FOLLOW_PRIM_SMOOTHED` mode |
| `anchor_rotation_custom_func` | `Callable` | identity | Custom rotation function for `CUSTOM` mode |
| `near_plane` | `float` | `0.15` | Near clipping plane distance for the XR device |
| `fixed_anchor_height` | `bool` | `True` | Fix anchor height to initial value of the reference prim |

## Utilities

- **`remove_camera_configs(env_cfg)`** -- strips camera sensors and their associated observation
  terms from an environment config. XR does not support additional cameras as they cause rendering
  conflicts.

## Dependencies

- **`isaaclab`** -- core Isaac Lab framework
- **`isaacteleop`** -- IsaacTeleop retargeting engine, device I/O, and session management
- **`isaacsim`** -- Isaac Sim runtime (provides the Kit XR bridge for OpenXR handle acquisition)
