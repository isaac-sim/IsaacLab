# Isaac Lab Teleop

`isaaclab_teleop` integrates the [Isaac Capture](https://github.com/NVIDIA/IsaacCapture) retargeting
framework with Isaac Lab, providing a single teleoperation device class that manages OpenXR sessions,
XR anchor synchronization, retargeting pipelines, and action-tensor generation.

## Key Features

- **`IsaacTeleopDevice`** -- unified device that wraps an Isaac Capture `TeleopSession` behind a
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
- **XR camera feedback** -- selected scene cameras use Kit Scene UI or the optional
  [OVXR path](OVXR_RUNTIME.md)'s OpenXR quad layers. The latter also
  supports a passive desktop eye-image mirror; separate quad layers are not in that image.

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
from isaaclab_teleop import (
    IsaacTeleopCfg,
    XrCameraFeedCfg,
    XrCfg,
)

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
            xr_camera_feeds=[
                XrCameraFeedCfg(camera_name="robot_pov_cam"),
            ],
        )
```

> Both `pipeline_builder` and `retargeters_to_tune` must be **callables** (lambdas or functions)
> because `@configclass` deep-copies mutable attributes and retargeter objects often contain
> non-picklable handles.

### 2. Define a Pipeline Builder

Create a function that builds your Isaac Capture retargeting pipeline. The builder should return an
`OutputCombiner` with an `"action"` key containing the flattened action tensor (typically via
`TensorReorderer`). A setup factory may also return retargeters for the tuning UI;
unpack those before assigning the two configuration callbacks:

```python
from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaacteleop.retargeters import (
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
uv run python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task My-Isaac-Capture-Env-v0 --xr
```

For kitless teleoperation, installation, MCAP replay, camera panels and the
optional desktop mirror, see [Kitless XR teleoperation](OVXR_RUNTIME.md).

See [XR Camera Feedback](../../../docs/source/features/isaac_teleop.rst#xr-camera-feedback) for
reference tasks, camera selection, layout, placement, renderer, disable, and OVXR behavior.

### 4. Programmatic Usage

`IsaacTeleopDevice` supports Python's context-manager protocol:

```python
from isaaclab_teleop import IsaacTeleopCfg, IsaacTeleopDevice

pipeline, retargeters = my_pipeline_builder()
cfg = IsaacTeleopCfg(
    pipeline_builder=lambda: pipeline,
    retargeters_to_tune=lambda: retargeters,
)

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
| `xr_camera_feeds` | `list[XrCameraFeedCfg]` | `[]` | Existing task scene cameras shown as image panels |
| `xr_camera_feed_layout` | `XrCameraFeedLayoutCfg` | viewer-start manual layout | Placement reference and ordered multi-feed packing |
| `pipeline_builder` | `Callable[[], OutputCombiner]` | *required* | Builds the retargeting pipeline |
| `retargeters_to_tune` | `Callable[[], list[BaseRetargeter]] \| None` | `None` | Retargeters to expose in the tuning UI |
| `plugins` | `list[PluginConfig]` | `[]` | Isaac Capture plugin configurations |
| `sim_device` | `str` | `"cuda:0"` | Torch device for output action tensors |
| `retargeting_execution` | `RetargetingExecutionConfig \| None` | `None` (resolved at session start to `mode="pipelined", pacing=DeadlinePacingConfig(safety_margin_s=0.025)`) | Isaac Capture retargeting execution settings; deferred so the config imports without `isaacteleop` |
| `teleoperation_active_default` | `bool` | `False` | Whether teleoperation is active on session start |
| `require_initial_bimanual_hand_tracking` | `bool` | `False` | Wait for initial hand presence before stepping |
| `action_conditioner` | `Callable \| None` | `None` | Task-owned stateful pre-step action conditioning |
| `action_validator` | `Callable \| None` | `None` | Task-owned fail-closed action validation |
| `app_name` | `str` | `"IsaacLabTeleop"` | Application name for the IsaacTeleop session |

The 25 ms `DeadlinePacingConfig` safety margin staggers Isaac Capture's Python work behind Isaac Lab's
step Python, giving native work such as rendering time to overlap instead of having both Python stacks
contend for the GIL at the start of the step.

### `XrCfg`

| Field | Type | Default | Description |
|---|---|---|---|
| `anchor_pos` | `tuple[float, float, float]` | `(0, 0, 0)` | XR anchor position in world frame |
| `anchor_rot` | `tuple[float, float, float, float]` | `(0, 0, 0, 1)` | XR anchor rotation (quaternion xyzw) |
| `render_anchor_pos` | `tuple[float, float, float] \| None` | `None` | Optional render-only anchor position |
| `render_anchor_rot` | `tuple[float, float, float, float] \| None` | `None` | Optional render-only anchor rotation |
| `hidden_visual_prims` | `list[str]` | `[]` | Prim-path regexes hidden before OVXR session materialization |
| `excluded_visual_meshes` | `list[str]` | `[]` | Visual-mesh regexes assigned to a separate render-export partition, without changing source USD visibility |
| `anchor_prim_path` | `str \| None` | `None` | Reference prim for dynamic positioning; OVXR reads committed physics poses and shares the same anchor policy |
| `anchor_rotation_mode` | `XrAnchorRotationMode` | `FIXED` | How anchor rotation tracks the reference prim |
| `anchor_rotation_smoothing_time` | `float` | `1.0` | Slerp time constant (seconds) for `FOLLOW_PRIM_SMOOTHED` mode |
| `anchor_rotation_custom_func` | `Callable` | identity | Custom rotation function for `CUSTOM` mode |
| `near_plane` | `float` | `0.15` | Near clipping plane distance for the XR device |
| `fixed_anchor_height` | `bool` | `True` | Fix anchor height to initial value of the reference prim |

For single-environment OVXR, prefer `excluded_visual_meshes` for a robot
head surrounding the eyes. Match only visual mesh paths, not head links or
collision geometry. Environment cameras select `env_0`; excluded meshes use
`xr_excluded`, with `showAllPartitionsByDefault=false`. This also excludes the
head from PiP cameras selecting `env_0`; other camera views require an explicit
partition policy. The source Stage's visibility and instancing are restored after
render export. GR1T2 and fixed-base G1 use this policy instead of global hiding.

## Capture and replay boundaries

MCAP records DeviceIO tracker/control input, not historical simulation scenes.
Environment/control reset does not rewind MCAP input. HDF5
demonstration scripts use the Kit/Isaac Sim application path; these OVXR
commands do not qualify HDF5 recording/replay. PiP and mirror consume latest
completed images rather than a history of simulation scenes; they are not lossless
recorders or rolling-shutter playback.

## Utilities

- **`remove_camera_configs(env_cfg)`** -- strips camera sensors and their associated policy
  observation terms.

## Run with Docker

Teleoperation with Isaac Lab runs in a **single container**. Build the image yourself and run a single container. **Do not use Docker Compose** for this workflow (no multi-container setup). Everything runs inside one container with Isaac Lab.

Inside the container: install Isaac Capture once (`uv pip install 'isaacteleop[retargeters,cloudxr]~=1.0.0' --extra-index-url https://pypi.nvidia.com`), then start the CloudXR runtime with `--accept-eula` so there is no interactive EULA prompt, and run your teleop script. Example:

```bash
uv run python -m isaacteleop.cloudxr --accept-eula &
source ~/.cloudxr/run/cloudxr.env
uv run python scripts/tools/record_demos.py --task IsaacContrib-PickPlace-Locomanipulation-G1-Abs --num_demos 5 --dataset_file ./datasets/dataset.hdf5 --xr --visualizer kit
```

In the Isaac Sim UI, set the AR panel to **System OpenXR Runtime** and click **Start XR**. For the full flow and options, see the [CloudXR teleoperation how-to](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html) and [Isaac Capture Quick Start](https://nvidia.github.io/IsaacCapture/main/getting_started/quick_start.html).

For a fully headless experience, omit `--visualizer kit` when running Docker, and the XR teleop session will
run automatically.

## Dependencies

- **`isaaclab`** -- core Isaac Lab framework
- **`isaaclab_ov`** -- required by the optional OVXR session adapter; the standalone `isaaclab_teleop` package does not install this backend automatically
- **`isaacteleop`** -- IsaacTeleop retargeting engine, device I/O, and session management
- **`isaacsim`** -- required for the Kit XR bridge, not for the OVXR session path
- **OVXR / OVRTX / OVStage** -- matched native runtime and stock peers for OVXR
- **OVUI** -- optional desktop-mirror extra; headset-only operation does not import it
