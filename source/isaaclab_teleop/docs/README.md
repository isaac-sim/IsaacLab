# Isaac Lab Teleop

This extension provides IsaacTeleop-based teleoperation for Isaac Lab environments.

## Overview

`isaaclab_teleop` integrates the IsaacTeleop framework with Isaac Lab, providing:

- **IsaacTeleopDevice**: A standalone device class that manages IsaacTeleop sessions
- **XR Anchor Support**: Reuses Isaac Lab's XR anchor synchronization for dynamic camera positioning
- **Pipeline-based Configuration**: Define retargeting pipelines in environment configs

## Usage

### 1. Configure Your Environment

Add a `isaac_teleop` configuration to your environment config:

```python
from isaaclab_teleop import IsaacTeleopCfg
from isaaclab.devices.openxr.xr_cfg import XrCfg

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
            sim_device="cuda:0",
        )
```

### 2. Define a Pipeline Builder

Create a function that builds your IsaacTeleop retargeting pipeline.
The builder can return just an `OutputCombiner`, or a tuple of
`(OutputCombiner, list[BaseRetargeter])` when you also want to
expose retargeters in the tuning UI:

```python
from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
from isaacteleop.retargeting_engine.retargeters import (
    GripperRetargeter, Se3AbsRetargeter, TensorReorderer
)
from isaacteleop.retargeting_engine.interface import OutputCombiner

def my_pipeline_builder():
    controllers = ControllersSource(name="controllers")
    # ... configure retargeters ...
    # ... use TensorReorderer to flatten outputs ...
    pipeline = OutputCombiner({"action": reorderer.output("output")})
    return pipeline, [se3_retargeter]  # retargeters for tuning UI
```

### 3. Run Teleoperation

The existing teleop scripts automatically detect `isaac_teleop` in the env config:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
    --task My-IsaacTeleop-Env-v0
```

## Dependencies

- `isaaclab`: Core Isaac Lab framework
- `isaacteleop`: IsaacTeleop framework for device tracking and retargeting
