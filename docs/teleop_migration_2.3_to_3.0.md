# Migrating a Custom Teleop Script and G1 Environment from Isaac Lab 2.3 to 3.0

This document is a step-by-step, machine-actionable guide for migrating a custom `record_demos.py`-style teleop recording script and a G1-based teleop environment config from Isaac Lab release/2.3 to Isaac Lab 3.0 with Isaac Teleop.

Each step below describes the exact code to find (using grep patterns), the exact code to delete or replace, and the exact code to insert. An LLM agent or human can follow these steps sequentially against the two target files:

- **File A**: Your custom teleop recording script (based on the 2.3 version of `scripts/tools/record_demos.py`)
- **File B**: Your custom G1 teleop environment config (a `*_env_cfg.py` file with `ManagerBasedRLEnvCfg`)

**Reference implementation** (already migrated, use as ground truth):
- `scripts/tools/record_demos.py` in Isaac Lab 3.0
- `scripts/environments/teleoperation/teleop_se3_agent.py` in Isaac Lab 3.0
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_unitree_g1_inspire_hand_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pick_place/pickplace_gr1t2_env_cfg.py`
- `source/isaaclab_teleop/isaaclab_teleop/isaac_teleop_cfg.py`
- `docs/source/migration/migrating_to_isaaclab_3-0.rst` (official 3.0 migration guide)

---

## Prerequisites

Before starting migration:

- **Isaac Sim 6.0** (required by Isaac Lab 3.0).
- **Python >= 3.12** (required by `isaaclab_teleop`).
- Install the `isaaclab_teleop` extension (bundled with Isaac Lab 3.0, or run `pip install -e source/isaaclab_teleop`). This automatically pulls in `isaacteleop` (the Isaac Teleop Python package from the TeleopCore repository) as a pip dependency -- you do not need to install `isaacteleop` separately.

---

## Part 1: Migrating the Recording Script (File A)

### Step 1: Remove `--enable_pinocchio` argument

**Why:** Pinocchio is now handled internally by the Pink IK controller. Task modules are auto-discovered. The pre-import hack and XR auto-detection via device name are no longer needed.

**Find (grep pattern):** `enable_pinocchio`

**Action 1a -- Delete the argparse argument.** Search for and delete the entire `parser.add_argument` block for `--enable_pinocchio`:

```python
# DELETE THIS ENTIRE BLOCK:
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
```

**Action 1b -- Delete the pre-AppLauncher gating block.** Search for `if args_cli.enable_pinocchio:` before `app_launcher = AppLauncher(...)` and delete the entire block including the handtracking XR check immediately below it:

```python
# DELETE THIS ENTIRE BLOCK:
if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True
```

**Action 1c -- Delete the post-AppLauncher conditional imports.** Search for `if args_cli.enable_pinocchio:` after the `"""Rest everything follows."""` comment and delete:

```python
# DELETE THIS ENTIRE BLOCK:
if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
```

---

### Step 2: Change `--teleop_device` default from `"keyboard"` to `None`

**Why:** When `None`, the script auto-detects: use IsaacTeleop if `env_cfg.isaac_teleop` exists, otherwise fall back to keyboard. Explicitly passing `--teleop_device keyboard` forces the legacy path.

**Find (grep pattern):** `--teleop_device` in the `add_argument` call

**Action:** In the `parser.add_argument("--teleop_device", ...)` call, change `default="keyboard"` to `default=None` and update the help text.

Replace:
```python
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
```

With:
```python
parser.add_argument(
    "--teleop_device",
    type=str,
    default=None,
    help=(
        "Legacy teleop device name. When omitted, the IsaacTeleop pipeline is used if configured in the env,"
        " otherwise keyboard is used as fallback. When explicitly provided, the script uses the legacy"
        " teleop_devices path and looks up this name in env_cfg.teleop_devices.devices."
    ),
)
```

---

### Step 3: Add CloudXR CLI arguments

**Why:** Isaac Lab 3.0 can auto-launch the CloudXR runtime for XR headsets. Two new arguments control this.

**Find (grep pattern):** `AppLauncher.add_app_launcher_args(parser)` -- insert the new arguments immediately *before* this line.

**Action 3a -- Add arguments.** Insert the following two `parser.add_argument` blocks immediately before the `AppLauncher.add_app_launcher_args(parser)` line:

```python
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default="cloudxrjs",
    help=(
        "Path to a CloudXR .env file, or a shorthand: 'cloudxrjs' (Quest/Pico, default) or 'avp' (Apple Vision Pro)."
        " Set to 'none' to disable CloudXR auto-launch entirely."
    ),
)
parser.add_argument(
    "--auto_launch_cloudxr",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Auto-launch the CloudXR runtime when --cloudxr_env is set. Use --no-auto_launch_cloudxr to disable.",
)
```

**Action 3b -- Add the CloudXR env resolver helper.** After the post-AppLauncher imports section (after `"""Rest everything follows."""` and its imports), add:

```python
_CLOUDXR_ENV_SHORTHANDS: dict[str, str] = {}


def _resolve_cloudxr_env(value: str | None) -> str | None:
    """Resolve ``--cloudxr_env`` shorthands to absolute ``.env`` file paths."""
    if value is None or value.strip() == "" or value.lower() == "none":
        return None
    if not _CLOUDXR_ENV_SHORTHANDS:
        from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV

        _CLOUDXR_ENV_SHORTHANDS["cloudxrjs"] = CLOUDXR_JS_ENV
        _CLOUDXR_ENV_SHORTHANDS["avp"] = CLOUDXR_AVP_ENV
    return _CLOUDXR_ENV_SHORTHANDS.get(value.lower(), value)
```

---

### Step 4: Replace `omni.log` with Python `logging`

**Why:** Isaac Lab 3.0 moves to standard Python logging throughout.

**Find (grep pattern):** `import omni.log` and any `omni.log.error` / `omni.log.warn`

**Action 4a -- Replace the import.** Delete `import omni.log`. Add (at the top of the post-AppLauncher imports section):

```python
import logging
```

**Action 4b -- Add a module logger.** After the imports section, add:

```python
logger = logging.getLogger(__name__)
```

**Action 4c -- Replace all call sites.** Apply these substitutions globally across the file:

| Find (exact string) | Replace with |
|---|---|
| `omni.log.error(` | `logger.error(` |
| `omni.log.warn(` | `logger.warning(` |

---

### Step 5: Add IsaacTeleop stack detection

**Why:** The script must decide at startup whether to use the IsaacTeleop pipeline or the legacy teleop_devices path.

**Find (grep pattern):** The function `create_environment_config(` -- this is where detection logic is added.

**Action 5a -- Add detection after `parse_env_cfg`.** Inside `create_environment_config()`, immediately after the `env_cfg = parse_env_cfg(...)` and `env_cfg.env_name = ...` lines, insert:

```python
    # When --teleop_device is explicitly provided, use the legacy teleop_devices path
    # even if isaac_teleop is configured. Otherwise prefer isaac_teleop when available.
    teleop_device_explicitly_set = args_cli.teleop_device is not None
    use_isaac_teleop = (
        not teleop_device_explicitly_set and hasattr(env_cfg, "isaac_teleop") and env_cfg.isaac_teleop is not None
    )
```

**Action 5b -- Update return type.** The function must return a 3-tuple. Change:
```python
    return env_cfg, success_term
```
To:
```python
    return env_cfg, success_term, use_isaac_teleop
```

**Action 5c -- Update the XR camera/DLSS guard.** Find the block:
```python
    if args_cli.xr:
```
Replace with:
```python
    if use_isaac_teleop or args_cli.xr:
```
This ensures camera configs are removed and DLSS is enabled for IsaacTeleop too.

**Action 5d -- Update all callers.** Wherever `create_environment_config(...)` is called, change from:
```python
    env_cfg, success_term = create_environment_config(output_dir, output_file_name)
```
To:
```python
    env_cfg, success_term, use_isaac_teleop = create_environment_config(output_dir, output_file_name)
```

---

### Step 6: Restructure teleop device setup for three-way branch

**Why:** There are now three device paths: (1) IsaacTeleop, (2) legacy `teleop_devices`, (3) keyboard fallback.

**Find (grep pattern):** `def setup_teleop_device(` or the section that creates the teleop device.

**Action 6a -- Add `use_isaac_teleop` parameter.** Update the function signature:
```python
def setup_teleop_device(callbacks: dict[str, Callable], use_isaac_teleop: bool = False) -> object:
```

**Action 6b -- Add a helper for built-in devices.** Add this function before `setup_teleop_device`:

```python
def _create_builtin_device(device_name: str) -> object | None:
    """Create a built-in teleop device by name, or return None if unrecognized."""
    name = device_name.lower()
    if name == "keyboard":
        return Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    elif name == "spacemouse":
        return Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    return None
```

**Action 6c -- Replace the device creation logic inside `setup_teleop_device`.** The old logic was a two-way branch (config lookup vs fallback). Replace the `try:` block body with this three-way branch:

```python
    teleop_device_explicitly_set = args_cli.teleop_device is not None
    teleop_interface = None
    try:
        if use_isaac_teleop:
            from isaaclab_teleop import create_isaac_teleop_device

            teleop_interface = create_isaac_teleop_device(
                env_cfg.isaac_teleop,
                sim_device=args_cli.device,
                callbacks=callbacks,
                cloudxr_env_file=_resolve_cloudxr_env(args_cli.cloudxr_env),
                auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
            )

        elif teleop_device_explicitly_set:
            device_name = args_cli.teleop_device
            if hasattr(env_cfg, "teleop_devices") and device_name in env_cfg.teleop_devices.devices:
                teleop_interface = create_teleop_device(device_name, env_cfg.teleop_devices.devices, callbacks)
            else:
                teleop_interface = _create_builtin_device(device_name)
                if teleop_interface is None:
                    logger.error(
                        f"--teleop_device={device_name} was passed but no matching entry exists in"
                        " env_cfg.teleop_devices and it is not a built-in device name. Either remove"
                        " --teleop_device to use the IsaacTeleop pipeline, or add a"
                        f" '{device_name}' entry under teleop_devices in the environment config."
                        " Built-in devices: keyboard, spacemouse."
                    )
                    exit(1)
                for key, callback in callbacks.items():
                    teleop_interface.add_callback(key, callback)
        else:
            # No --teleop_device and no isaac_teleop: fall back to keyboard
            teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            for key, callback in callbacks.items():
                teleop_interface.add_callback(key, callback)
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        exit(1)
```

**Action 6d -- Update the call site** to pass the new parameter:
```python
    teleop_interface = setup_teleop_device(teleoperation_callbacks, use_isaac_teleop)
```

---

### Step 7: Wrap the simulation loop in a context manager

**Why:** `IsaacTeleopDevice` manages the OpenXR/TeleopCore session lifecycle via a context manager (`with teleop_interface:`). Without it the session never starts.

**Find (grep pattern):** The main `while simulation_app.is_running():` loop.

**Action:** Extract the loop body (including the initial `env.sim.reset()` / `env.reset()` / `teleop_interface.reset()` and the `while` loop) into an `inner_loop()` function, then call it conditionally:

```python
    def inner_loop():
        nonlocal current_recorded_demo_count, success_step_count, should_reset_recording_instance
        nonlocal running_recording_instance, label_text

        env.sim.reset()
        env.reset()
        teleop_interface.reset()

        # ... existing while simulation_app.is_running(): loop body here ...

    if use_isaac_teleop:
        with teleop_interface:
            inner_loop()
    else:
        inner_loop()
```

The `nonlocal` declarations are needed because `inner_loop` modifies these variables that are defined in the enclosing `run_simulation_loop` scope.

---

### Step 8: Add `poll_control_events()` inside the loop

**Why:** IsaacTeleop uses an OpenXR message channel for start/stop/reset commands from the XR client app. `poll_control_events()` reads these and returns a `ControlEvents` object.

**Find (grep pattern):** `action = teleop_interface.advance()` inside the main loop.

**Action 8a -- Import at loop entry.** At the start of `inner_loop()` (after the reset calls), add:

```python
        if use_isaac_teleop:
            from isaaclab_teleop import poll_control_events
```

**Action 8b -- Poll after advance.** Immediately after `action = teleop_interface.advance()`, insert:

```python
                if use_isaac_teleop:
                    ctrl = poll_control_events(teleop_interface)
                    if ctrl.is_active is not None:
                        running_recording_instance = ctrl.is_active
                    if ctrl.should_reset:
                        should_reset_recording_instance = True
```

---

### Step 9: Handle `None` actions from `advance()`

**Why:** `IsaacTeleopDevice.advance()` returns `None` until the XR session starts (e.g. user clicks "Start AR" on their headset). The old code always returned a tensor.

**Find (grep pattern):** `action = teleop_interface.advance()` -- immediately after the `poll_control_events` block from Step 8.

**Action:** Add a `None` guard before the line that expands to batch dimension:

```python
                if action is None:
                    env.sim.render()
                    continue
```

---

### Step 10: Reset the teleop interface on environment reset

**Why:** `IsaacTeleopDevice.reset()` re-anchors the XR headset origin and clears retargeter state. Without this, the robot snaps to a stale pose after reset.

**Find (grep pattern):** `def handle_reset(` or the code block that resets the environment.

**Action 10a -- Add `teleop_interface` parameter to `handle_reset`:**

```python
def handle_reset(
    env, success_step_count, instruction_display, label_text, teleop_interface=None
) -> int:
```

**Action 10b -- After `env.reset()`, add:**

```python
    if teleop_interface is not None and hasattr(teleop_interface, "reset"):
        teleop_interface.reset()
```

**Action 10c -- Update call sites** to pass `teleop_interface`:

```python
                    success_step_count = handle_reset(
                        env, success_step_count, instruction_display, label_text, teleop_interface
                    )
```

---

### Step 11: Default recording to inactive for IsaacTeleop

**Why:** For XR/IsaacTeleop, recording should only start when the user sends a START command from the headset.

**Find (grep pattern):** `running_recording_instance = not args_cli.xr`

**Action:** Replace:
```python
    running_recording_instance = not args_cli.xr
```
With:
```python
    running_recording_instance = not (args_cli.xr or use_isaac_teleop)
```

---

### Step 12: Update the app shutdown sequence

**Why:** `env.close()` closes the USD stage. The viewport needs one more event-loop pump to process the closure before the app exits, otherwise a crash can occur.

**Find (grep pattern):** `simulation_app.close()` at the bottom of the file.

**Action:** Replace:
```python
if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
```
With:
```python
if __name__ == "__main__":
    main()
    simulation_app.update()
    simulation_app.close()
```

---

### Step 13: Reorder `main()` so env config is parsed before rate limiter

**Why:** In 3.0, `use_isaac_teleop` (determined from env config) decides whether to use a rate limiter or XR visualization. So env config must be parsed first.

**Find (grep pattern):** `def main()` -- look at the order of operations.

**Action:** Ensure the order inside `main()` is:

1. `setup_output_directories()`
2. `create_environment_config(...)` -- now returns `(env_cfg, success_term, use_isaac_teleop)`
3. Rate limiter decision based on `use_isaac_teleop`:
   ```python
   if args_cli.xr or use_isaac_teleop:
       rate_limiter = None
       from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization
       XRVisualization.assign_manager(TeleopVisualizationManager)
   else:
       rate_limiter = RateLimiter(args_cli.step_hz)
   ```
4. `create_environment(env_cfg)`
5. `run_simulation_loop(env, None, success_term, rate_limiter, use_isaac_teleop)`

If your 2.3 script had rate limiter setup *before* env config parsing, swap those blocks.

---

## Part 2: Migrating the G1 Environment Config (File B)

### Step 14: Replace teleop-related imports

**Find (grep pattern):** `from isaaclab.devices import` or `from isaaclab.devices.openxr`

**Action:** Delete all of these import lines:
```python
# DELETE any/all of these that appear:
from isaaclab.devices import DevicesCfg, OpenXRDeviceCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.devices.openxr.retargeters import Se3AbsRetargeterCfg, GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters import ...  # any retargeter imports
```

Replace with:
```python
from isaaclab_teleop import IsaacTeleopCfg, XrCfg
```

---

### Step 15: Create a pipeline builder function

**Why:** The old `DevicesCfg` / `OpenXRDeviceCfg` / retargeter-list pattern is replaced by an imperative pipeline builder that constructs an Isaac Teleop retargeting graph.

**Action:** Add a module-level function that builds the retargeting pipeline. The function must return an `OutputCombiner` with a single `"action"` key whose value is the flattened action tensor matching your env's action space.

Use this skeleton and adapt to your robot's action layout:

```python
def _build_my_g1_pipeline():
    """Build the IsaacTeleop retargeting pipeline for the G1 task.

    Returns an OutputCombiner with an "action" output of shape matching
    the environment's action space.
    """
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix
    from isaacteleop.retargeters import Se3AbsRetargeter, Se3RetargeterConfig, TensorReorderer

    # Source nodes for XR controller and hand tracking data
    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    # Anchor transform (populated at runtime by IsaacTeleopDevice)
    world_T_anchor = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(world_T_anchor.output(ValueInput.VALUE))

    # Retargeters -- adapt to your action space
    # Example: single-arm SE3 absolute retargeter + gripper
    ee_retargeter = Se3AbsRetargeter(
        Se3RetargeterConfig(input_device=ControllersSource.RIGHT),
        name="ee_pose",
    )
    c_ee = ee_retargeter.connect({
        ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
    })

    # TensorReorderer flattens outputs into action vector
    # The output_order must match your env's ActionsCfg joint/EE ordering
    reorderer = TensorReorderer(
        input_config={
            "ee_pose": ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"],
        },
        output_order=["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"],
        name="reorder",
        input_types={"ee_pose": "array"},
    )
    c_reorder = reorderer.connect({"ee_pose": c_ee.output("ee_pose")})

    pipeline = OutputCombiner({"action": c_reorder.output("output")})
    return pipeline
```

**Key notes for the agent:**
- Quaternion elements in the `output_order` are always `quat_x, quat_y, quat_z, quat_w` (XYZW convention).
- For bimanual robots, create left + right `Se3AbsRetargeter` instances and include both in the reorderer.
- For dexterous hands, add `DexHandRetargeter` nodes connected to `HandsSource` outputs.
- See `pickplace_gr1t2_env_cfg.py` function `_build_gr1t2_pickplace_pipeline()` for a complete bimanual + dex hand example (36D action).
- See `pickplace_unitree_g1_inspire_hand_env_cfg.py` function `_build_g1_inspire_pickplace_pipeline()` for a G1 Inspire example (38D action).

---

### Step 16: Replace `teleop_devices` with `isaac_teleop` in the env config class

**Find (grep pattern):** `teleop_devices` in your env config class, or `DevicesCfg(` / `OpenXRDeviceCfg(`

**Action 16a -- Delete the old `teleop_devices` class attribute.** Remove any field like:
```python
    teleop_devices: DevicesCfg = field(default_factory=lambda: DevicesCfg(
        handtracking=OpenXRDeviceCfg(
            xr_cfg=None,
            retargeters=[...]
        ),
    ))
```

**Action 16b -- Add `isaac_teleop` in `__post_init__`.** Inside your env config class's `__post_init__` method (create one if it doesn't exist), add:

```python
    def __post_init__(self):
        super().__post_init__()

        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, 0.0),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),  # identity quaternion, XYZW
        )
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_my_g1_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
```

**Critical rules for `pipeline_builder`:**
- It must be a **callable** (function reference or lambda), never a pre-built object. `@configclass` deep-copies all mutable fields, and pipeline objects contain non-picklable C++ handles.
- If your builder returns a tuple `(pipeline, retargeters)`, use `lambda: _build_my_g1_pipeline()[0]` for `pipeline_builder` and `lambda: _build_my_g1_pipeline()[1]` for `retargeters_to_tune`.
- Adjust `anchor_pos` and `anchor_rot` to position the XR user relative to the robot in the scene.

---

### Step 17: Quaternion convention change (WXYZ to XYZW)

**Why:** Isaac Lab 3.0 globally switches from WXYZ to XYZW quaternion ordering (matching Warp, PhysX, USD, SciPy, and OpenXR conventions).

**Find (grep pattern):** Any 4-element tuple or list that looks like a quaternion: `(1.0, 0.0, 0.0, 0.0)` or `(w, x, y, z)` pattern.

**Action:** For every hardcoded quaternion in your env config, swap from `(w, x, y, z)` to `(x, y, z, w)`:

| Old WXYZ | New XYZW | Meaning |
|---|---|---|
| `(1.0, 0.0, 0.0, 0.0)` | `(0.0, 0.0, 0.0, 1.0)` | Identity |
| `(0.7071, 0.0, 0.0, 0.7071)` | `(0.0, 0.0, 0.7071, 0.7071)` | 90 deg yaw |
| `(w, x, y, z)` | `(x, y, z, w)` | General rule |

**Affected locations** (search each):
- `init_state.pos` / `init_state.rot` on robot, object, and table configs
- `anchor_rot` in `XrCfg`
- Any `idle_action` tensor with quaternion components
- Euler-to-quaternion helper outputs if stored as constants

---

### Step 18: Update `ProxyArray` access patterns

**Why:** In 3.0, robot/sensor `.data` properties return `ProxyArray` instead of raw `torch.Tensor`. Torch operations still work via `__torch_function__` interop (with a deprecation warning), but explicit `.torch` is recommended.

**Find (grep pattern):** `asset.data.body_pos_w`, `asset.data.joint_pos`, `asset.data.root_pos_w`, or any `*.data.*` access in observation/termination/event functions.

**Action:** Append `.torch` to each access:

| Before | After |
|---|---|
| `asset.data.body_pos_w` | `asset.data.body_pos_w.torch` |
| `asset.data.joint_pos` | `asset.data.joint_pos.torch` |
| `asset.data.root_pos_w` | `asset.data.root_pos_w.torch` |
| `asset.data.root_quat_w` | `asset.data.root_quat_w.torch` |
| `asset.data.body_quat_w` | `asset.data.body_quat_w.torch` |

**Note:** If you are using standard Isaac Lab MDP functions from `isaaclab_tasks`, those are already updated. Only your *custom* observation/termination/event functions need this change.

---

### Step 19: Update Pink IK config objects (if applicable)

**Why:** Pink IK task objects are now config-based, and USD-to-URDF conversion is deferred.

**Find (grep pattern):** `pink.tasks.FrameTask(`, `pink.tasks.DampingTask(`, `pink.tasks.NullSpacePostureTask(`

**Action:** Replace instantiated Pink task objects with their `Cfg` equivalents:

| Before | After |
|---|---|
| `pink.tasks.FrameTask(...)` | `FrameTaskCfg(...)` |
| `pink.tasks.DampingTask(...)` | `DampingTaskCfg(...)` |
| `pink.tasks.NullSpacePostureTask(...)` | `NullSpacePostureTaskCfg(...)` |

Also, if your config calls `ControllerUtils.convert_usd_to_urdf(...)` to pre-generate a URDF, remove that call. Instead, set `controller.usd_path` and `controller.urdf_output_dir` and let the controller handle conversion at init time.

---

### Step 20: Update Isaac Sim API imports (if used directly)

**Find (grep pattern):** `omni.physics.tensors.impl.api`, `isaacsim.core.utils`, `carb.settings.get_settings`

**Action:** Apply these import replacements if they appear in your files:

| Before | After |
|---|---|
| `import omni.physics.tensors.impl.api as physx` | `import omni.physics.tensors.api as physx` |
| `from isaacsim.core.utils.stage import ...` | `from isaaclab.sim.utils.stage import ...` |
| `from isaacsim.core.utils.prims import ...` | `from isaaclab.sim.utils.prims import ...` |
| `carb.settings.get_settings()` | `from isaaclab.app.settings_manager import get_settings_manager; get_settings_manager()` |

---

## Verification Checklist

After completing all steps, verify:

1. **No remaining `enable_pinocchio` references:** `grep -r "enable_pinocchio" <your_files>` returns nothing.
2. **No remaining `omni.log` calls:** `grep -r "omni\.log\." <your_files>` returns nothing.
3. **No remaining `DevicesCfg` / `OpenXRDeviceCfg`:** `grep -r "DevicesCfg\|OpenXRDeviceCfg" <your_files>` returns nothing.
4. **`isaac_teleop` is set in your env config:** `grep -r "isaac_teleop\s*=" <your_env_cfg>` returns a match.
5. **Quaternions are XYZW:** `grep -rn "(1\.0,\s*0\.0,\s*0\.0,\s*0\.0)" <your_files>` returns nothing (no WXYZ identity quaternions remain).
6. **The script runs:** `./isaaclab.sh -p <your_script> --task <your_task>` launches without import errors.
7. **IsaacTeleop auto-detected:** The script prints `IsaacTeleop recording started.` (not `native recording started.`) when no `--teleop_device` is passed and your env config has `isaac_teleop` set.
