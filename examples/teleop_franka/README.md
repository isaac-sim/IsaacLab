# Franka Reach teleoperation

This optional task package adds Isaac Capture keyboard or Quest controller input to the
existing Franka Reach RL configuration. It reuses the environment implementation and
controllers, registering `Isaac-Reach-Franka-Teleop` without modifying `Isaac-Reach-Franka`.

Start with [the task configuration](isaaclab_teleop_franka/franka_env_cfg.py).
The [workflow guide](../../docs/source/how-to/teleop_rl_env.rst#franka-reach-examples)
contains installation prerequisites, runnable commands, controls, and test instructions.

- `presets=diffik,keyboard`: keyboard input to the existing relative IK controller.
- `presets=diffik_abs,xr`: calibrated right Quest controller input to the existing absolute
  IK controller, with 1:1 position displacement and an operator-only XR view.
- No presets: the original joint-position controller, with no teleop input or XR view.

[XR tracking and calibration](isaaclab_teleop_franka/xr.py) live in this example package;
they do not change the shared playback loop or the controller implementation.
