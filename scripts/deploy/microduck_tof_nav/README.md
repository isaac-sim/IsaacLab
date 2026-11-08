# MicroDuck ToF navigation

Drive MicroDuck with a gamepad while a navigation policy helps it avoid obstacles. The navigation
ONNX runs on your laptop; the walking policy runs on the robot.

The original controller was tested on robot software **0.18.1**. The command-history and watchdog
fixes included here have passed offline tests but still need a robot test.

## Before you start

You need a MicroDuck with a working head ToF sensor and paired gamepad, plus a Linux or macOS
laptop on the same network with [uv](https://docs.astral.sh/uv/getting-started/installation/) and
OpenSSH. uv installs Python dependencies on first use.

Use the **same walking policy that was used to train the navigation model**. The current
[environment configuration](../../../source/isaaclab_tasks/isaaclab_tasks/contrib/microduck_obstacle/obstacle_avoidance_env_cfg.py)
uses the upstream walker. The navigation export must include normalization and accept 1343 inputs
with three outputs.

Keep `policy.onnx` and its companion `policy.onnx.data` together if the export has both files.
Pass the navigation export to `--model` directly; it stays on the laptop.

## Load the walking policy

Use the walking ONNX exported by `isaaclab play`. First make a single-file copy so it can be
loaded on the robot even if the original export uses a separate data file:

```bash
uv run --with onnx python - <<'PY'
import onnx

model = onnx.load("/path/to/walker/exported/policy.onnx")
onnx.save_model(model, "walking_policy.onnx", save_as_external_data=False)
PY
scp walking_policy.onnx microduck@ROBOT_HOST:/home/microduck/my_walking.onnx
ssh microduck@ROBOT_HOST
robotctl policy list
sudo robotctl policy load walk /home/microduck/my_walking.onnx
```

Replace `ROBOT_HOST` with your robot's hostname or IP address. Support the duck when loading, then
check walking and stopping with the gamepad. `robotctl monitor` shows state and ToF readings;
`sudo robotctl policy reset walk` restores the default walker. Run `exit` to return to your laptop.

See Pollen's [policy-loading guide](https://github.com/pollen-robotics/microduck/blob/main/docs/robot/cheatsheet.md#trying-your-own-file)
for robot setup.

## Check the policy without moving the robot

From the Isaac Lab repository root:

```bash
cd scripts/deploy/microduck_tof_nav
ssh-copy-id microduck@ROBOT_HOST
uv run nav_pad.py --robot ROBOT_HOST \
    --model /path/to/navigation/exported/policy.onnx \
    --dry-run --seconds 30
```

Dry-run copies two helpers to `~/tof-nav` on the robot, reads body state and ToF, and prints the
commands it would send. It sends no movement commands and does not restart the gamepad daemon.

Stand the duck with its head in the neutral training pose. Move an obstacle through the sensor's
view and watch the commands: positive yaw turns left, negative turns right. Logs go to `runs/`.

## Drive with navigation

From the same folder:

```bash
uv run nav_pad.py --robot ROBOT_HOST \
    --model /path/to/navigation/exported/policy.onnx
```

The script asks for sudo on the robot to route the gamepad through the navigation controller.
When the proxy exits, its temporary service requests a restart of the normal gamepad daemon.

| Control | Result |
|---|---|
| Stick forward or turn | Navigation adjusts your request; forward speed is capped at 0.15 m/s |
| Release the stick | Stop |
| Pull the stick back | Ordinary gamepad driving, without navigation |
| X (`BTN_NORTH` on Switch Pro) | Toggle navigation; off means ordinary gamepad driving |
| Ctrl-C | Stop navigation and restore the normal gamepad connection |

Stale body state or ToF stops navigation. The watchdog stops active navigation if the stick request
expires or no policy command arrives for 0.25 s. An extra brake stops translation for valid returns
within 0.20 m in the central four sensor columns, excluding floor rows 6–7. These checks do not cut
servo power. Ordinary gamepad driving bypasses navigation and its extra brake.

Check stick release, controller disconnect and Ctrl-C in a clear area before trying obstacles at
low speed. Record collisions, falls and interventions. If the gamepad does not work after exit:

```bash
ssh -t microduck@ROBOT_HOST 'sudo systemctl start padd'
```

## Options and logs

| Option | Purpose |
|---|---|
| `--max-vx 0.1` | Set the navigation forward cap; maximum 0.2 m/s |
| `--stop-dist 0.3` | Set the extra braking distance in metres |
| `--stick-turn-override` | Use the operator's turn instead of learned yaw |
| `--toggle-button BTN_WEST` | Choose another navigation toggle button |
| `--seconds 60` | End the session after a minute |
| `--request 0.15 0 0` | Set the simulated stick request for dry-run only |
| `--user NAME` | Change the SSH user from `microduck` |
| `--log-dir PATH` | Change the CSV destination from `runs/` |

Logs include requests, actions, commands, sensor age/validity, braking, mode and falls.
`policy_drives=0` marks ordinary gamepad control. Sensor age does not measure the full network delay.

## Model inputs and command scaling

Body state arrives at about 50 Hz and ToF at about 15 Hz. Ten control-tick samples cover about
0.2 s, so history entries can repeat a sensor frame.

| Input indices | Values |
|---|---|
| 0–2 | Body gyro, rad/s |
| 3–5 | Gravity in the robot frame |
| 6–19 | Fourteen joint positions relative to the default pose, rad |
| 20–33 | Fourteen joint velocities, delayed one control tick, rad/s |
| 34–36 | Previous navigation action |
| 37–39 | Requested forward, sideways and turning speeds |
| 40–49 | Zero head/body pose commands, as used in training |
| 50–52 | Previous applied command, including caps, braking and overrides |
| 53–1332 | Ten samples, oldest first: 64 ranges in metres, then 64 validity flags per sample |
| 1333–1342 | The stored age of each sample, seconds |

Joint order is left hip yaw/roll/pitch, left knee/ankle, neck pitch, head pitch/yaw/roll, then
right hip yaw/roll/pitch and right knee/ankle. The mouth at robotd index 9 is omitted. Position
offsets are listed in `DEFAULT_POSE` in `nav_pad.py`. ToF statuses 5 and 9 are valid within
0.12–4 m; invalid zones have range zero and validity zero.

Outputs in `[-1, 1]` are multiplied by `[0.35, 0.12, 0.75]` and added to the requested forward,
sideways and turning speeds. Training speed, acceleration and lateral limits are applied first,
then the lower deployment caps, optional turn override and extra brake. Stops caused by an
explicit stop request or stale inputs take effect immediately, without acceleration limiting.

## Files and tests

- `nav_pad.py` runs laptop inference and logs commands.
- `relay_on_duck.py` reads robot state and ToF over the robot's local sockets.
- `padproxy_on_duck.py` forwards gamepad/navigation commands and runs the watchdog.

The robot helpers need only standard Python. To run offline tests from the repository root:

```bash
uv run --no-sync --with onnxruntime --with pytest python -m pytest \
    --confcutdir scripts/deploy/microduck_tof_nav \
    scripts/deploy/microduck_tof_nav/test_nav_pad.py -q
```

These tests check command mapping against the training code, CPU ONNX inference, stale-input
stopping and the watchdog using a fake robot daemon. They never contact a robot.
