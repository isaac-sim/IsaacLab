# MicroDuck ToF navigation

Drive MicroDuck with a gamepad while the navigation policy avoids obstacles. Navigation runs on
your laptop; the walking policy runs on the robot.

You need a duck with a head ToF sensor and paired gamepad, plus a Linux or macOS laptop with
[uv](https://docs.astral.sh/uv/getting-started/installation/) and SSH on the same network.
Replace `ROBOT_HOST` below with the duck's hostname or IP address.

## Load the walker

Use the same walking policy that trained the navigation model. For an ONNX export from
`isaaclab play`, combine any external weights into one file before copying it:

```bash
uv run --with onnx python - <<'PY'
import onnx

model = onnx.load("/path/to/walker/exported/policy.onnx")
onnx.save_model(model, "walking_policy.onnx", save_as_external_data=False)
PY
scp walking_policy.onnx microduck@ROBOT_HOST:/home/microduck/my_walking.onnx
ssh -t microduck@ROBOT_HOST 'sudo robotctl policy load walk /home/microduck/my_walking.onnx'
```

Support the duck while loading, then check walking and stopping with the gamepad.
[Pollen's policy-loading guide](https://github.com/pollen-robotics/microduck/blob/main/docs/robot/cheatsheet.md#trying-your-own-file)
covers the robot commands.

## Try navigation

Keep the navigation ONNX and its `.onnx.data` file together if the export has both.
From the Isaac Lab repository root:

```bash
cd scripts/deploy/microduck_tof_nav
ssh-copy-id microduck@ROBOT_HOST
uv run nav_pad.py --robot ROBOT_HOST \
    --model /path/to/navigation/exported/policy.onnx \
    --dry-run --seconds 30
```

Dry-run reads the sensors and prints commands without moving the duck. Keep the head in its
neutral training pose. To drive, run again without `--dry-run --seconds 30`:

```bash
uv run nav_pad.py --robot ROBOT_HOST \
    --model /path/to/navigation/exported/policy.onnx
```

Enter the robot's sudo password when prompted to connect the gamepad through navigation.

| Control | Action |
|---|---|
| Stick forward or turn | Drive with obstacle avoidance |
| Release the stick | Stop |
| Pull the stick back | Drive backwards without avoidance |
| X on Switch Pro | Toggle navigation |
| Ctrl-C | Stop and restore normal gamepad control |

Forward speed is capped at 0.15 m/s. Stale sensor data or missing policy commands stop navigation.
An extra brake stops translation for valid central ToF returns within 0.20 m, excluding the bottom
two floor rows. Manual driving bypasses avoidance and this brake.

Use `--max-vx` to change the speed cap (up to 0.2 m/s), `--stop-dist` for braking distance, or
`--toggle-button` for another controller. See `--help` for all options. CSV logs go to `runs/`.

If normal gamepad control does not return after exit:

```bash
ssh -t microduck@ROBOT_HOST 'sudo systemctl start padd'
```
