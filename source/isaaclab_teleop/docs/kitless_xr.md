# Kitless XR teleoperation

OVXR presents the Isaac Lab scene from an OVStage that OVRTX renders, while Isaac
Capture provides tracking and retargeting. This workflow supports Linux x86-64 with
Newton or OVPhysX physics and one environment.

## Prerequisites

Before installing, complete the workstation and streaming setup in the
[CloudXR prerequisites](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html#prerequisites).
Use the OVXR package selection below for this workflow; the Kit installation
commands in that guide belong to Kit XR.

- Follow [CloudXR network setup](https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html)
  for connectivity and firewall requirements between the workstation and client.
- Follow [XR client setup](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html#connect-an-xr-device)
  for a supported headset or the CloudXR.js browser client, including certificate setup.
  The [Isaac Capture Quick Start](https://nvidia.github.io/IsaacCapture/main/teleop/quick_start.html#connect-an-xr-headset)
  also covers the desktop IWER client.
- Review the [CloudXR license](https://github.com/NVIDIA/IsaacCapture/blob/main/deps/cloudxr/CLOUDXR_LICENSE)
  and complete the first-run EULA acceptance described in
  [CloudXR startup](https://nvidia.github.io/IsaacCapture/main/references/cloudxr.html#start-the-service)
  before launching without an interactive terminal.

## Installation

Use Python 3.12 and the Isaac Capture wheel selected by the installation
guide.
Use the checked-in `ovxr` dependency selection described in
[Installing kitless XR](kitless_xr_source.md), then install and
check that selection:

```bash
uv sync --python 3.12 --locked --extra ovxr
```

The runtime commands below use this prepared `ovxr` selection.

The live, recording and MCAP workflows are implemented in `isaaclab_teleop.entrypoints`.
The command-line entrypoints load registered downstream tasks before parsing
and restore the caller's arguments on return. The scripts under
`scripts/environments/teleoperation/` and `scripts/tools/record_demos.py` delegate to
these package entrypoints.

## Live teleoperation

Start with CPU OvPhysX using the command below. The same registered-task
selectors apply to live input and MCAP replay:

| Task | Physics selectors |
| --- | --- |
| `IsaacContrib-PickPlace-GR1T2-Abs` | `ovphysx`, `newton_mjwarp`, `isaacsim_physx` |
| `IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs` | `ovphysx`, `newton_mjwarp`, `isaacsim_physx` |
| `IsaacContrib-PickPlace-Locomanipulation-G1-Abs` | `ovphysx`, `newton_mjwarp`, `isaacsim_physx` |
| `IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs` | `ovphysx`, `newton_mjwarp`, `isaacsim_physx` |
| `IsaacContrib-Stack-Cube-SO101-IK-Abs-v0` | `ovphysx`, `isaacsim_physx` |

Omitting `--physics` retains the Kit/Isaac Sim default. Use `--physics ovphysx
--device cpu` for standalone CPU physics, or `--physics newton_mjwarp --device
cuda:0` for Newton; these standalone selections support `--ovxr`. Kit's
`isaacsim_physx` selection uses its separate `--xr` workflow and Isaac Sim
installation, not `--ovxr`. The task presets retain backend-specific solver
settings and, for SO101, choose the matching USD Physics variant. SO101 also
defines a `newton_mjwarp` preset, but the environment catalog
(`tools/environ_docs.py`) disables that combination after the environment smoke
matrix found runtime constraints, so it is not listed here.

For example, replace the task in this command with any row above:

```bash
uv run --extra ovxr isaaclab teleop run \
    --task IsaacContrib-PickPlace-GR1T2-Abs --num_envs 1 \
    --device cpu --physics ovphysx --ovxr
```

`--ovxr` selects OVStage and starts CloudXR with the `cloudxrjs` profile by
default. Connect a supported CloudXR client, then use its Play and Reset
controls. Pause/resume is available only on clients that expose it; Reset is a
separate action.

The SO101 IK task resets its joints to the configured nominal pose without
joint-position noise, giving the clutch a repeatable reset pose. Its configured
home transform supplies the clutch calibration. This applies to its Kit and
kitless backends. The separate joint-control task retains its randomized reset.

For SO101, check right-controller grip/squeeze engagement, release and
re-engagement, proportional trigger control and Reset separately from cube
grasping. Object retention and cube stacking also depend on contact properties
and grasp placement; validate them separately from the input controls.

For desktop IWER, select **VR** and set **Reference Space** to **local-floor**
under **Teleoperator adjustment** before connecting. This keeps the emulated
head and controllers relative to the floor. Between trials, end the previous XR
session and reload the client page before reconnecting; recheck both settings.

`--ovxr` and Kit's `--xr` select separate XR runtimes and are mutually exclusive.

No local window opens by default. `--ovxr_session_mode {ar,vr}` (default `ar`) selects the
OpenXR session mode in live teleoperation, recording and replay. Headset rendering still
uses the GPU and works with `DISPLAY` unset.
Renderer startup waits up to 20 minutes for attachment; it does not bound waiting for a
client to establish its OpenXR session. A first start compiles shaders, and other rendering
processes on the same GPU slow attachment; either can take several minutes. Attachment cannot be
interrupted: Ctrl+C takes effect once it completes.
Installing Capture and resolving
Python packages do not warm application-specific RTX shaders or Newton/Warp
physics kernels. First runtime preparation and the first physics step can take
longer than subsequent runs. Retain startup logs and distinguish stage/render
preparation from client connection and active simulation.

Use `--ovxr_step_hz 20` to cap live control at 20 updates per second while XR rendering
runs independently. Time spent reading input, stepping physics and publishing
the scene counts toward that interval. Slow steps do not cause catch-up steps.
Omit the option or use zero for unthrottled control. Paused polling uses the
environment action interval when no explicit rate is set. This option requires
`--ovxr` and does not change physics dt or decimation.

## MCAP replay

```bash
uv run --extra ovxr python scripts/environments/teleoperation/teleop_replay_agent.py \
    --task IsaacContrib-PickPlace-GR1T2-Abs --num_envs 1 \
    --device cpu --physics ovphysx --ovxr \
    --replay_file /absolute/path/to/session.mcap
```

MCAP supplies tracker and control input; the connected headset supplies the
view. Without `--ovxr`, the replay runs headless on the selected kitless backend, which suits
automated success checks; its report carries each run's outcome, with empty render timing because
no application render loop runs. A task that sets `target_frame_prim_path`, such as
`IsaacContrib-Stack-Cube-SO101-IK-Abs-v0`, cannot replay headless without Kit and is rejected before
launch: use `--ovxr` or a Kit run. A kitless replay consumes exactly one recorded frame per step, so
it runs as fast as the simulator allows and does not depend on rendering or loop timing. Add
`--require_hand_replay_data` to end the run at the end of the capture's recorded hand data, which the
agent counts before the run, and report it as a success when the applied actions changed; this checks
that a capture plays back in full without relying on the task's success condition. A tracking dropout
inside the recording does not end the run, and a capture without a two-hand frame is rejected. A recorded STOP, task success/failure, or `--max_replay_duration_s` ends
the run. Playback waits for recorded START messages, including after a recorded
RESET. Reset does not rewind recorded input. For legacy captures started locally
without recorded START messages, add `--start_on_replay` to start automatically
at startup and after RESET. The replay duration starts after renderer startup and warmup, and
includes any lazy first-step physics initialization. Deadlines are checked
between steps. Application render-call timing statistics are unavailable when
OVXR owns rendering independently.

## Demonstration recording

```bash
uv run --extra ovxr isaaclab teleop record \
    --task IsaacContrib-Stack-Cube-SO101-IK-Abs-v0 \
    --device cpu --physics ovphysx --ovxr \
    --dataset_file ./datasets/so101_stack.hdf5 --num_demos 5
```

`record --ovxr` writes HDF5 demonstrations through the same OVXR session as live
teleoperation, with the same `--ovxr_session_mode` and `--ovxr_step_hz` options;
`--ovxr_step_hz` replaces the `--step_hz` limit. Recording starts with the client's Play
control. Resets between attempts keep the kitless simulator instead of rebuilding it, as in
MCAP replay. Instructions are logged because no Kit window opens. Add `--mcap_record_path`
to also capture the session's tracker and control input for MCAP replay. `replay_demos.py`
replays the HDF5 file on the kitless backend selected with `--physics`. A task that keeps an
Isaac Sim RTX camera also needs a kitless renderer, such as `--renderer ovrtx`; otherwise the
camera's Kit renderer conflicts with the kitless physics.

Recording applies a task's initial hand-tracking gate and action conditioning and validation
before each step, as live teleoperation and MCAP replay do, so the humanoid tasks also record
with kitless physics.
