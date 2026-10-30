# Kitless XR teleoperation

OVXR presents Isaac Lab's OVRTX renderer and OVStage scene while Isaac Capture
provides tracking and retargeting. This workflow supports Linux x86-64 with
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
Configure the `ovxr` dependency selection using
[Installing kitless XR](kitless_xr_source.md), then install and
check that selection:

```bash
uv sync --python 3.12 --locked --extra ovxr
uv pip check
```

The runtime commands below use this prepared `ovxr` selection.

The live and MCAP workflows are implemented in `isaaclab_teleop.entrypoints`.
The command-line entrypoints load registered downstream tasks before parsing
and restore the caller's arguments on return. The scripts under
`scripts/environments/teleoperation/` delegate to these package entrypoints.

## Live teleoperation

```bash
uv run --extra ovxr isaaclab teleop run \
    --task IsaacContrib-PickPlace-GR1T2-Abs --num_envs 1 \
    --device cuda:0 --physics newton_mjwarp --ovxr
```

The same registered-task selectors apply to live input and MCAP replay:

| Task | Physics selectors |
| --- | --- |
| `IsaacContrib-PickPlace-GR1T2-Abs` | `ovphysx`, `newton_mjwarp`, `isaacsim_physx` |
| `IsaacContrib-PickPlace-Locomanipulation-G1-Abs` | `ovphysx`, `newton_mjwarp`, `isaacsim_physx` |
| `IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs` | `ovphysx`, `newton_mjwarp`, `isaacsim_physx` |
| `IsaacContrib-Stack-Cube-SO101-IK-Abs-v0` | `ovphysx`, `newton_mjwarp`, `isaacsim_physx` |

Omitting `--physics` retains the Kit/Isaac Sim default. Use `--physics ovphysx
--device cpu` for standalone CPU physics, or `--physics newton_mjwarp --device
cuda:0` for Newton; these standalone selections support `--ovxr`. Kit's
`isaacsim_physx` selection uses its separate `--xr` workflow and Isaac Sim
installation, not `--ovxr`. The task presets retain backend-specific solver
settings and, for SO101, choose the matching USD Physics variant.

The SO101 IK task resets its joints to the configured nominal pose without
joint-position noise, giving the clutch a repeatable reset pose. Its configured
home transform supplies the clutch calibration. This applies to its Kit and
kitless backends. The separate joint-control task retains its randomized reset.

For example, replace the task in this command with any row above:

```bash
uv run --python 3.12 --extra ovxr --locked isaaclab teleop run \
    --task IsaacContrib-PickPlace-GR1T2-Abs --num_envs 1 \
    --device cpu --physics ovphysx --ovxr --cloudxr_env cloudxrjs
```

 `--ovxr` selects OVStage
and starts CloudXR with the `cloudxrjs` profile by default. Connect a supported
CloudXR client, then use Play, Pause and Reset as in Kit-based teleoperation.

For desktop IWER, select **VR** and set **Reference Space** to **local-floor**
under **Teleoperator adjustment** before connecting. This keeps the emulated
head and controllers relative to the floor. Between trials, end the previous XR
session and reload the client page before reconnecting; recheck both settings.
A blank or below-scene main view with a working camera panel warrants checking
this client origin before changing the renderer.

`--ovxr` and Kit's `--xr` select separate XR runtimes and are mutually exclusive.
An explicit `ISAAC_LAB_OVRTX_USE_OVSTAGE=0` is incompatible with OVXR.

No local window opens by default. Headset rendering still uses the GPU and works
with `DISPLAY` unset. The teleoperation CLI does not need a `--headless` flag.
Renderer startup has a bounded attachment timeout; it does not bound waiting for a
client to establish its OpenXR session. Installing Capture and resolving
Python packages do not warm application-specific RTX shaders or Newton/Warp
physics kernels. First runtime preparation and the first physics step can take
longer than subsequent runs. Retain startup logs and distinguish stage/render
preparation from client connection and active simulation; do not treat a warm
retry as proof that an earlier timeout was fixed.

Use `--step_hz 20` to cap live control at 20 updates per second while XR rendering
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
view. A recorded STOP, task success/failure, or `--max_replay_duration_s` ends
the run. Playback waits for recorded START messages, including after a recorded
RESET. Reset does not rewind recorded input. For legacy captures started locally
without recorded START messages, add `--start_on_replay` to start automatically
at startup and after RESET. `--max_stage_load_wait_s` bounds
renderer startup separately; the replay duration starts after warmup and
includes any lazy first-step physics initialization. Deadlines are checked
between steps. Application render-call timing statistics are unavailable when
OVXR owns rendering independently.

HDF5 demonstration recording and replay continue to use the Kit/Isaac Sim path.
