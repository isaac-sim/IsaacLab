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

## Switching a Kit workflow to kitless

Each step of the Kit teleoperation and imitation-learning workflow runs without Isaac Sim. Install
the `ovxr` selection (add `mimic` for the imitation-learning scripts) instead of `isaacsim`, then:

| Step | Kit | Kitless |
|---|---|---|
| Teleoperate | `isaaclab teleop run --xr` | `isaaclab teleop run --ovxr` |
| Record demonstrations | `isaaclab teleop record --xr` | `isaaclab teleop record --ovxr` |
| Replay demonstrations | `isaaclab teleop replay` | same command; add `--renderer ovrtx` for an RTX camera |
| Replay an MCAP capture | `teleop_replay_agent.py` | `teleop_replay_agent.py --ovxr` |
| Annotate, generate (Mimic) | `annotate_demos.py`, `generate_dataset.py` | same commands with `physics=...` |
| Train, play (robomimic) | `train.py`, `play.py` | same commands; `play.py` takes `physics=...` |

Select the physics backend with `--physics ovphysx` or `--physics newton_mjwarp`;
the Mimic scripts take the same choice as `physics=...`. The automatic `physx` selector follows the usual
Isaac Lab rule: Isaac Sim PhysX when the run needs Kit, otherwise the task's OvPhysX settings.
Isaac Sim PhysX (`isaacsim_physx`) is not available
without Kit. Cameras that render through Isaac Sim RTX need a kitless renderer, such as `renderer=ovrtx`,
when a script keeps them; HDF5 replay of a task such as GR1T2, whose `robot_pov_cam` renders through
Isaac Sim RTX, needs `--renderer ovrtx`. The OVXR session strips camera observations, so kitless HDF5
demonstrations are state-only. The G1 locomanipulation SDG task still requires Isaac Sim. To restart an
OVXR session, start the command again.

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
A blank or below-scene main view with a working camera panel warrants checking
this client origin before changing the renderer.

`--ovxr` and Kit's `--xr` select separate XR runtimes and are mutually exclusive.

OVXR live teleoperation, recording and MCAP replay run without a local window by default.
Add `--ovxr_desktop_mirror` to open the mirror on a host with a desktop display;
`--no-ovxr_desktop_mirror` explicitly disables it; `--viz` remains the general
visualizer selector. `--ovxr_session_mode {ar,vr}` (default `ar`) selects the OpenXR
session mode in live teleoperation, recording and replay. Headset rendering still uses the GPU
and works with `DISPLAY` unset when the mirror is disabled. PiP is independent of the mirror.
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
teleoperation, with the same `--ovxr_session_mode`, `--ovxr_step_hz`, camera-panel and
desktop-mirror options;
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

## Camera panels and desktop mirror

Select up to three task cameras:

```bash
uv run --extra ovxr isaaclab teleop run --ovxr \
    --task IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs \
    --num_envs 1 --device cpu --physics ovphysx presets=rgb \
    --ovxr_camera_feed robot_pov_cam left_wrist_cam right_wrist_cam
```

`--ovxr_camera_feed_fps` limits panel updates. Panel cameras must be
enabled in the task's `xr_camera_feeds`. OVXR shows camera feeds head-locked, using the
size, gap and distance of the task's `xr_camera_feed_layout`.
The session snapshots its camera selection at
preparation; create a new session to change it. Panels use the same renderer as
the headset and do not require a desktop display. OVXR panels show no label; a
feed's `label` and DLSS settings apply to Kit panels only, and the desktop mirror
lists each camera by its sensor name.

GR1T2 and G1 tasks provide `robot_pov_cam`. Fixed-base G1 adds `left_wrist_cam` and
`right_wrist_cam` with `presets=rgb`, as in the example above. SO101 does not declare
a PiP camera in its scene. Use the desktop mirror without `--ovxr_camera_feed` for that task.

Selected OVXR cameras feed the panels only. Their policy image observations are
removed even with `presets=rgb`, so the application does not request a second
camera render while OVXR owns rendering. The `presets=rgb` policy-image selection
applies to Kit workflows.

In Kit XR, the fixed-base G1 task shows its head camera panel with the state-only
policy. `presets=rgb` adds the wrist cameras and their panels, and resets then
perform three rerenders when RTX sensors are present so image consumers receive
refreshed frames. For Kit XR without image feeds, use `--disable_external_cameras`.
OVXR keeps only the explicitly selected panel cameras. For a custom state-only
application, remove unused camera configurations before environment construction;
selecting a state-only policy alone does not disable sensors used by a viewer or recorder.

The optional desktop mirror shows completed eye or camera images locally.
`--ovxr_desktop_mirror` enables it; `--ovxr_desktop_mirror_fps` limits its update rate. The mirror requires the
`desktop-mirror` extra, Python 3.12 or newer, and a usable display. On Linux,
use a GPU-backed desktop session for the mirror. OVUI's default Vulkan backend
requires no `OMNIUI_BACKEND` setting. Lab and Capture preserve the application's
backend choice; an already initialized OVUI backend cannot be reconfigured.
The mirror requires Vulkan; remove conflicting backend settings before OVUI
initialization. The mirror opens after the first submitted
XR image; closing it requests application shutdown. Eye images exclude the
separate OpenXR camera panels; select a camera product to inspect that feed.

## Physics and rendering on separate GPUs

For GPU-intensive physics, select the physics device with `--device` and the
rendering devices with `--ovrtx_active_cuda_gpus`. The rendering option requires
`--ovxr` in the live teleoperation and recording CLIs. The MCAP replay entrypoint does
not expose GPU selection.

For live teleoperation:

```bash
uv run --extra ovxr isaaclab teleop run --ovxr \
    --task IsaacContrib-PickPlace-GR1T2-Abs --num_envs 1 \
    --physics newton_mjwarp --device cuda:1 \
    --ovrtx_active_cuda_gpus 0
```

By default, GPU physics and rendering use the same CUDA device; CPU physics
leaves GPU selection to OVRTX. When the rendering GPUs exclude the physics GPU,
the renderer copies deforming geometry and rigid transforms from the physics GPU
to reusable buffers on the first rendering GPU, and OVStage reads them after the
copy completes. This adds a copy to each such update. Other
rendering GPUs, and other inputs such as camera poses, read the physics GPU directly.
CPU data and already-local geometry keep their placement; mixed CPU/CUDA geometry
batches are unsupported by OVStage.

OVRTX fixes the rendering GPU selection at the first native initialization in
each process. The OVXR renderer guards its own selection: the first OVXR session
fixes it, recreating a session with the same selection is supported, and a
different selection is rejected, including after teardown or failed startup.
Start a new process to change rendering GPUs. For CPU simulation, consistently
use native selection or an explicit GPU list throughout the process.

Ordinary OVRTX cameras render on the simulation GPU. When the session renders on
any other GPUs, the OVXR renderer rejects an ordinary OVRTX camera registered in
the same simulation; the OVXR entrypoints remove such cameras. The guard does not
cover an ordinary OVRTX camera created in another simulation of the same process,
before or after a split-GPU session: OVRTX keeps the selection of whichever
renderer initialized it first, so run such cameras in a separate process.

Custom geometry producers must order updates on their current Warp stream and
avoid concurrent mutation during publication. These controls do not change
physics fidelity or guarantee a rendering rate.

The compositor/encoder has a separate GPU choice. Align CloudXR's
`NV_CXR_GPU_INDEX_CUDA` with the rendering GPU and verify effective physical GPU
identities in the runtime logs. `CUDA_VISIBLE_DEVICES` can remap CUDA ordinals;
two installed GPUs alone do not enable a split.

Before splitting GPUs, check that PCIe peer copies return correct values in both
directions on this host, with the application's `CUDA_VISIBLE_DEVICES` selection:

```bash
uv run --extra ovxr python -m isaaclab_teleop.tools.check_cuda_peer --devices cuda:0 cuda:1
```

The check copies test patterns both ways through the renderer's own geometry-publication
path and exits nonzero on a transfer error or a value mismatch. A pass covers only those
copies on this host, not rendering, physics or performance. Peer-capability flags and CUDA
completion alone do not establish data integrity; see NVIDIA's
[bare-metal Linux IOMMU limitation for PCIe peer copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#host-iommu-hardware-pci-access-control-services-and-vms).
If the check fails, keep physics, rendering and compositor on one GPU; changing the host's
IOMMU configuration is an administrator's task.

Only rigid transforms and geometry points use the completed-copy publication path. Camera
poses, markers and runtime meshes, and visual materials are small per-publication writes that
OVStage still reads from the physics GPU. OVStage chooses how it reads them; the check above
does not cover that route, and these writes have no completed-copy guarantee.
