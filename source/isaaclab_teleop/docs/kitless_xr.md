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

Start with CPU OvPhysX using the command below. The same registered-task
selectors apply to live input and MCAP replay:

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

`--ovxr` selects OVStage and starts CloudXR with the `cloudxrjs` profile by default. Connect a supported
CloudXR client, then use its Play and Reset controls. Pause/resume is available only on clients
that expose it; Reset is a separate action.

For desktop IWER, select **VR** and set **Reference Space** to **local-floor**
under **Teleoperator adjustment** before connecting. This keeps the emulated
head and controllers relative to the floor. Between trials, end the previous XR
session and reload the client page before reconnecting; recheck both settings.
A blank or below-scene main view with a working camera panel warrants checking
this client origin before changing the renderer.

`--ovxr` and Kit's `--xr` select separate XR runtimes and are mutually exclusive.
An explicit `ISAAC_LAB_OVRTX_USE_OVSTAGE=0` is incompatible with OVXR.

OVXR live teleoperation and MCAP replay open a desktop mirror by default. Add
`--headless` to disable the local window, including on hosts without a desktop
display. Headset rendering still uses the GPU and works with `DISPLAY` unset
when `--headless` is selected. PiP remains available in headless mode.
Renderer startup has a bounded attachment timeout; it does not bound waiting for a
client to establish its OpenXR session. Installing Capture and resolving
Python packages do not warm application-specific RTX shaders or Newton/Warp
physics kernels. First runtime preparation and the first physics step can take
longer than subsequent runs. Retain startup logs and distinguish stage/render
preparation from client connection and active simulation.

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

## Camera panels and desktop mirror

Select up to three task cameras and a head-locked layout:

```bash
uv run --extra ovxr isaaclab teleop run --ovxr \
    --task IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs \
    --num_envs 1 --device cpu --physics ovphysx \
    --ovxr_pip_camera robot_pov_cam left_wrist_cam right_wrist_cam \
    --ovxr_pip_head_locked
```

`--ovxr_pip_fps` limits panel updates. `--ovxr_pip_view_position X Y Z` overrides
one panel's position; multiple panels use the task layout. Panel cameras must be
enabled in the task's `xr_camera_feeds`. World and viewer-start layouts are not
supported by the OVXR presenter. The session snapshots its camera selection at
preparation; create a new session to change it. Panels use the same renderer as
the headset and do not require a desktop display.

GR1T2 and G1 locomanipulation provide `robot_pov_cam`; fixed-base G1 also provides
`left_wrist_cam` and `right_wrist_cam`. SO101 does not declare a PiP camera in
its scene. Use the desktop mirror without `--ovxr_pip_camera` for that task.

The default desktop mirror shows completed eye or camera images locally.
`--headless` disables it and takes precedence over `--ovxr_desktop_mirror`.
The explicit mirror flag is also accepted, but is unnecessary for the default.
`--ovxr_desktop_mirror_fps` limits its update rate. The mirror requires the
`desktop-mirror` extra, Python 3.12 or newer, and a usable display. On Linux,
use a GPU-backed desktop session for the mirror. OVUI's default Vulkan backend
requires no `OMNIUI_BACKEND` setting. Lab and Capture preserve the application's
backend choice; an already initialized OVUI backend cannot be reconfigured.
The mirror requires Vulkan; remove conflicting backend settings before OVUI
initialization. The mirror opens after the first submitted
XR image; closing it requests application shutdown. Eye images exclude the
separate OpenXR camera panels; select a camera product to inspect that feed.

For SO101, check right-controller grip/squeeze engagement, release and re-engagement,
proportional trigger control and Reset separately from cube grasping. Object retention and cube stacking also depend on contact properties and
grasp placement; validate them separately from the input controls.

## Physics and rendering on separate GPUs

For GPU-intensive physics, select the physics device with `--device` and the
rendering devices with `--ovrtx_active_cuda_gpus`. Both rendering device options
require `--ovxr` in the live teleoperation CLI. The MCAP replay entrypoint does
not expose these GPU-selection options.

For live teleoperation:

```bash
uv run --extra ovxr isaaclab teleop run --ovxr \
    --task IsaacContrib-PickPlace-GR1T2-Abs --num_envs 1 \
    --physics newton_mjwarp --device cuda:1 \
    --ovrtx_active_cuda_gpus 0 --geometry_publication_device cuda:0
```

By default, GPU physics and rendering use the same CUDA device; CPU physics
leaves GPU selection to OVRTX. `--geometry_publication_device cuda:N` copies
foreign-GPU deforming geometry and rigid transforms to reusable buffers on the selected device and
waits for completion before publishing it. It does not select the physics or
rendering device. Omit it to preserve source-buffer placement. CPU data and
already-local geometry keep their placement; mixed CPU/CUDA geometry batches
are unsupported by OVStage.

Custom geometry producers must order updates on their current Warp stream and
avoid concurrent mutation during publication. These controls do not change
physics fidelity or guarantee a rendering rate.

The compositor/encoder has a separate GPU choice. Align CloudXR's
`NV_CXR_GPU_INDEX_CUDA` with the rendering GPU and verify effective physical GPU
identities in the runtime logs. `CUDA_VISIBLE_DEVICES` can remap CUDA ordinals;
two installed GPUs alone do not enable a split.

Before using PCIe peer transfers, verify synchronized copy **values** in both
directions with initialized destination buffers:

```bash
uv run --extra ovxr python -m isaaclab_teleop.check_cuda_peer --devices cuda:0 cuda:1
```

Run this with the same `CUDA_VISIBLE_DEVICES` selection as the application. It
prints each device's UUID and PCI identity, checks two patterns in each direction,
and exits nonzero on unsupported access, a detected transfer error or a value mismatch. This
small transfer check does not qualify rendering, physics or performance.
Peer-capability flags and CUDA
completion alone do not establish data integrity. NVIDIA documents a
[bare-metal Linux IOMMU limitation for PCIe peer copies](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html#host-iommu-hardware-pci-access-control-services-and-vms).
GPU-only identity/passthrough can bypass translation while leaving IOMMU enabled
globally. Validate the selected configuration before use. Host preparation belongs to an administrator
with idle GPUs; application startup must not detach drivers or change IOMMU or
host services. Keep physics, rendering and compositor on one physical GPU when
the required peer-transfer path is unsupported. No automatic CPU staging is used.
