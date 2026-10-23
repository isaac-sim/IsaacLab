# Kitless XR teleoperation

OVXR presents Isaac Lab's OVRTX renderer and OVStage scene while Isaac Capture
provides tracking and retargeting. This workflow supports Linux x86-64 with
Newton or OVPhysX physics and one environment.

## Installation

Use Isaac Capture and Isaac Lab checkouts containing the OVXR integration.
Build Isaac Capture with Python 3.12, CMake 3.24 or newer, a C++20 compiler
and `patchelf` available on `PATH`, using its
[build guide](https://github.com/NVIDIA/IsaacCapture/blob/main/docs/source/development/build_from_source/index.rst).
For camera panels, enable `BUILD_OVXR_CAMERA_QUADS=ON`; SDK discovery and
optional build-managed acquisition are described in the
[OVXR guide](https://github.com/NVIDIA/IsaacCapture/blob/main/docs/source/getting_started/ovxr.rst).
Review the licenses supplied with the selected packages.

From Isaac Lab, select the built wheel in a local project extra:

```bash
CAPTURE_WHEEL=/absolute/path/to/isaaccapture-1.7+local-cp312-cp312-linux_x86_64.whl
uv add --optional ovxr --frozen \
    "isaaccapture[ovxr,retargeters,ui,cloudxr] @ file://${CAPTURE_WHEEL} ; platform_system == 'Linux' and platform_machine == 'x86_64'"
uv add --optional ovxr --frozen \
    "isaaclab-dev[ov] ; platform_system == 'Linux' and platform_machine == 'x86_64'" \
    "isaaclab-teleop ; platform_system == 'Linux' and platform_machine == 'x86_64'" \
    "isaaclab-mimic ; platform_system == 'Linux' and platform_machine == 'x86_64'" \
    "ipywidgets>=8.1.5 ; platform_system == 'Linux' and platform_machine == 'x86_64'" \
    "dex-retargeting==0.5.0 ; platform_system == 'Linux' and platform_machine == 'x86_64'" \
    "physx-usd-schemas==25.11.2.72696984 ; platform_system == 'Linux' and platform_machine == 'x86_64'"
```

Add `extra = "ovxr"` to the generated `isaaccapture` entry in
`tool.uv.sources`, preserving its wheel path and platform marker. In the same
table, map the optional external PhysX schemas to the existing public PyPI index:

```toml
physx-usd-schemas = { index = "pypi-public", extra = "ovxr" }
```

OvPhysX 0.7 loads these schemas at runtime but does not declare the package as
a dependency, so the optional requirement above is explicit.
Add the following pair to `tool.uv.conflicts`, preserving existing entries:

```toml
[tool.uv]
conflicts = [
    [{ extra = "ovxr" }, { extra = "teleop" }],
]
```

The general `teleop` extra selects an older Isaac Capture package. Use the local
`ovxr` extra for this workflow. Keep local wheel paths and lock changes out of
commits. To install the optional desktop mirror, include `desktop-mirror` among
the extras on the same wheel requirement.

Until the coordinated OV provider upgrade lands, a local premerge selection
also needs the following entries in the existing `tool.uv.override-dependencies`
list, preserving its other entries. These overrides and the workspace edit
below are temporary consuming-checkout settings. Do not commit them; the default
provider pins in these PRs remain unchanged pending the coordinated upgrade:

```toml
"ovphysx==0.7.0.74035321",
"ovrtx==0.6.0.1791947691",
"ovstage==0.3.0.1791986420",
```

Before locking or synchronizing, also change `ovstage==0.2.1.385922` to
`ovstage==0.3.0.1791986420` in `[project].dependencies` in your local
`source/isaaclab_ov/pyproject.toml`. A root override controls resolution but does
not rewrite the workspace package's installed `Requires-Dist`. Refresh the
editable installation with the scoped reinstall command below so `uv pip check`
sees the matching declaration. Keep this local dependency edit out of commits.

The project selects NumPy `>=2.3,<2.4`, the compatible range for the selected
Numba and cmeel-boost dependencies. Keep this range when adding local overrides.

Resolve the selected environment and check its dependencies:

```bash
uv lock --refresh-package isaaccapture --upgrade-package isaaccapture --upgrade-package packaging==23.2
uv sync --locked --extra ovxr --reinstall-package isaaclab-ov
uv run --extra ovxr python -c 'import importlib.metadata as m; print(m.version("isaaccapture"), m.version("ovxr"))'
uv pip check
```

OVPhysX declares `packaging<24`; the explicit selection above satisfies its
metadata in this non-Kit environment. The project deliberately widens this
requirement to allow Isaac Sim's packaging version too; leave that upstream
override unchanged. A metadata mismatch alone does not establish a runtime
failure. Kit uses a different environment or resolution.
Repeat this selection if a later lock upgrade replaces it. Isaac Capture's
`ovxr` extra selects OVXR and its compatible OVRTX/OVStage dependencies; see its
OVXR guide for application version overrides.

Stop applications using the environment before replacing a wheel. After a
rebuild, refresh its lock identity before synchronizing.

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

For example, replace the task in this command with any row above:

```bash
uv run --python 3.12 --extra ovxr --locked isaaclab teleop run \
    --task IsaacContrib-PickPlace-GR1T2-Abs --num_envs 1 \
    --device cpu --physics ovphysx --ovxr --cloudxr_env cloudxrjs
```

 `--ovxr` selects OVStage
and starts CloudXR with the `cloudxrjs` profile by default. Connect a supported
CloudXR client, then use Play, Pause and Reset as in Kit-based teleoperation.
`--ovxr` and Kit's `--xr` select separate XR runtimes and are mutually exclusive.
An explicit `ISAAC_LAB_OVRTX_USE_OVSTAGE=0` is incompatible with OVXR.

No local window opens by default. Headset rendering still uses the GPU and works
with `DISPLAY` unset. The teleoperation CLI does not need a `--headless` flag.
`--ovxr_worker_timeout` bounds renderer startup; it does not bound waiting for a
client to establish its OpenXR session. Building a Capture wheel and resolving
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
the run. Reset does not rewind recorded input. `--max_stage_load_wait_s` bounds
renderer startup separately; the replay duration starts after warmup and
includes any lazy first-step physics initialization. Deadlines are checked
between steps. Application render-call timing statistics are unavailable when
OVXR owns rendering independently.

HDF5 demonstration recording and replay continue to use the Kit/Isaac Sim path.
