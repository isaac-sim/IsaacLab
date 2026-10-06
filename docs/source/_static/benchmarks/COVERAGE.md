# Weekly benchmark coverage — 8,192 environments

The environment browser compares paired Collection FPS (rollouts and inference)
and Training FPS (including policy updates) from successful WARM training entries.
Both metrics always come from the same source record and entry.

## Weekly history

Sunday graph positions identify weekly snapshots with a Monday cutoff. Each full
selection window runs Tuesday through Monday inclusive; select the latest eligible
run within that window, never carry a measurement from an earlier window.
This includes September 28 and October 5 measurements at September 27 and October 4
positions. The first snapshot contains only August 24 because the export starts on
that date. The current snapshot is partial as October 5 collection is ongoing.
Ingestion timestamps and original measurement timestamps remain unchanged.

| Sunday position | Selected pairs | Tasks | Expected pairs |
| --- | ---: | ---: | ---: |
| 2026-08-23 | 101 | 34 / 44 | 146 |
| 2026-08-30 | 105 | 35 / 44 | 146 |
| 2026-09-06 | 105 | 35 / 44 | 146 |
| 2026-09-13 | 105 | 35 / 44 | 146 |
| 2026-09-20 | 105 | 35 / 44 | 146 |
| 2026-09-27 | 105 | 35 / 44 | 146 |
| 2026-10-04 | 105 | 35 / 44 | 146 |

Develop contains 731 paired measurements. Release retains its September 9 EA 3.0
baseline, restricted to the requested matrix (105 measurements). EA 3.0 is a
snapshot label, not verification that the source commits are release tags.

## Supported matrix

State tasks retain every physics backend supported by their registered task,
including Kamino and Newton MJWarp/VBD where applicable. Camera tasks retain only:

- Isaac Sim PhysX + Isaac Sim RTX.
- Newton MJWarp (or the task's MJWarp/VBD variant) + Newton renderer.
- OV PhysX + OV RTX.
- Newton MJWarp (or MJWarp/VBD) + OV RTX.

Intersect these pairs with each task's registered selectors; unsupported pairs
are not reported as missing. There are 146 expected pairs across 44 browser tasks.
Other camera combinations are removed from both channels and the coverage matrix.

## Selection and provenance

Read-only export from `omni_runtime_isaac_lab_v3` in `ov-runtime-performance`
on `omniperf-trace.nvidia.com:5432`: 43,570 records on
`XEON_GOLD_5512U_1XRTXPRO6000_BW_SV`, ingested August 24 through October 5.
Only successful single-GPU, world-size-one WARM RSL-RL training entries with finite,
positive paired FPS and matching configured, task-metadata and measured environment
counts are eligible. Plotted counts must be 8,192.

Non-camera tasks use the headless profile. Camera tasks require explicit
`simple_shading_full_mdl` (Kuka: `simple_shading_full_mdl64`) and exclude dual-camera
runs. RGB and other counts are not substituted. Missing resolved camera dimensions
are flagged; preset names alone do not establish verified sensor dimensions.

Match successful WARM task metadata by task, workflow and entry-key variant.
Prefer database backend columns, then explicit per-entry preset selectors, then
matching successful WARM task presets. Normalize historical `physx`/`newton` physics
and renderer aliases. Group names and implicit defaults never establish identity.
Strip backend selectors and preserve sorted domain presets. Break ingestion-time
ties by the largest source record ID. Retain full numeric precision, standard
deviations, peaks, hardware, commits, original timestamps and identity sources.
Quality notes flag dirty source trees and FPS coefficients of variation above 25%.

## Current missing measurements

There are 41 missing pairs, including 16 Kamino state pairs and six Newton-renderer
camera pairs without the required simple-shading profile. RGB results exist for
those six camera pairs. They remain missing under the fixed comparison profile.

| Task | Missing physics / renderer |
| --- | --- |
| Isaac-Ant | newton_kamino/none |
| Isaac-Ant-Direct | newton_kamino/none |
| Isaac-Cartpole | newton_kamino/none |
| Isaac-Cartpole-Camera | newton_mjwarp/newton_renderer |
| Isaac-Cartpole-Camera-Direct | newton_mjwarp/newton_renderer |
| Isaac-Cartpole-Direct | newton_kamino/none |
| Isaac-Fourbar-Pole-Swingup | newton_kamino/none |
| Isaac-Lift-Cable-Franka | newton_mjwarp_vbd_proxy/none |
| Isaac-Lift-Cable-Franka-Camera | newton_mjwarp_vbd_proxy/newton_renderer, newton_mjwarp_vbd_proxy/ovrtx |
| Isaac-Lift-Cloth-Franka | isaacsim_physx/none, newton_mjwarp_vbd_proxy/none |
| Isaac-Lift-Cloth-Franka-Camera | isaacsim_physx/isaacsim_rtx, newton_mjwarp_vbd_proxy/newton_renderer, newton_mjwarp_vbd_proxy/ovrtx |
| Isaac-Lift-KukaAllegro-Camera | newton_mjwarp/newton_renderer |
| Isaac-Lift-Soft-Franka | isaacsim_physx/none, newton_mjwarp_vbd_proxy/none |
| Isaac-Lift-Soft-Franka-Camera | isaacsim_physx/isaacsim_rtx, newton_mjwarp_vbd_proxy/newton_renderer, newton_mjwarp_vbd_proxy/ovrtx |
| Isaac-Pendulum-MARL-Direct | isaacsim_physx/none, newton_kamino/none, newton_mjwarp/none, ovphysx/none |
| Isaac-Reorient-Cube-Shadow-Camera | newton_mjwarp/newton_renderer |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | newton_mjwarp/newton_renderer |
| Isaac-Reorient-KukaAllegro-Camera | newton_mjwarp/newton_renderer |
| Isaac-Shadow-Handover | isaacsim_physx/none, newton_mjwarp/none, ovphysx/none |
| Isaac-Velocity-Flat-AnymalD | newton_kamino/none |
| Isaac-Velocity-Flat-Cassie | newton_kamino/none |
| Isaac-Velocity-Flat-G1 | newton_kamino/none |
| Isaac-Velocity-Flat-H1 | newton_kamino/none |
| Isaac-Velocity-Flat-UnitreeGo2 | newton_kamino/none |
| Isaac-Velocity-Rough-AnymalD | newton_kamino/none |
| Isaac-Velocity-Rough-Cassie | newton_kamino/none |
| Isaac-Velocity-Rough-G1 | newton_kamino/none |
| Isaac-Velocity-Rough-H1 | newton_kamino/none |
| Isaac-Velocity-Rough-UnitreeGo2 | newton_kamino/none |

Pendulum MARL supports RL-Games/SKRL rather than RSL-RL; it needs a separately
identified training-library comparison. Database `Isaac-Shadow-Handover-Direct`
runs do not fill `Isaac-Shadow-Handover` gaps: these are separate registered tasks.
Missing measurements are not zeros, interpolations or renamed-task substitutions.

## Coverage exports

[coverage-configurations.csv](coverage-configurations.csv) lists all 1,022 expected
weekly configurations with paired coverage statuses, missing reasons, selected
source IDs and ingestion dates. `other_measured_counts` lists valid matching-profile
measurements at other counts. Missing measurements are not represented as zero FPS.

[coverage-environments.csv](coverage-environments.csv) summarizes expected and missing
pairs per task; [coverage-observed-profiles.csv](coverage-observed-profiles.csv)
contains the 731 selected measurements with both FPS metrics, quality notes and
source provenance. The older `coverage-data-issues.csv`,
`coverage-preset-selectors.csv` and `coverage-environment-defaults.csv` are historical
August 14–September 9 audits, not refreshed current coverage.
