# Benchmark coverage — 8,192 environments

All plotted Collection and Training benchmarks use **num_envs = 8192**.
Camera rows use only `simple_shading_full_mdl`, with 64×64 images and a single
camera (Kuka uses the equivalent `simple_shading_full_mdl64` preset). RGB, dual-camera,
and unspecified camera profiles are excluded. Non-camera tasks use headless runs.
All selected runs use RSL-RL. Select the latest eligible run per task/physics/renderer
and snapshot; full source presets remain in the CSV and tooltips. These fixed
profiles give exactly one series and one table row per physics/renderer pair.

## Snapshot coverage

| Snapshot | Paired records | Tasks | Backend/renderer pairs |
| --- | ---: | ---: | ---: |
| August 28 | 114 | 36 / 44 | 114 / 156 |
| Current (September 9 baseline) | 114 | 36 / 44 | 114 / 156 |

Release contains 114 current records. Develop contains 228 records: August 28
plus the identical current records. Each record supplies both Collection FPS
(`Mean Collection FPS`, stepping plus inference) and Training FPS (`Mean Total FPS`,
including policy updates). Their coverage is identical; separate runtime runs are
not required. Standard deviation and peak values remain in the data.

The figure labels the environment count beside Release/Develop. The compact table shows only the latest
available FPS per physics + renderer pair, rounded to an integer. Full configurations and
measurement dates remain in tooltips, and the legend appears above the graph.
Missing supported combinations retain “Not available”; entirely absent tasks use
the benchmark-data-unavailable placeholder.

August 28 requires an exact ingestion-date match. Current selects the latest
eligible record per physics/renderer pair through September 9. All 114 current
records are from September 9; there are no older fallbacks under this profile.
August 14 remains omitted as requested.

## Why 8,192

Before restricting camera profiles, the same hardware, validity, and date filters
gave this current coverage across all recorded camera variants:

| Count policy | Tasks | Backend/renderer pairs | Recorded configurations |
| --- | ---: | ---: | ---: |
| Task defaults | 36 | 70 | 87 |
| 4,096 | 27 | 52 | 61 |
| **8,192** | **36** | **126** | **209** |
| 16,384 | 34 | 110 | 149 |

Combining all measured counts also covers only 126 distinct pairs before the
camera-profile restriction. Using 8,192 adds 56 pairs over task defaults without
losing any task. Restricting to simple shading now selects 114 pairs, including
36 ovphysx pairs at each date. The omitted 12 pairs had RGB Newton-renderer runs
but no matching simple-shading measurement in the queried data.

## Missing data

There are **42 missing backend/renderer pairs out of 156** at each snapshot,
for both metrics. Of these, 24 pairs belong to eight entirely absent tasks and
18 are Newton-renderer camera pairs without matching simple-shading runs.
Six of those 18 were already unavailable before the profile restriction;
the restriction removes 12 additional RGB-only pairs.
The latter retain “Not available”; RGB measurements are not substituted.
No WARM runtime/training entries for the eight absent task names were found on
any queried machine in the August 14–September 9 window. Kamino, the Kamino-only
Fourbar task, and incompatible physx/ovrtx and ovphysx/Isaac Sim RTX pairs are excluded.

| Task | Missing backend/renderer combinations |
| --- | --- |
| Isaac-Cartpole-Camera | isaacsim_physx/newton_renderer, newton_mjwarp/newton_renderer, ovphysx/newton_renderer |
| Isaac-Cartpole-Camera-Direct | isaacsim_physx/newton_renderer, newton_mjwarp/newton_renderer, ovphysx/newton_renderer |
| Isaac-Lift-Cable-Franka | newton_mjwarp_vbd_proxy |
| Isaac-Lift-Cable-Franka-Camera | newton_mjwarp_vbd_proxy/isaacsim_rtx, newton_mjwarp_vbd_proxy/newton_renderer, newton_mjwarp_vbd_proxy/ovrtx |
| Isaac-Lift-Cloth-Franka | isaacsim_physx, newton_mjwarp_vbd_proxy |
| Isaac-Lift-Cloth-Franka-Camera | isaacsim_physx/isaacsim_rtx, isaacsim_physx/newton_renderer, newton_mjwarp_vbd_proxy/isaacsim_rtx, newton_mjwarp_vbd_proxy/newton_renderer, newton_mjwarp_vbd_proxy/ovrtx |
| Isaac-Lift-KukaAllegro-Camera | isaacsim_physx/newton_renderer, newton_mjwarp/newton_renderer, ovphysx/newton_renderer |
| Isaac-Lift-Soft-Franka | isaacsim_physx, newton_mjwarp_vbd_proxy |
| Isaac-Lift-Soft-Franka-Camera | isaacsim_physx/isaacsim_rtx, isaacsim_physx/newton_renderer, newton_mjwarp_vbd_proxy/isaacsim_rtx, newton_mjwarp_vbd_proxy/newton_renderer, newton_mjwarp_vbd_proxy/ovrtx |
| Isaac-Pendulum-MARL-Direct | isaacsim_physx, newton_mjwarp, ovphysx |
| Isaac-Reorient-Cube-Shadow-Camera | isaacsim_physx/newton_renderer, newton_mjwarp/newton_renderer, ovphysx/newton_renderer |
| Isaac-Reorient-Cube-Shadow-Camera-Direct | isaacsim_physx/newton_renderer, newton_mjwarp/newton_renderer, ovphysx/newton_renderer |
| Isaac-Reorient-KukaAllegro-Camera | isaacsim_physx/newton_renderer, newton_mjwarp/newton_renderer, ovphysx/newton_renderer |
| Isaac-Shadow-Handover | isaacsim_physx, newton_mjwarp, ovphysx |

[Configuration matrix](coverage-configurations.csv) lists all 156 pairs for each
snapshot, statuses for both metrics, selected source IDs/dates, and other counts.
[Environment coverage](coverage-environments.csv) flags missing 8,192 runs per task.
[Observed profiles](coverage-observed-profiles.csv) preserves full configurations.
[Preset selectors](coverage-preset-selectors.csv) flags explicitly recorded domain
selectors; it does not assert that every Cartesian combination is compatible.
[Task-default audit](coverage-environment-defaults.csv) retains the earlier default-count
flags for reference only; it does not describe the current plotted count policy.

## Data quality and provenance

The query covers 63,631 database records from
`2026-08-14 <= created_at < 2026-09-10`. Selected hardware is
`XEON_GOLD_5512U_1XRTXPRO6000_BW_SV`: one RTX PRO 6000 Blackwell Server Edition GPU,
world size 1. Only successful WARM training entries with positive finite paired
FPS metrics, matching configured/measured counts of 8,192, supported backends,
and unambiguous task metadata are selected.

The [data-issues audit](coverage-data-issues.csv) covers training entries on that
hardware throughout the query window. Its counts overlap: 3,696 entries lack the
database physics column, 1,598 camera entries lack the database renderer column,
182 have misleading physics group labels, and 182 have misleading count labels.
Explicit per-entry presets or matching successful WARM task presets can recover
backend identity. The CSV records that evidence; group labels and implicit defaults
are not used to invent identities. Unresolved camera renderers remain excluded.

Per-record `quality_notes` flag dirty-source metadata, older fallbacks, missing
camera dimensions/domain metadata, and FPS coefficients of variation above 25%.
These observations are not guaranteed clean release-source reproductions.
Original source IDs, commits, measured iterations, and timestamps remain in the CSV.
Dates are ingestion dates interpreted as UTC; original measurement timestamps lack
timezones and are preserved verbatim. EA 3.0 is the release display label, not an
assertion that measured commits are release tags.

## Remaining work

1. Collect missing supported pairs at 8,192 environments and the fixed camera
   profile where supported. Use each task's
   supported training library. Pendulum MARL requires RL-Games or SKRL, not RSL-RL.
2. Collect clean release-source runs and record resolved
   physics, renderer, camera dimensions/types/count, domain presets, and dependencies.
3. Repeat noisy runs with longer measurement windows. Keep unavailable placeholders
   until valid paired measurements exist.
