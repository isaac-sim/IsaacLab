# Environment benchmark snapshot

Release and Develop load byte-identical CSV exports from
`omni_runtime_isaac_lab_v3` in `ov-runtime-performance` on
`omniperf-trace.nvidia.com:5432`. Credentials are not stored here.
The legacy `environment-performance.csv` is retained as an older snapshot.

The shared **September 9, 2026** documentation baseline contains **115 measured
configurations across 36 CORE tasks**: 28 Collection and 87 Training, including
34 camera measurements. Release displays EA 3.0; Develop displays September 9.
These are the same observations, not independent release/develop experiments.

## Selection

Queried the complete August 17–September 9 ingestion window, inclusive. No single
day was complete, so selected the most recent eligible measurement **per exact
configuration**, preferring later `created_at`, then larger source record ID.
76 values come from September 9; 39 come from August 24, 26, or 27 ingestions.
Older camera measurements are provisional release documentation coverage.

Eligibility requires:

- `machine_kind = XEON_GOLD_5512U_1XRTXPRO6000_BW_SV`, one GPU, world size 1,
  successful run, and finite positive WARM `runtime["Mean Total FPS"]`.
- CORE runtime/training task; actual `benchmark_info.num_envs` equals both
  `runtime.num_envs` and the configured training-default task count. Collection
  uses the same count as Training. Group names do not override measured counts.
- One matching successful WARM task metadata entry for the task, workflow, and
  entry-key variant. Its preset identifies the domain profile; remove physics
  and renderer selectors and sort the remaining selectors.
- Supported physics backend; no Kamino or incompatible Isaac Sim PhysX/OV RTX
  and OV PhysX/Isaac Sim RTX pairs. Camera measurements require an explicit
  supported database renderer. Missing renderer rows remain in the audit only.

Series identity is task, workload, physics, renderer, domain presets, measured
count, RL library, and recorded camera resolution. Different configurations
are never averaged or merged. Non-camera runs with explicit renderers retain
their renderer as a distinct configuration.

Historical aliases are normalized: `physx` → `isaacsim_physx`, `newton` →
`newton_mjwarp` (cross-checked with session presets), `IsaacRtxRenderer` →
`isaacsim_rtx`, `NewtonWarpRenderer`/`newton` renderer → `newton_renderer`.
Missing renderer metadata is not inferred from a requested preset.

## Fields and interpretation

- `snapshot_date_utc` is the shared graph position, **not the measurement date**.
- `benchmark_date_utc` and `recorded_at_utc` preserve database ingestion time.
- `measurement_timestamp` preserves the original timestamp without assuming a
  timezone; its date appears in the FPS table and chart tooltip.
- `source_record_id`, `source_entry_key`, and `comparison_group_name` locate the
  source record and WARM entry. `git_commit` preserves the measured base commit.
- FPS, VRAM, utilization, and iterations come from the same WARM runtime object;
  hardware and versions come from the corresponding hardware/version objects.
  Driver comes from session metadata. Missing values remain empty.
- `camera_resolution` preserves the session's explicit camera-resolution field
  when present. It is not inferred from presets or comparison-group names.
- `quality_notes` flags older fallback measurements, dirty-source metadata,
  unresolved camera metadata, and FPS coefficient of variation above 25%.

Collection is total FPS from the runtime workflow; Training is total FPS from
training, including policy updates. Training's separate collection-only FPS
is not substituted for runtime data. All selected training measurements use
RSL-RL; this does not establish support/coverage for every advertised RL library.

Defaults: most tasks and Kuka camera tasks use 4,096; cube reorientation 8,192;
Shadow camera 1,225; Cartpole camera 512; handover 2,048; soft/cloth Franka 2,048;
cable Franka 8,192; deformable Franka camera variants 128. The complete per-task
counts and missing backend/renderer/workload combinations are in the report.

## Audit files

- [Coverage report](COVERAGE.md): findings, complete minimum matrix, and collection plan.
- [Environment-default flags](coverage-environment-defaults.csv): explicit missing
  Collection/Training flags for all 44 tasks and available alternative counts.
- [Configuration coverage](coverage-configurations.csv): all 312 expected pairs
  and workloads, selected record IDs/dates, and available alternative counts.
- [Observed profiles](coverage-observed-profiles.csv): missing workload counterparts.
- [Preset selectors](coverage-preset-selectors.csv): explicit individual preset
  coverage; not a declaration that every Cartesian combination is compatible.
- [Data issues](coverage-data-issues.csv): source IDs/entry keys for count-label
  conflicts and missing physics/camera renderers on the target hardware.

The graph does not interpolate missing values or claim that these develop commits
are verified release tags. Both channels intentionally share this baseline.
