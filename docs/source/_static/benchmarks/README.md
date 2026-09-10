# Environment benchmark snapshots

The browser loads `environment-performance-release.csv` and
`environment-performance-develop.csv`. Both were exported directly from
`omni_runtime_isaac_lab_v3` in the `ov-runtime-performance` database on
`omniperf-trace.nvidia.com:5432`. No credentials are stored here.
The old `environment-performance.csv` is retained as a previous snapshot.

Release uses the September 9, 2026 ingestion date and is displayed as **EA 3.0**.
This is the requested documentation snapshot label, not a claim that its measured
commits are release tags. Develop uses August 17, August 24, August 31, and September 7.

## Selection and provenance

- Selected `machine_kind = XEON_GOLD_5512U_1XRTXPRO6000_BW_SV`,
  `num_gpus = 1`, and `omniperf_type = success`.
- Expanded every entry under `kpis.WARM`; kept CORE tasks with world size 1,
  runtime or training workflows, positive mean total FPS, and the task's
  configured training-default environment count. Used that same per-task count
  for Collection and Training, with no fallback to another count.
- Included all comparison groups, explicit renderers, and measured task presets
  matching that count. Counts are retained in the CSV for provenance but omitted
  from the legend. A task/workload without matching results shows the unavailable state.
- Excluded Kamino and incompatible Isaac Sim PhysX/OV RTX and
  OV PhysX/Isaac Sim RTX pairs.
- Selected latest `created_at`, then largest source record ID, per ingestion date,
  task, workload, physics backend, renderer, task presets, environment count,
  and RL library. Different configurations remain separate chart series.

`benchmark_date_utc` and `recorded_at_utc` come from database `created_at`.
`measurement_timestamp` preserves the original benchmark timestamp verbatim
(it has no timezone). `source_record_id`, `source_entry_key`, and
`comparison_group_name` identify the source row, WARM entry, and comparison group.

FPS, VRAM, and utilization come from the same WARM entry's `runtime` metrics;
hardware and versions come from that entry's `hardware_info` and `version_info`.
Environment count and seed come from `benchmark_info`, not the comparison-group
name: some historical groups requested a different count than was actually measured.
The driver comes from session metadata. Missing values remain empty.

Task presets come from matching successful WARM task metadata (including the
entry's variant), falling back to the session preset when necessary. Physics and
renderer selectors are removed and domain presets are sorted for a stable series
identity. RL library is taken from the WARM entry key; all selected training
measurements use RSL-RL.

Historical backend names are normalized: `physx` to `isaacsim_physx`,
`newton` physics to `newton_mjwarp` (verified against session presets),
`IsaacRtxRenderer` to `isaacsim_rtx`, and `NewtonWarpRenderer`/`newton`
renderers to `newton_renderer`. Camera records without an explicit renderer
remain labeled “Unspecified renderer”; their renderer is not guessed.

## Coverage

Release contains 76 measurements across 30 tasks; develop contains 225 across
36 tasks. Release has no camera measurements at the default counts.

Defaults were checked against the training configurations under
`source/isaaclab_tasks/isaaclab_tasks/core/`, excluding play-mode overrides:
most tasks and KukaAllegro camera tasks use 4,096; cube reorientation uses 8,192
(1,225 for Shadow camera tasks); Cartpole camera uses 512; handover uses 2,048;
soft/cloth Franka uses 2,048; cable Franka uses 8,192; and all three deformable
Franka camera variants use 128. The coverage table records the count for each task.

Selection uses the measured count, not just groups named `EDEFAULT`: a fixed-count
run that matches the configured default is eligible. Unmeasured defaults remain unavailable.

[CORE coverage](COVERAGE.md) lists all missing task/backend/renderer/workload
combinations and the queried database window. Missing points are not interpolated.
The previously recovered camera file was replaced by these traceable database
exports; neither channel depends on a local documentation-build artifact.
