# Environment benchmark snapshots

The environment browser loads `environment-performance-release.csv` and
`environment-performance-develop.csv` independently. The original
`environment-performance.csv` is retained as a previous snapshot.

Source: `omni_runtime_isaac_lab_v3` in the `ov-runtime-performance` database on
`omniperf-trace.nvidia.com:5432`. No credentials are stored here.

The release snapshot uses the database ingestion date September 9, 2026 and is
displayed as **EA 3.0**, as requested. This label identifies the documentation
snapshot; it does not assert that the measured commit is a release tag.
Develop uses ingestion dates August 17, August 24, August 31, and September 7.

Selection criteria:

- `machine_kind = XEON_GOLD_5512U_1XRTXPRO6000_BW_SV`, `num_gpus = 1`,
  `omniperf_type = success`, and SQL `renderer IS NULL`.
- Comparison groups `ISAAC_LAB_SINGLE_GPU_GTL_PRESET_PHYSX_E16384`,
  `ISAAC_LAB_SINGLE_GPU_GTL_PRESET_NEWTON_MJWARP_E16384`, and
  `ISAAC_LAB_SINGLE_GPU_GTL_PRESET_OVPHYSX_E16384`.
- `WARM` KPI entries with 16,384 environments and world size 1.
- Runtime benchmarks or RSL-RL training benchmarks with positive mean total FPS.
- Latest `created_at`, then largest record ID, per ingestion date, task,
  workload, and physics backend. Missing measurements are not interpolated.

`benchmark_date_utc` and `recorded_at_utc` come from database `created_at`.
`measurement_timestamp` preserves the original benchmark timestamp verbatim
(it has no timezone); `source_record_id` identifies the source database row.
FPS fields use runtime `Mean/Std/Max Total FPS` for both workloads. Hardware,
memory, utilization, iteration counts, seed, and version fields come from the
same WARM entry; the driver comes from session metadata. Missing fields remain
empty. `rendering_backend=none` denotes no explicit renderer in the selected
comparison group, not a guarantee that a camera task performs no rendering.

Only `Isaac-Velocity-Rough-G1` training on PhysX and Newton MJWarp has matching
August 17 records. Other curves therefore have fewer than four measurements.
Kamino has no matching records in these comparison groups.
