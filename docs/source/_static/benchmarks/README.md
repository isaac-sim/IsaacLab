# Paired environment benchmark snapshots

Both graph modes read the same WARM training records from
`omni_runtime_isaac_lab_v3` in `ov-runtime-performance` on
`omniperf-trace.nvidia.com:5432`. Credentials are not stored here.

- **Collection** uses `collection_fps_mean` (`Mean Collection FPS`): rollouts,
  including environment stepping and policy inference.
- **Training** uses `total_fps_mean` (`Mean Total FPS`): end-to-end training,
  including policy updates.

Each CSV row contains both metrics and their own standard deviation and peak.
`workload=training` identifies the source workflow; it is not a graph-mode filter.
Standalone runtime measurements are not mixed into either mode.

## Snapshots

- Release: 114 current records across 36 environments, labeled EA 3.0.
- Develop: August 28 and September 9, with 228 records total. August 14 remains
  omitted. August 28 has 114 records across 36 environments. The current 114 rows
  are identical to Release.

Historical dates use exact database ingestion dates. The current baseline selects
latest eligible measurements through September 9 per physics/renderer pair.
All 114 current records are from September 9. The graph shows the
two available snapshot dates. Missing values are not zeroed,
interpolated, or backfilled into historical snapshots.

## Selection contract

- Query August 14 through September 9, inclusive, using read-only access.
- Select successful WARM training records on
  `XEON_GOLD_5512U_1XRTXPRO6000_BW_SV`, one GPU and world size 1.
- Require finite positive Collection and Total FPS; matching
  `benchmark_info.num_envs` and `runtime.num_envs`, both equal to **8,192**.
  Neither mode substitutes a different environment count.
- Match successful WARM task metadata by task, workflow, and entry-key variant.
- Prefer explicit database physics/renderer columns. If absent, use an explicit
  selector in the WARM entry's `benchmark_info.presets`, then the matching
  successful WARM task preset. Record the source in `physics_backend_source` and
  `rendering_backend_source`. Never infer missing identity from group names or
  implicit defaults. Camera renderers must be identifiable.
- Normalize historical `physx`/`newton` physics to `isaacsim_physx`/`newton_mjwarp`,
  and `IsaacRtxRenderer`/`NewtonWarpRenderer`/`newton` renderers to their canonical
  selectors. Exclude Kamino and incompatible PhysX/renderer pairs.
- Use only RSL-RL, headless non-camera tasks, and 64×64 single-camera
  `simple_shading_full_mdl` runs (including Kuka's equivalent `simple_shading_full_mdl64`).
  Do not substitute RGB, dual-camera, or unspecified camera profiles.
- Keep one record per task/physics/renderer and snapshot. Prefer latest ingestion
  timestamp, then largest record ID.
  The two metrics always retain the exact same source record and WARM entry.

Full domain presets come from per-entry metadata when available, otherwise the
matching task preset. Strip physics/renderer selectors and sort remaining tokens.
Missing resolved camera dimensions stay empty; preset names do not establish
verified sensor dimensions.

## Provenance and audit

`snapshot_date_utc` is the graph position. `benchmark_date_utc` and
`recorded_at_utc` preserve actual ingestion time. `measurement_timestamp`
preserves the original timestamp without assigning a timezone. Mean/std/peak
FPS, VRAM, utilization, hardware, versions, source IDs, entry keys, and measured
base commits are retained. `quality_notes` flags older fallbacks, dirty source,
unresolved camera details, and variability above 25% for each metric separately.

The shared current baseline is assembled from develop runs; EA 3.0 is a snapshot
label, not a verified release-commit claim. The legacy `environment-performance.csv`
is retained as an older unused snapshot.

- [Coverage report](COVERAGE.md): corrected interpretation, dates, OV PhysX,
  missing 8,192 runs, full current pair matrix, and collection plan.
- [Environment coverage](coverage-environments.csv): per-task/per-date 8,192-run flags.
- [Environment defaults](coverage-environment-defaults.csv): retained earlier default-count
  audit, independent of the current plotted count policy.
- [Configuration coverage](coverage-configurations.csv): all 156 pairs per displayed date,
  statuses for both metrics, alternative counts, and source IDs/dates.
- [Observed profiles](coverage-observed-profiles.csv): selected paired configurations.
- [Preset selectors](coverage-preset-selectors.csv): explicitly recorded individual
  selectors, not a validated Cartesian compatibility matrix.
- [Data issues](coverage-data-issues.csv): training-entry metadata problems and
  recovered identities across the queried window.

The compact FPS table shows only the latest available value per configuration in
both channels, rounded to a whole number and labeled as physics + renderer.
Actual measurement dates and detailed configurations remain in tooltips. The
legend above the graph uses the same labels; the count sits beside Release/Develop; the CSV retains full precision.
