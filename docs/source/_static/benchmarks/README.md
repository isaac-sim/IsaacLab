# Environment benchmark exports

The environment browser reads `environment-performance-release.csv` and
`environment-performance-develop.csv`. Each measured row supplies Collection FPS
(rollouts and policy inference) and Training FPS (including policy updates), with
standard deviations, peaks and provenance from the same successful WARM entry.

- Release: September 9 EA 3.0 baseline (111 pairs) and October 12 GA 3.0
  snapshot (133 pairs), 244 measurements total.
- Develop (default tab): eight weekly Sunday positions from August 23 through
  October 11, 905 measurements. Selection windows end Monday, including October 12.
  Sunday positions are presentation dates; actual timestamps are retained.


All plotted runs use 8,192 environments and the same single-GPU hardware.
Runs use RSL-RL except for the October 12 Pendulum MARL measurements, which use
its supported RL-Games workflow. The library is retained in each row and tooltip.
Camera comparisons use a single camera with `rgb` for the Newton renderer and
`simple_shading_full_mdl` for OV RTX and Isaac Sim RTX (`rgb64` and
`simple_shading_full_mdl64` for Kuka tasks). Each pair retains its actual profile
in the exported presets and chart tooltips. State tasks include every registered physics backend;
Kamino remains excluded from camera comparisons.
Camera tasks include only PhysX/Isaac Sim RTX, Newton/Newton renderer,
OV PhysX/OV RTX and Newton/OV RTX, intersected with supported task selectors.
Missing data is never zeroed, interpolated or copied across windows.

Source database:
`omni_runtime_isaac_lab_v3` in `ov-runtime-performance` on
`omniperf-trace.nvidia.com:5432`. Credentials are not stored here.

`environment-performance.csv` is an unused legacy snapshot. The data-issues,
preset-selectors and environment-defaults coverage files are historical audits;
current coverage is in the configurations, environments and observed-profiles
exports. The release display label does not verify release-tag source commits.

## October 12 refresh

The read-only query selected `2026-10-12 <= created_at < 2026-10-13` on
`XEON_GOLD_5512U_1XRTXPRO6000_BW_SV`: 958 records through ID 278131. Selection
requires successful WARM task metadata matching task, workflow and variant;
one GPU and world size 1; matching configured/runtime counts of 8,192; and finite
positive Collection and Total FPS from the same entry. Latest ingestion timestamp
wins per task/physics/renderer, with record ID as the tiebreaker. Explicit database
backend identity takes precedence over entry presets and matching task presets.

The 133 selected pairs cover 40 of 44 tasks, including 14 Kamino state-task pairs,
Cable and Cloth with Newton MJWarp VBD proxy, and all four Pendulum MARL physics
backends. The same measurements populate the October 11 develop position and
October 12 GA 3.0 release position. Earlier snapshots remain unchanged.

Thirteen supported pairs remain missing. These include the Soft-Franka tasks,
Cable/Cloth camera tasks, PhysX Cloth state task, and Kamino Flat/Rough G1.
Available Cable/Cloth camera runs use 128×128 with unspecified shading presets;
they do not qualify for the fixed 64×64 comparison. Missing data is not backfilled.
Resolved camera dimensions remain empty when metadata does not record them.
Dirty-source and FPS variability above 25% are flagged in `quality_notes`.
