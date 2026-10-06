# Environment benchmark exports

The environment browser reads `environment-performance-release.csv` and
`environment-performance-develop.csv`. Each measured row supplies Collection FPS
(rollouts and policy inference) and Training FPS (including policy updates), with
standard deviations, peaks and provenance from the same successful WARM entry.

- Release: September 9 EA 3.0 baseline, 105 pairs after camera-matrix filtering.
- Develop: seven weekly Sunday positions from August 23 through October 4,
  731 measurements. Selection windows end Monday, including September 28 and
  October 5. Sunday positions are presentation dates; actual timestamps are retained.


All plotted runs use 8,192 environments, RSL-RL, the same single-GPU hardware and
fixed camera profile. State tasks include every registered physics backend.
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
