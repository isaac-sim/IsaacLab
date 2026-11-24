* Added ``IsaacContrib-VLA-Heterogeneous-Franka``: a shared tabletop scene for vision-language-action policies
  on Newton, with bounded combinations from a tagged object catalog instead of fixed per-benchmark layouts.
  Sampling chooses categories uniformly before assets uniformly within each category, without enumerating combinations.
  Reset places objects collision-free inside the intersection of the table and a supplied reachable workspace;
  category and size requirements expose plate-placement and container-placement instructions only where evaluable.
  Objects use the pinned HeteroTask dataset, its local cache, or ``VLA_BENCHMARKS_ASSETS_DIR``.
