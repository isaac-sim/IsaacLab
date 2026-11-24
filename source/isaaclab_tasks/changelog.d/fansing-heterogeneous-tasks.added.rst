* Added ``IsaacContrib-VLA-Heterogeneous-Franka``: a shared tabletop scene for vision-language-action policies
  on Newton, with bounded combinations from a tagged object catalog instead of fixed per-benchmark layouts.
  Sampling chooses categories uniformly before assets uniformly within each category, without enumerating combinations.
  Reset places objects collision-free inside the polygon intersection of the declared, measured tabletop outline and a
  supplied reachable workspace, preserving outlines and holes. The actual table has mesh collisions, without
  a box proxy; category and size requirements expose plate-placement and container-placement instructions only where evaluable.
  Camera and object collection constants compose directly through config copies and replacements, without
  config factories or post-init scene generation. Objects use the pinned HeteroTask dataset, its local cache,
  or ``VLA_BENCHMARKS_ASSETS_DIR``.
