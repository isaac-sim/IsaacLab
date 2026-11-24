* Added ``IsaacContrib-VLA-Heterogeneous-Franka``: a shared tabletop scene for vision-language-action policies
  on Newton, with bounded combinations from a tagged object catalog instead of fixed per-benchmark layouts.
  Sampling chooses categories uniformly before assets uniformly within each category, without enumerating combinations.
  Reset places objects collision-free inside the polygon intersection of the declared, measured tabletop outline and a
  hand-defined robot workspace, preserving outlines and holes. Placement samples COM poses using
  collision geometry relative to the simulator COM, accounting for mesh and COM offsets from the asset origin.
  The nine real table assets retain
  their actual collisions, without a box proxy. Reset jointly samples a different table and feasible object placement, retrying smaller tables
  that cannot fit the sampled objects. Table variants switch on reset with their matching
  poses and measured surfaces; category and size requirements expose plate-placement and container-placement instructions only where evaluable.
  Camera and object collection constants compose directly through config copies and replacements, without
  config factories or post-init scene generation. The robot preserves its original home pose, arm impedance and
  disabled self-collisions, preventing internal collision hulls from deforming the open Robotiq linkage. Objects use the pinned
  HeteroTask dataset, its local cache, or ``VLA_BENCHMARKS_ASSETS_DIR``.
