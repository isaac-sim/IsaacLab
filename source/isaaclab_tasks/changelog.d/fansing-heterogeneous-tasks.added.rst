* Added ``IsaacContrib-VLA-Heterogeneous-Franka``: a shared tabletop scene for vision-language-action policies
  on Newton, with bounded combinations from a tagged object catalog instead of fixed per-benchmark layouts.
  Sampling chooses categories uniformly before assets uniformly within each category, without enumerating combinations.
  The first object of a world is a tray or bowl and the second a prop that fits it, so every world offers a task.
  Reset places objects collision-free inside the polygon intersection of the declared, measured tabletop outline and a
  hand-defined robot workspace, preserving outlines and holes. Placement samples COM poses using
  collision geometry relative to the simulator COM, accounting for mesh and COM offsets from the asset origin.
  The nine real table assets retain
  their actual collisions, without a box proxy. Reset jointly samples a different table and feasible object placement, retrying smaller tables
  that cannot fit the sampled objects. Table variants switch on reset with their matching
  poses and measured surfaces; category and size requirements expose tray- and container-placement
  instructions only where evaluable: the object fits its target and the gripper. Instructions name objects in
  plain words without asset codes, the bowls by their color, and stand for every object and target of the names
  they use. A placement counts when the target carries the released object at rest.
  Camera and object collection constants compose directly through config copies and replacements, without
  config factories or post-init scene generation. The robot preserves its original home pose, arm impedance and
  disabled self-collisions, preventing internal collision hulls from deforming the open Robotiq linkage.
  The flat catalog references 100 NVIDIA-hosted SimReady props, with large assets restricted to sparse worlds through
  inclusion tags. Newton targets 75% fewer collision triangles and generates native SDFs while preserving the
  original visual meshes and receiver cavities. The known failing large plastic plate is excluded.
  Four solver substeps with refreshed contacts and a shared gripper/object contact response reduce finger penetration
  during closing, without changing actuator gains or the robot's initial pose.
  Table and stand assets continue to use the pinned HeteroTask dataset, its local cache, or ``VLA_BENCHMARKS_ASSETS_DIR``.
