* Added ``IsaacContrib-TableTop-PickPlace-Franka``: a shared tabletop scene for vision-language-action policies
  on Newton, with bounded combinations from a tagged object catalog instead of fixed per-benchmark layouts.
  Sampling chooses categories uniformly before assets uniformly within each category, without enumerating combinations.
  Every slot offers the complete catalog and a non-colliding placeholder. Every reset resamples the object count,
  category composition and concrete assets, including a size-compatible pair for an evaluable task. Slot roles
  are interchangeable, and any environment can request any globally indexed task family.
  Geometry uses SAH BVHs to keep dense reset-time mesh-overlap queries stable.
  Reset places objects collision-free inside the polygon intersection of the declared, measured tabletop outline and a
  hand-defined robot workspace, preserving outlines and holes. Batched footprint/boundary checks use
  tensor operations without a task dependency on Shapely. Placement samples COM poses using
  collision geometry relative to the simulator COM, accounting for mesh and COM offsets from the asset origin.
  Nine explicit NVIDIA-hosted SimReady table assets retain
  their actual geometry and triangle collisions, without a box proxy. Articulated table equipment is frozen
  at its authored assembly pose into a single kinematic table body. Measured usable panels exclude mounting
  holes, grommets and elevated shelves. Reset jointly samples a different table and feasible object placement, retrying smaller tables
  that cannot fit the sampled objects. Table variants switch on reset with their matching
  poses and measured surfaces; category and size requirements expose tray- and container-placement
  instructions only where evaluable: the object fits its target and the gripper. Instructions name objects in
  plain words without asset codes, the bowls by their color, and stand for every object and target of the names
  they use. A placement counts when the target carries the released object at rest.
  Scene-authored dome and distant lights supply Newton camera illumination through builder translation.
  Camera and object collection constants compose directly through config copies and replacements, without
  config factories or post-init scene generation. The robot preserves its original home pose, arm impedance and
  disabled self-collisions, preventing internal collision hulls from deforming the open Robotiq linkage.
  The flat catalog references 100 NVIDIA-hosted SimReady props. Infeasible placements retry concrete compositions
  while preserving the sampled cardinality. Ordinary USD configs and collision fragments declare the catalog; backend-owned
  ``sdf_mesh_reduction=0.75`` targets 75% fewer collision triangles and generates native SDFs while preserving the
  original visual meshes and receiver cavities. The known failing large plastic plate is excluded.
  Four solver substeps with refreshed contacts and a shared gripper/object contact response reduce finger penetration
  during closing, without changing actuator gains or the robot's initial pose.
  The robot support reuses the stainless SimReady table. Tables, support and movable objects need no
  Hugging Face dataset, custom uploads or ``VLA_BENCHMARKS_ASSETS_DIR`` resolver.
