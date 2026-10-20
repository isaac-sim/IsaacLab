* Fixed :class:`~isaaclab.sensors.MultiMeshRayCaster` and :class:`~isaaclab.sensors.MultiMeshRayCasterCamera` giving
  every environment the source environment's copy of a static environment-local target
  (``track_mesh_transforms=False``) on kitless backends, which author only the source on the USD stage. Static
  targets now resolve through the clone plan like tracked targets: each environment holds its own copy, and in a
  heterogeneous scene only the targets of its own prototype.
  A static target that names one environment's prim, such as ``/World/envs/env_0/Crate``, now belongs to that
  environment only, as a tracked target does; it used to be given to every environment.
