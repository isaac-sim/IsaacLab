* Added G1 platform, ramp, and granular terrain demonstrations with pretrained depth and SONIC policies,
  automatic checkpoint downloads, and RTX rendering through ``examples/g1_terrain.py``
  and ``isaaclab demo sonic``.
  Supported Isaac Sim PhysX and Newton MJWarp, with PBD and MPM respectively for granular terrain.
  Selected courses through a shared task preset and declared terrain materials and collision properties
  with standard spawner configurations.
  Kept pretrained depth and SONIC policy inference in the task's ``agents`` package.
  Shared spectator setup across the entry points, derived SONIC joint mappings from named joints,
  and used the launcher's selected CUDA device for PhysX particles and SONIC inference.
