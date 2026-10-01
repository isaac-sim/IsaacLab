* **Breaking:** changed ``FRANKA_PANDA_CFG`` and ``FRANKA_PANDA_HIGH_PD_CFG`` to use the shared
  flat Franka asset with gripper-only collisions, a single ``panda_arm`` actuator group,
  and a driven leader finger with a passive mimic follower.
  Use ``FRANKA_PANDA_LEGACY_CFG`` and ``FRANKA_PANDA_LEGACY_HIGH_PD_CFG`` to retain the
  original USD path, shoulder/forearm actuator groups, gains, and collision settings.
  Select the ``Physics`` variant for the backend when using the main configs.
* **Breaking:** the published ``franka_panda.usda`` now contains the flat asset. Consumers
  needing the former nested hierarchy can explicitly select ``franka_panda_nestedInstance.usda``.
