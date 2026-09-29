Added
^^^^^

* Added ``FRANKA_PANDA_FLAT_CFG`` and ``FRANKA_PANDA_FLAT_HIGH_PD_CFG`` for the shared flat Franka
  asset. These configurations selected gripper-only colliders and the ``panda_arm`` actuator group;
  select the ``Physics`` variant for the target backend.

Deprecated
^^^^^^^^^^

* Deprecated ``FRANKA_PANDA_CFG``, ``FRANKA_PANDA_HIGH_PD_CFG``, and
  ``FRANKA_PANDA_MENAGERIE_CFG`` for removal in Isaac Lab 4.0. Their USD paths and actuator groups
  remain available during deprecation. **Breaking:** the published ``franka_panda.usda`` now contains
  the flat asset; consumers needing the old nested hierarchy can explicitly select
  ``franka_panda_nestedInstance.usda``. Migrate maintained tasks to the corresponding
  ``FRANKA_PANDA_FLAT_*`` config, replace shoulder/forearm actuator overrides with ``panda_arm``,
  and select the backend's ``Physics`` variant.
