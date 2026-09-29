Added
^^^^^

* Added ``FRANKA_PANDA_FLAT_CFG`` and ``FRANKA_PANDA_FLAT_HIGH_PD_CFG`` for the shared flat Franka
  asset. These configurations selected gripper-only colliders and the ``panda_arm`` actuator group;
  select the ``Physics`` variant for the target backend.

Deprecated
^^^^^^^^^^

* Deprecated ``FRANKA_PANDA_CFG``, ``FRANKA_PANDA_HIGH_PD_CFG``, and
  ``FRANKA_PANDA_MENAGERIE_CFG`` for removal in Isaac Lab 4.0. The legacy Panda USD and actuator
  groups remain available through the first two names. **Breaking:** the nested Menagerie USD moved
  from ``franka_panda.usda`` to ``franka_panda_nestedInstance.usda``; code that reads or overrides
  its path must use the new filename. Migrate to the corresponding ``FRANKA_PANDA_FLAT_*`` config,
  replace shoulder/forearm actuator overrides with ``panda_arm``, and select the backend's ``Physics`` variant.
