Added
^^^^^

* Added ``FRANKA_PANDA_FLAT_CFG`` and ``FRANKA_PANDA_FLAT_HIGH_PD_CFG`` for the shared flat Franka
  asset. These configurations selected gripper-only colliders and the ``panda_arm`` actuator group;
  select the ``Physics`` variant for the target backend.

Deprecated
^^^^^^^^^^

* Deprecated ``FRANKA_PANDA_CFG``, ``FRANKA_PANDA_HIGH_PD_CFG``, and
  ``FRANKA_PANDA_MENAGERIE_CFG``. Their previous asset and actuator contracts remained available
  during the deprecation window. Use the corresponding ``FRANKA_PANDA_FLAT_*`` configuration to
  migrate to the shared flat asset, and replace shoulder/forearm actuator overrides with ``panda_arm``.
