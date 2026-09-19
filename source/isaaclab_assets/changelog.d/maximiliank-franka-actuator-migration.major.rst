Changed
^^^^^^^

* **Breaking:** Made the flat multi-physics Franka asset the canonical configuration with gripper-only
  collisions by default. Its actuator groups are now ``panda_arm``, ``panda_hand``, and
  ``panda_finger2_passive``. Update overrides of ``panda_shoulder`` and ``panda_forearm`` to target
  ``panda_arm``, or use ``FRANKA_PANDA_LEGACY_CFG`` for compatibility with the previous asset and
  actuator grouping.
