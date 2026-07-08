Fixed
^^^^^

* Fixed the initial actuator gain snapshot used by
  :func:`~isaaclab.envs.mdp.events.randomize_actuator_gains` corrupting (or
  crashing) for multi-environment floating-base articulations with Newton
  actuators. The per-environment stride of the actuator DOF indices was
  decoded with the articulation-local joint count instead of the whole
  model's per-environment DOF count, so on a floating base the free-root DOFs
  shifted every environment past the first to the wrong (or out-of-bounds)
  snapshot rows, corrupting the ``stiffness`` / ``damping`` randomization
  baseline.
