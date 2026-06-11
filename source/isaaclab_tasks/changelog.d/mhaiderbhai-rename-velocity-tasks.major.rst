Changed
^^^^^^^

* **Breaking:** Dropped the ``-v0`` version suffix from the core locomotion-velocity environment IDs
  (AnymalD, Cassie, Digit, G1, H1, Spot, UnitreeGo2; ``Flat`` and ``Rough``, plus their ``-Play``
  variants). Multi-word robot names also drop the internal ``-`` (``-`` now separates task aspects
  only): ``Isaac-Velocity-Flat-Anymal-D-v0`` → ``Isaac-Velocity-Flat-AnymalD`` and
  ``Isaac-Velocity-Flat-Unitree-Go2-v0`` → ``Isaac-Velocity-Flat-UnitreeGo2``. Update ``gym.make`` /
  ``--task`` calls accordingly. The ``isaaclab_tasks.core.velocity`` Python API
  (:class:`~isaaclab_tasks.core.velocity.velocity_env_cfg.LocomotionVelocityRoughEnvCfg`, the
  shared ``mdp`` module, per-robot configs, and agent configs) is unchanged, so the contributed and
  experimental locomotion tasks that build on it continue to work. (Those tasks keep their ``-v0`` /
  ``-Warp-v0`` suffixes; the robot-name spelling change is covered separately.)
