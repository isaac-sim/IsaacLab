Added
^^^^^

* Added Newton backend support for the multi-agent
  ``Isaac-Shadow-Hand-Over-Direct-v0`` (MAPPO/IPPO) env. Mirrors the
  single-agent Shadow Hand Newton port: per-hand
  :class:`~isaaclab.actuators.ImplicitActuatorCfg`,
  ``shadow_hand_instanceable_newton.usd``, per-backend
  :class:`~isaaclab_tasks.utils.PresetCfg` wrappers for sim physics, the
  hand-over object (``RigidObjectCfg`` on both backends, dropping
  PhysX-only knobs on Newton), and the two robot configs. Selectable via
  ``--preset newton`` / Hydra preset resolution; PhysX behavior unchanged.
  Migration details (Newton-side actuator gain overrides for ``fingers``
  and ``distal_passive``, and the ``ccd_iterations`` bump for multi-finger
  contacts) live in
  ``source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand_over/shadow_hand_over_env_cfg.py``.
