Removed
^^^^^^^

* **Breaking:** Removed the ``newton_kamino`` physics preset from
  :class:`~isaaclab_tasks.core.reach.reach_env_cfg.ReachPhysicsCfg`, so ``Isaac-Reach-Franka``,
  ``Isaac-Reach-Franka-OSC`` and ``Isaac-Reach-UR10`` no longer accept ``physics=newton_kamino``.
  Use ``physics=newton_mjwarp`` (the default) instead, or redeclare a task-local
  ``newton_kamino`` preset if the Kamino solver is needed.
