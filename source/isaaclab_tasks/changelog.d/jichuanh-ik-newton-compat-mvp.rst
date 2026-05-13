Changed
^^^^^^^

* Removed the ``self.sim.physics = PhysxCfg(...)`` overrides from
  ``Isaac-Reach-Franka-{IK-Abs,IK-Rel,OSC}-v0`` env configs so they
  inherit the parent ``ReachPhysicsCfg`` preset. Selecting
  ``presets=newton`` now picks ``NewtonCfg``; the previous
  ``bounce_threshold_velocity=0.2`` PhysX behavior is preserved as
  the default in ``ReachPhysicsCfg``. Direct-workflow callers in
  ``automate``, ``factory``, and the deploy MDP events module were
  migrated to the new
  :class:`~isaaclab.assets.BaseArticulationData` properties
  (:attr:`body_link_jacobian_w`, :attr:`mass_matrix`).
