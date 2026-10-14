* Fixed :class:`~isaaclab_ov.sensors.ContactSensor` failing to create its contact binding when a
  :attr:`~isaaclab.sensors.ContactSensorCfg.filter_prim_paths_expr` expression matched several partner
  bodies, such as the feet of a robot whose links are nested. Each matched body now gets its own entry of
  the force matrix, also for a bare ``.*``, ``[^/]*`` or ``[^/]+`` last segment below a prim with a single
  child, which gave only that child before.
