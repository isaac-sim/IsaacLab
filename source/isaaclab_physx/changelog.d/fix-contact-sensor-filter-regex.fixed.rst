* Fixed :class:`~isaaclab_physx.sensors.ContactSensor` failing to initialize when a
  :attr:`~isaaclab.sensors.ContactSensorCfg.filter_prim_paths_expr` expression matched several partner
  bodies, such as the feet of a robot whose links are nested. Each matched body now gets its own entry of
  the force matrix, also for a bare ``.*``, ``[^/]*`` or ``[^/]+`` last segment below a prim with a single
  child, which gave only that child before, and for an expression that matched as many prims in one
  environment as there are environments, which paired each environment with one of those prims before.
  An expression that matches one body now gets that body, such as the ground collider for
  ``/World/ground/.*``, which gave an entry for the ``terrain`` prim that reported no force before.
* Fixed :class:`~isaaclab_physx.sensors.ContactSensor` raising an ``AttributeError`` when PhysX rejected a
  pattern of the contact view. It now raises a :class:`RuntimeError` with the sensor and filter patterns.
