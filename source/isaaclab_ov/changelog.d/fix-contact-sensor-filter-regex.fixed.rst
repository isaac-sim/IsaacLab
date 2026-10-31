* Fixed :class:`~isaaclab_ov.sensors.ContactSensor` failing to create its contact binding when a
  :attr:`~isaaclab.sensors.ContactSensorCfg.filter_prim_paths_expr` expression matched several partner
  bodies, such as the feet of a robot whose links are nested. Each matched body now gets its own entry of
  the force matrix. An expression that matches one body now gets that body, such as the ground collider for
  ``/World/ground/.*``, which gave an entry for the ``terrain`` prim that reported no force before.
