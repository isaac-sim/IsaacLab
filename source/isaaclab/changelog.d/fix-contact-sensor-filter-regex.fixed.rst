* Fixed :attr:`~isaaclab.sensors.ContactSensorCfg.filter_prim_paths_expr` expressions that match several
  partner bodies, such as the feet of a robot whose links are nested. The contact sensor base class now
  resolves such an expression into one expression per body for the PhysX and OVPhysX contact sensors.
