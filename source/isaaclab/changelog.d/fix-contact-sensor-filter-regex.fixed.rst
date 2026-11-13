* Fixed :attr:`~isaaclab.sensors.ContactSensorCfg.filter_prim_paths_expr` expressions that match several
  partner bodies, such as the feet of a robot whose links are nested. The contact sensor base class now
  resolves an expression whose last segment is a regular expression into one expression per matched body
  for the PhysX and OVPhysX contact sensors, without repeating a body that several expressions match, and
  expands ``{ENV_REGEX_NS}`` in the filter expressions of a sensor created outside an
  :class:`~isaaclab.scene.InteractiveScene`.
