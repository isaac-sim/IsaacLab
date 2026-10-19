* **Breaking:** Changed :class:`~isaaclab_ov.sensors.ContactSensor` to give every body that a
  :attr:`~isaaclab.sensors.ContactSensorCfg.filter_prim_paths_expr` expression with a regular expression in
  its last segment matches its own entry of the force matrix, also when the binding matched one prim per
  environment before. ``{ENV_REGEX_NS}/Robot/.*`` below a robot whose only child is ``base`` gave one entry
  for ``base`` and now gives one per link. To keep one entry, name the body or write the last segment as a
  glob, such as ``{ENV_REGEX_NS}/Robot/*``; code that reshapes the force matrix must use the new entry count.
