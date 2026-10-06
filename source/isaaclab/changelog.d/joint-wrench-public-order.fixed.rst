* Fixed scene joint-wrench sensors ignoring the owning articulation's public body ordering.
  Force and torque entries now follow ``ArticulationCfg.body_ordering``, restricted to reportable bodies.
  With an explicit body ordering, resolve body indices against the sensor and update any code or
  checkpoint that assumed native sensor order. Default articulation ordering and standalone sensors
  retain backend ordering.
