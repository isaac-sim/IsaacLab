Changed
^^^^^^^

* Changed Newton replication to import the physics scene and explicitly declared
  :attr:`isaaclab.cloner.ClonePlan.global_paths` without stage discovery. Hand-built clone plans must declare
  every shared USD asset root.
