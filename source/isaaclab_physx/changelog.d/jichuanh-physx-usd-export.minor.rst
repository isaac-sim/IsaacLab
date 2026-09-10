Added
^^^^^

* Added :func:`~isaaclab_physx.sim.usd_export.export_articulation_to_usd` to export a running PhysX
  articulation, as simulated, to a USD file. Prim paths are read from the PhysX tensor view and the
  state is authored by the shared exporter in :mod:`isaaclab.sim.usd_export`.

* Added complete selected-environment snapshots with effective rigid-body and articulation
  properties, gravity and timestep, and a fresh PhysX round-trip regression with two environments.
