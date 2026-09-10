Added
^^^^^

* Added :func:`~isaaclab_ov.sim.usd_export.export_articulation_to_usd` to export a running OVPhysX
  articulation, as simulated, to a USD file. Prim paths are resolved from the stage, since an
  OVPhysX binding does not record them, and the state is authored by the shared exporter in
  :mod:`isaaclab.sim.usd_export`.

* Added complete selected-environment snapshots using the shared USD writer and effective
  OVPhysX buffers, with a fresh runtime round-trip regression covering multiple objects.
