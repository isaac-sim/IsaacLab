Added
^^^^^

* Added :func:`~isaaclab_newton.sim.usd_export.export_model_to_usd` to export a finalized Newton
  model back to a USD stage. Core physics is authored with standard ``UsdPhysics`` schemas and
  Newton-specific properties as ``newton:*`` attributes, at the prim paths the model was imported
  from, so that reimporting the exported stage reproduces the same model. Scenes cloned across
  environments hold one world per environment; the ``world`` argument selects which one is exported.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.get_stage_info` returning the USD import
  provenance of environment 0 for a running scene, in the form :meth:`newton.ModelBuilder.add_usd`
  returns it, so a live scene can be exported without holding on to the importer's result.
* Added export of plane shapes as ``UsdGeom.Plane``; every task's ground plane previously raised
  :class:`NotImplementedError`.
