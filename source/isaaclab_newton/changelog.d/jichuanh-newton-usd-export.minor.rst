Added
^^^^^

* Added :func:`~isaaclab_newton.sim.usd_export.export_model_to_usd` to export a finalized Newton
  model back to a USD stage. Core physics is authored with standard ``UsdPhysics`` schemas and
  Newton-specific properties as ``newton:*`` attributes, at the prim paths the model was imported
  from, so that reimporting the exported stage reproduces the same model.
