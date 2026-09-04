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
* Added :func:`~isaaclab_newton.sim.usd_export.resolve_world_prim_paths`, returning a
  :class:`~isaaclab_newton.sim.usd_export.WorldPrimPaths` that maps one world's bodies, shapes and
  joints to the prim paths the export authors them at. Visual copies the importer makes when
  approximating a mesh for collision are authored as visual-only ``<prim>_visual`` siblings.
