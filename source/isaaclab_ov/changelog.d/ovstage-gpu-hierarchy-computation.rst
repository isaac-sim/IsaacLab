Changed
^^^^^^^

* Changed :func:`~isaaclab_ov.stage.create_ovstage` to select the ovstage hierarchy computation
  model from the installed ovstage version instead of always requesting
  ``HierarchyComputationModel.CPU_INCREMENTAL``. ovstage 0.2 and later place objects correctly
  under ``GPU_INCREMENTAL``, which moves world-transform computation off the host; ovstage 0.1
  keeps the host model. The version is resolved once at import by the new
  :mod:`isaaclab_ov.ovstage_compat` module. No migration is required: the public extras stay
  pinned to ``ovstage==0.1.1.355824``, so the host model remains in force until that pin moves,
  and a missing or unparsable install also keeps the host model.
