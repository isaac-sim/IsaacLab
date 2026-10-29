* Changed OVRTX to render scenes with :class:`~isaaclab.sim.SwitchableAssetsCfg` assets through an independent
  OVStage by default, since every environment holds all candidates of such an asset and native cloning renders the
  first frame far slower. ``ISAAC_LAB_OVRTX_USE_OVSTAGE`` still overrides the default.
