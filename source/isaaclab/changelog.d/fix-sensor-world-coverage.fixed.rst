* Fixed :class:`~isaaclab.sensors.RayCaster` writing past its buffers when its asset exists in only some
  environments, for example through :attr:`~isaaclab.cloner.CloneCfg.clone_combinations`. Ray casters and
  ray-caster cameras now raise a :class:`RuntimeError` at initialization in that case.
