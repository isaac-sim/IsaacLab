* Fixed :class:`~isaaclab.sensors.RayCaster` writing past its buffers when its asset exists in only some
  environments, for example through :attr:`~isaaclab.cloner.CloneCfg.clone_combinations`. Ray casters and
  ray-caster cameras now raise a :class:`RuntimeError` at initialization in that case.
* Fixed :class:`~isaaclab.sensors.MultiMeshRayCaster` and :class:`~isaaclab.sensors.MultiMeshRayCasterCamera`
  hiding a tracked target in every environment, or showing it in environments without it, when the target exists
  in only some environments. They now raise a :class:`RuntimeError` at initialization in that case.
