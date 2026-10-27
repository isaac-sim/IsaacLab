* Changed masked :meth:`~isaaclab.sensors.ray_caster.BaseRayCaster.reset` to draw its drift from the shared
  :class:`~isaaclab.utils.seed.WarpRng` state instead of random states of its own. It raises a ``RuntimeError``
  when that state is not initialized.
