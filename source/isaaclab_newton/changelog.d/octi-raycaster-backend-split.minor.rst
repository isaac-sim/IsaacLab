Added
^^^^^

* Added Newton backend for :class:`~isaaclab.sensors.ray_caster.RayCaster` /
  :class:`~isaaclab.sensors.ray_caster.RayCasterCamera` /
  :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCaster` /
  :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCasterCamera`. Site-based,
  matching :class:`~isaaclab_newton.sensors.pva.Pva` and
  :class:`~isaaclab_newton.sensors.frame_transformer.FrameTransformer`:
  registers body-attached sites via
  :meth:`~isaaclab_newton.physics.NewtonManager.cl_register_site` for both
  the sensor frame and any tracked target meshes, and reads per-step
  transforms off :class:`~newton.sensors.SensorFrameTransform` against a
  world-origin reference. Static parents/targets bypass the site
  machinery and serve cached per-env ``wp.transformf`` arrays.

Changed
^^^^^^^

* Changed Newton tracked target mesh updates to copy site poses directly into
  Warp mesh pose tables instead of staging through torch views.
