Changed
^^^^^^^

* Updated :class:`~isaaclab.sensors.camera.Camera` to construct its internal
  :class:`~isaaclab.sim.views.FrameView` without the now-removed
  ``sync_usd_on_fabric_write`` kwarg.  USD attributes on camera prims are
  no longer kept in sync with Fabric writes; read poses through the view's
  getters instead.
