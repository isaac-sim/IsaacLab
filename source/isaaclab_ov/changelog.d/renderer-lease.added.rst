* Added :class:`~isaaclab_ov.renderers.OvLibrariesRenderLease` and
  :meth:`~isaaclab_ov.renderers.OVRTXRenderer.acquire_ov_libraries_render_lease` for application-owned
  OVRTX/OVStage borrowing by external consumers, with private-stage selection, capability-checked
  render-thread ownership transfer, deferred Stage materialization, scene publication through
  :class:`~isaaclab_ov.renderers.OvLibrariesScenePublisher`, and ordered cleanup.
* Published dirty visual materials with leased scene updates, including material-only changes.
