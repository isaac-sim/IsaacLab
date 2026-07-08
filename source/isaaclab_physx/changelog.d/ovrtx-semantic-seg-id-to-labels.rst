Fixed
^^^^^

* Fixed :meth:`~isaaclab_physx.renderers.isaac_rtx_renderer.IsaacRtxRenderer.read_output`
  leaving a stale segmentation ``idToLabels`` mapping in ``camera.data.info`` when an
  annotator stopped emitting metadata on a later frame. Per-output metadata is now
  replaced (not merged) each frame, so a dropped mapping resets to ``None``.
