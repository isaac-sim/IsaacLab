Added
^^^^^

* Added ``semantic_segmentation`` camera data-type support to
  :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`, and brought the existing
  ``instance_segmentation_fast`` output up to the full Isaac RTX contract (colorized palettes plus
  ``idToLabels`` / ``idToSemantics`` metadata on ``camera.data.info``). Both are reconstructed on
  the host from Newton's per-shape index buffer and the USD stage's :class:`UsdSemantics.LabelsAPI`
  labels, reaching parity with :class:`~isaaclab_physx.renderers.IsaacRtxRenderer`.
* Added :attr:`~isaaclab_newton.renderers.NewtonWarpRendererCfg.semantic_filter`,
  :attr:`~isaaclab_newton.renderers.NewtonWarpRendererCfg.colorize_semantic_segmentation`, and
  :attr:`~isaaclab_newton.renderers.NewtonWarpRendererCfg.semantic_segmentation_mapping` to
  :class:`~isaaclab_newton.renderers.NewtonWarpRendererCfg`, mirroring the Isaac RTX renderer.
