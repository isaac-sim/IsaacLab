Changed
^^^^^^^

* **Breaking:** Renamed the ``"instance_segmentation_fast"`` camera data type to
  ``"instance_segmentation"``. The new name conveys the functionality (requires prims tagged
  with semantic labels) and aligns with the existing ``"semantic_segmentation"`` data type. The
  ``_fast`` suffix leaked an implementation detail (non-stable instance IDs) that is not meaningful
  for Newton, where IDs are always stable.

  Migration: replace ``"instance_segmentation_fast"`` with ``"instance_segmentation"`` in all
  :attr:`~isaaclab.sensors.camera.CameraCfg.data_types` lists and ``camera.data.output`` /
  ``camera.data.info`` key lookups.

  :attr:`~isaaclab.renderers.output_contract.RenderBufferKind.INSTANCE_SEGMENTATION_FAST` has been
  removed; use :attr:`~isaaclab.renderers.output_contract.RenderBufferKind.INSTANCE_SEGMENTATION`
  instead.
