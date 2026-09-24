Added
^^^^^

* Added ``BaseRenderer.render_batch()`` with a default loop over the existing single-camera
  ``render()`` interface, allowing renderers to optimize multiple camera captures.
* Added ``SensorBase.supports_batch_update`` to opt sensors into eager scene batching and
  ``SensorBase.update_batch()`` for standalone updates of batch-capable sensors.
* Added ``RenderContext.render_into_cameras()`` to group prepared captures by renderer and
  batch due cameras during eager scene updates. Preserved per-camera lazy reads, update
  periods, and reset state.
