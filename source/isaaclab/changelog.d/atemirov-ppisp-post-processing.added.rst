* Added ``isaaclab.sensors.post_processing`` with :class:`~isaaclab.sensors.SensorPostProcessor`,
  :class:`~isaaclab.sensors.SensorPostProcessorCfg`, and
  :class:`~isaaclab.sensors.SensorPostProcessingPipeline` for ordered, renderer-independent
  post-processing of camera image buffers.
* Added ``mdp.processed_image`` for ordered image-processing chains owned by observation terms,
  with explicit buffer requirements, persistent intermediates, cached frame processing, and
  partial-reset and cleanup callbacks. Added early ``ManagerTermBase.prepare_scene`` preparation
  so terms could resolve renderer inputs before setup. Added ``rgb_radiance`` for scene-linear RGB
  before exposure and camera response, in renderer-relative intensity units. Preserved existing
  ``rgb_hdr`` settings when radiance was not requested. Prepared all cameras' renderer inputs before
  shared stage export, including public outputs and private processor inputs. Added the
  ``Camera.render_generation`` attribute to track completed render batches without advancing on cached reads.
  Added ``Camera.render_frame`` to match processing state to published asynchronous captures;
  repeated priming frames did not advance PPISP controllers.
