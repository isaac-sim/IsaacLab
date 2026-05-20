Added
^^^^^

* Added an HDR output (:attr:`~isaaclab.renderers.RenderBufferKind.RGB_HDR`) to :class:`~isaaclab_ov.renderers.OVRTXRenderer`, sourced from the OVRTX HDR render var.
* Added internal :class:`~isaaclab.renderers.PpispPipeline` composition in :class:`~isaaclab_ov.renderers.OVRTXRenderer`: when :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg` is set the renderer allocates its own HDR scratch tensor and dispatches the PPISP kernel into the camera's ``rgb`` / ``rgba`` output after each render.
* Added a :meth:`~isaaclab.renderers.BaseRenderer.prepare_cameras` override on :class:`~isaaclab_ov.renderers.OVRTXRenderer` that authors a neutral ``OmniRtxCameraExposureAPI_1`` schema on each camera prim so RTX-side tonemapping does not double-process the ISP output.
