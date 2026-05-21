Added
^^^^^

* Added an HDR output (:attr:`~isaaclab.renderers.RenderBufferKind.RGB_HDR`) to :class:`~isaaclab_physx.renderers.IsaacRtxRenderer`, sourced from the Replicator ``HdrColor`` annotator.
* Added internal :class:`~isaaclab.renderers.PpispPipeline` composition in :class:`~isaaclab_physx.renderers.IsaacRtxRenderer`: when :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg` is set the renderer allocates its own HDR scratch buffer and dispatches the PPISP kernel into the camera's ``rgb`` / ``rgba`` output after each render.
* Added a :meth:`~isaaclab.renderers.BaseRenderer.prepare_cameras` override on :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` that authors a neutral ``OmniRtxCameraExposureAPI_1`` schema on each camera prim so RTX-side tonemapping does not double-process the ISP output.
