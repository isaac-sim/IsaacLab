Added
^^^^^

* Added an HDR output (:attr:`~isaaclab.renderers.RenderBufferKind.RGB_HDR`) to :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`, sourced from its native scene-linear color buffer.
* Added internal :class:`~isaaclab.renderers.PpispPipeline` composition in :class:`~isaaclab_newton.renderers.NewtonWarpRenderer`: when :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg` is set the renderer allocates its own HDR scratch tensor and dispatches the PPISP kernel into the camera's ``rgb`` / ``rgba`` output after each render.
