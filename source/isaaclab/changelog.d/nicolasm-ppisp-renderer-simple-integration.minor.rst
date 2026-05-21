Added
^^^^^

* Added :class:`~isaaclab.sensors.camera.CameraISPMode` (values ``AUTO_CAMERA`` and ``AUTO_ANY``) and the :attr:`~isaaclab.sensors.camera.CameraCfg.isp_cfg` field to author or auto-discover an ISP cfg (:class:`~isaaclab_ppisp.PpispCfg`) from a USD ``RenderProduct`` bound to the camera (or anywhere on the stage as a fallback).
* Added :attr:`~isaaclab.renderers.RenderBufferKind.RGB_HDR` to the renderer output contract so backends can advertise a 3-channel float HDR AOV.
* Added :meth:`~isaaclab.renderers.BaseRenderer.prepare_cameras` hook so backends can author per-camera USD overrides.
* Added :class:`~isaaclab.renderers.CameraRenderSpec`, an immutable description of a tiled camera passed to render backends so they no longer hold a reference to the camera sensor instance.
