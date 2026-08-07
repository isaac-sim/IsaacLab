Fixed
^^^^^

* Fixed :class:`~isaaclab.renderers.camera_render_spec.CameraRenderSpec` accepting a ``view_count``
  that disagrees with ``camera_prim_paths``. A single fixed camera framing several environments
  produces one camera prim while the sensor knows about N environments; the tiled reshape was then
  launched over N tiles against a render product built from one camera, reading past the end of the
  annotator buffer. That surfaced asynchronously as an illegal memory access inside an unrelated
  device free, a long way from its cause. The spec now rejects the mismatch where it is introduced,
  and :class:`~isaaclab.sensors.Camera` sizes ``view_count`` from the camera prims it found.
