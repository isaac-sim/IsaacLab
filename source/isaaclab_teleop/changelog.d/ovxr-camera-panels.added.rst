* Added optional OVXR camera panels and a desktop mirror as application-configured
  Isaac Capture frame consumers using completed render products, with OVUI's default
  Vulkan backend for the mirror. :class:`~isaaclab_teleop.OvLibrariesXrCameraFeedCfg` selects a panel, and
  :class:`~isaaclab_teleop.OvLibrariesXrDesktopMirrorCfg` enables the mirror.
  Selected cameras render through the session's :class:`~isaaclab_teleop.OvxrRendererCfg` renderer,
  each with its own native update rate, and provide colour or depth outputs.
* Added ``--ovxr_desktop_mirror`` to enable the local window in OVXR live teleoperation, recording and
  MCAP replay.
  The mirror is disabled by default; XR and camera-panel rendering remain independent.
* Supported task-configured automatic layouts for head-locked panels
  in live teleoperation, recording and replay. OVXR shows camera feeds head-locked, using the size,
  gap and distance of the task's ``xr_camera_feed_layout``. Recording and replay accept the live
  ``--ovxr_camera_feed_fps`` and ``--ovxr_desktop_mirror_fps`` options.
* Added :attr:`~isaaclab_teleop.XrCfg.excluded_visual_meshes`, which hides matching visual meshes
  from the OVXR render without changing the source scene's visibility.
* Added the ``keep`` argument to :func:`~isaaclab_teleop.remove_camera_configs` to retain selected
  cameras while removing their observation terms.
* Added scene partitions to OVXR: tracking markers show in the eye views and the desktop mirror and stay out
  of sensor camera images and camera panels, while task geometry and lighting are shared between them.
