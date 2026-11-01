* Added optional OVXR camera panels and a desktop mirror as application-configured
  Isaac Capture frame consumers using completed render products, with OVUI's default
  Vulkan backend for the mirror. :class:`~isaaclab_teleop.OvLibrariesXrCameraCfg` selects a panel.
* Enabled the mirror by default in OVXR live and MCAP replay entrypoints; ``--headless``
  disables the local window while preserving XR and camera-panel rendering.
* Supported task-configured automatic layouts for head-locked panels
  in live teleoperation and replay. ``--ovxr_pip_head_locked`` places a task layout that is
  not head-locked in the OpenXR VIEW space. Replay accepts the live ``--ovxr_pip_fps``,
  ``--ovxr_pip_view_position`` and ``--ovxr_desktop_mirror_fps`` options.
* Added :attr:`~isaaclab_teleop.XrCfg.excluded_visual_meshes`, which hides matching visual meshes
  from the OVXR eye cameras and camera panels without changing the source scene's visibility.
* Added the ``keep`` argument to :func:`~isaaclab_teleop.remove_camera_configs` to retain selected
  cameras while removing their observation terms.
