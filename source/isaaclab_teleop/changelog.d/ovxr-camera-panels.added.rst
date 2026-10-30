* Added optional OVXR camera panels and a desktop mirror as application-configured
  Isaac Capture frame consumers using completed render products, with OVUI's default
  Vulkan backend for the mirror.
* Enabled the mirror by default in OVXR live and MCAP replay entrypoints; ``--headless``
  disables the local window while preserving XR and camera-panel rendering.
* Supported task-configured automatic layouts for head-locked panels
  in live teleoperation and replay.
