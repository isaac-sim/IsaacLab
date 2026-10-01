* Fixed runs failing on hosts without CUDA, such as macOS, when the resolved device was a CUDA device.
  :func:`~isaaclab.app.launch_simulation` now falls back to ``cpu`` with a warning.
