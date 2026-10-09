* Fixed runs on hosts without CUDA, such as macOS, failing inside PyTorch when the resolved device was a CUDA
  device. :func:`~isaaclab.app.launch_simulation` now raises a clear error that asks for ``--device cpu``.
