Fixed
^^^^^

* Fixed deformable-body support being uninstallable on Linux aarch64: ``pytetwild`` was
  platform-gated to x86_64 from when it had no aarch64 wheel, so the deformables demo and
  tutorial failed with ``ModuleNotFoundError`` after a full install. Version 0.3.0 ships
  manylinux aarch64 wheels and is now installed there (with its ``[all]`` extra, since it
  imports ``pyvista`` at import time but only declares it in that extra).
