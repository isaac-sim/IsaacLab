Fixed
^^^^^

* Fixed docker installs deleting ``pillow`` from Isaac Sim's
  ``omni.kit.pip_archive`` prebundle by relaxing the exact ``pillow==12.2.0``
  pin to a ``>=12.1.1`` floor. An exact pin below the version prebundled by a
  future Isaac Sim base image forces a downgrade that deletes the prebundled
  copy, dangling the per-file symlink farm on aarch64 and breaking extension
  startup.
