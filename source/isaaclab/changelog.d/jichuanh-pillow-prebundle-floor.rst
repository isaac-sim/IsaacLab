Fixed
^^^^^

* Fixed docker installs deleting ``pillow`` from Isaac Sim's
  ``omni.kit.pip_archive`` prebundle by relaxing the exact ``pillow==12.1.1``
  pin to a ``>=12.1.1`` floor. The exact pin forced a downgrade once the Isaac
  Sim base image prebundled a newer pillow, and the deletion dangled the
  per-file symlink farm on aarch64, breaking extension startup.
