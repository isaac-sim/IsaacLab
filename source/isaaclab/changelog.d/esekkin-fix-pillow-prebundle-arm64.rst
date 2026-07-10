Fixed
^^^^^

* Fixed the Docker ``base`` image build failing on ``linux/arm64`` by aligning the
  ``pillow`` pin with Isaac Sim's prebundled version (``12.1.1`` -> ``12.2.0``). The stale
  pin forced ``./isaaclab.sh --install`` to uninstall the prebundled Pillow and replace it,
  dangling the shared ``PIL/__init__.py`` symlink that other Isaac Sim extensions rely on and
  tripping the prebundle-safety guard.
