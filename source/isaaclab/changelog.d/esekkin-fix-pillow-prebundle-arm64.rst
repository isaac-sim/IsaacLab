Changed
^^^^^^^

* Updated the Isaac Sim pip dependency to ``6.0.1.0`` and aligned the ``pillow`` pin with
  Isaac Sim's prebundled version by pinning ``pillow==12.2.0``.

Fixed
^^^^^

* Fixed the Docker ``base`` image build failing on ``linux/arm64``. The stale ``pillow==12.1.1``
  pin forced ``./isaaclab.sh --install`` to uninstall the prebundled Pillow and replace it,
  dangling the shared ``PIL/__init__.py`` symlink that other Isaac Sim extensions rely on and
  tripping the prebundle-safety guard.
