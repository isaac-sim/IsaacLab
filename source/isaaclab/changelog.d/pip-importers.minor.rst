Added
^^^^^

* Added support for loading URDF and MJCF importer APIs from the standalone
  ``isaacsim-asset-isolated`` package when Isaac Sim is unavailable.

Changed
^^^^^^^

* **Breaking:** Changed the default ``--collision-type`` in ``scripts/tools/convert_mjcf.py`` from
  ``default`` to ``Convex Hull`` and restricted the flag to the collision approximations
  supported by the MJCF importer. Previously accepted free-form values now fail argument
  parsing; pass one of the listed choices instead.
* **Breaking:** Changed ``scripts/tools/convert_urdf.py`` and ``scripts/tools/convert_mjcf.py``
  to accept only converter arguments plus ``--viz {kit,none}``. :class:`~isaaclab.app.AppLauncher`
  flags (e.g. ``--headless``, ``--livestream``, ``--device``) are no longer accepted; they never
  affected the conversion output. Use ``--viz kit`` for the viewport preview and the
  ``LIVESTREAM`` environment variable for remote streaming.

Fixed
^^^^^

* Fixed ``scripts/tools/convert_urdf.py`` and ``scripts/tools/convert_mjcf.py`` exiting with
  code 0 when the conversion failed, which hid failures from shell scripts and CI pipelines.
