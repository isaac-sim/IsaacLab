Added
^^^^^

* Added a ``require_kit`` launcher argument, read by :func:`~isaaclab.app.launch_simulation`
  alongside ``physics``, so a tool can declare that it needs Kit for a reason its config cannot
  express, such as reaching a Kit-only extension API. The override is additive: it can turn a
  kitless launch into a Kit one, never the reverse.

Changed
^^^^^^^

* Changed ``scripts/tools/convert_urdf.py`` and ``scripts/tools/convert_mjcf.py`` to bootstrap
  through :func:`~isaaclab.app.add_launcher_args` and :func:`~isaaclab.app.launch_simulation`, like
  the other tool and demo scripts. Both scripts accept the full launcher argument set again,
  including ``--device``, ``--livestream``, ``--experience``, and the comma-separated
  ``--viz kit,newton`` spelling.

Fixed
^^^^^

* Fixed :meth:`~isaaclab.app.AppLauncher.add_app_launcher_args` taking the parser down when the
  script declares required positional arguments and is invoked with ``--help``. The launcher
  arguments now appear in the help output of ``scripts/tools/convert_mesh.py``,
  ``convert_urdf.py``, and ``convert_mjcf.py`` instead of an "arguments are required" error.
* Changed the URDF and MJCF converters to prefer the standalone ``isaacsim-asset-isolated``
  importer wheel whenever it is installed, rather than always launching Kit when Isaac Sim is
  present. Kit is now launched only when the wheel is absent or a Kit preview is requested.
* **Breaking:** Removed the converter-local ``--viz`` flag in favor of the launcher's
  ``--visualizer`` / ``--viz`` argument that every other script already uses. ``--viz auto`` and a
  bare ``--viz`` are no longer accepted; name the backend explicitly, for example ``--viz kit`` or
  ``--viz newton``.
