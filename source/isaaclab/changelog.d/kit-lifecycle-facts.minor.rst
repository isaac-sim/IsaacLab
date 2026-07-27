Added
^^^^^

* Added :meth:`~isaaclab.app.AppLauncher.is_available` to report whether the full Isaac Sim
  runtime is importable, i.e. Kit can be launched in the current process.
* Added :meth:`~isaaclab.app.AppLauncher.has_gui` to report whether the resolved app state has
  an interactive GUI (local window, livestream, or XR), wrapping the ``/isaaclab/has_gui``
  setting that AppLauncher publishes.
* Added :func:`~isaaclab.sim.utils.show_stage_in_viewport` to display a USD file in the Kit
  viewport until the window is closed.
* Added kitless ``--viz`` preview backends (``newton``, ``rerun``, ``viser``) to
  ``scripts/tools/convert_urdf.py`` and ``scripts/tools/convert_mjcf.py``, opening the converted
  asset in an Isaac Lab visualizer without Isaac Sim (reusing
  :func:`~isaaclab.app.launch_simulation` and :class:`~isaaclab.sim.SimulationContext`).
  ``--viz kit`` keeps showing the asset in the Isaac Sim viewport and now reports an error when
  selected without a full Isaac Sim installation, instead of being silently unavailable.

Changed
^^^^^^^

* Changed the URDF/MJCF converter CLI to enable the Isaac Sim importer extensions through the
  existing :func:`~isaaclab.sim.utils.enable_extension` helper, guarded by
  :func:`~isaaclab.utils.version.has_kit`, and to resolve them from the standalone importer wheel
  when running kitless.
* Changed ``scripts/tools/convert_mesh.py`` to open its post-conversion preview through the Kit
  USD context (:func:`~isaaclab.sim.utils.show_stage_in_viewport`) so the converted asset is
  visible in the viewport.
