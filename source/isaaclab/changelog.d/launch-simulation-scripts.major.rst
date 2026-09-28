Changed
^^^^^^^

* Changed every script, example, tutorial, and benchmark to start the simulation runtime with
  :func:`~isaaclab.app.launch_simulation` and to construct runtime objects from their configs with
  :func:`~isaaclab.utils.instantiate`.
* Changed the ``convert_mesh``, ``convert_urdf``, and ``convert_mjcf`` tools to preview the converted asset in the
  visualizer selected with ``--viz`` (including ``kit``) instead of a Kit-only viewport.
* Changed the ``run_usd_camera`` and ``run_ray_caster_camera`` tutorials to save images as PNG files with
  :func:`~isaaclab.utils.save_images_to_file` instead of Replicator writers.
* Changed ``pick_and_place.py`` to use :class:`~isaaclab.devices.Se3Keyboard`.

Removed
^^^^^^^

* **Breaking:** Reduced the deprecated ``isaaclab.app.AppLauncher`` to a shim for scripts that still use it.
  ``AppLauncher(args).app`` still starts Isaac Sim / Kit through :func:`~isaaclab.app.launch_simulation`, and
  ``AppLauncher.add_app_launcher_args`` still adds the launcher arguments. ``is_available``, ``has_gui``,
  ``has_window``, and ``device`` were removed: use ``isaaclab_physx.app.KitLauncher.is_available()``,
  :attr:`~isaaclab.sim.SimulationContext.has_gui`, and the simulation config's ``device``.
* Removed the ``check_*`` scripts under the core and PhysX test folders, which duplicated pytest coverage.
