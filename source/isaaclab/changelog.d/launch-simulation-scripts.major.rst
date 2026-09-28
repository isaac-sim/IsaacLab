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
* Changed :func:`~isaaclab.app.launch_simulation` to end the process with the status of a ``sys.exit(n)`` raised in
  its block; Kit previously exited with 0.
* Changed the ``run_usd_camera`` tutorial to also save floating-point outputs, such as depth and normals, as NumPy
  ``.npy`` files when ``--save`` is passed.

Removed
^^^^^^^

* **Breaking:** Reduced the deprecated ``isaaclab.app.AppLauncher`` to a shim for scripts that still use it.
  ``AppLauncher(args).app`` still starts Isaac Sim / Kit through :func:`~isaaclab.app.launch_simulation` and
  ``app.close(exit_code=n)`` ends the process with ``n``, and
  ``AppLauncher.add_app_launcher_args`` still adds the launcher arguments. ``is_available``, ``has_gui``,
  ``has_window``, and ``device`` were removed: use ``isaaclab_physx.app.KitLauncher.is_available()``,
  :attr:`~isaaclab.sim.SimulationContext.has_gui`, and the simulation config's ``device``.
* Removed the ``check_*`` scripts under the core and PhysX test folders, which duplicated pytest coverage.
* **Breaking:** Renamed ``SimulationContext.is_headless_or_exist_active_visualizer`` to
  :meth:`~isaaclab.sim.SimulationContext.is_running`, which also returns False once every started visualizer is
  closed instead of treating the emptied visualizer list as a headless run. Replace the calls.
