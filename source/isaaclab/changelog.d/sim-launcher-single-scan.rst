Added
^^^^^

* Added :func:`~isaaclab.app.sim_launcher.make_physics_cfg` so a script can build a
  physics config for the backend selected via ``--physics``.
* Added a physics config as the value yielded by
  :func:`~isaaclab.app.sim_launcher.launch_simulation`, so callers can write
  ``with launch_simulation(PhysicsCfg(), args) as physics_cfg:`` and reuse the resolved backend.
