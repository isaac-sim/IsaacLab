Changed
^^^^^^^

* Changed the shared rough-locomotion Newton preset
  :class:`~isaaclab_tasks.core.velocity.velocity_env_cfg.RoughPhysicsCfg` to raise the MuJoCo
  constraint budget from ``njmax=200``/``nconmax=100`` to ``njmax=1000``/``nconmax=300``. Rough
  terrain needs far more constraint rows than the old budget allowed, and MuJoCo drops the excess
  rows silently, so the overflow presented as unstable contact rather than as a resource limit.
* Changed the same preset's ``default_shape_cfg`` margin from ``0.01`` to ``0.0``, and removed the
  per-robot ``margin = 0.001`` override from
  :class:`~isaaclab_tasks.core.velocity.config.anymal_d.rough_env_cfg.AnymalDRoughEnvCfg`.
  ``margin`` is a rest offset that inflates every collider, so a shared non-zero value changes the
  shape of every asset using the preset: at ``margin=0.01`` opposing surfaces engage across a 20 mm
  cushion and never touch, which breaks sim2real transfer. The margin was previously believed to be
  required for non-AnymalD robots on triangle-mesh terrain, but it was masking the constraint
  overflow addressed above; with the budgets raised, every robot in the suite trains at
  ``margin=0``.
* Changed the Cassie, Unitree Go2, and H1 rough environments to set contact stiffness
  ``ke=10000`` with ``kd=200`` on the Newton MJWarp shape config. The shared ``ke=2500`` leaves the
  contact time constant at ``4 * sim.dt``, which corrects penetration too slowly on mesh terrain.
  ``kd = 2*sqrt(ke)`` keeps the contact critically damped; the damping ratio matters considerably
  more than the stiffness itself. The stiffness is set per robot rather than in the shared preset
  because it does not help every robot in the suite.
