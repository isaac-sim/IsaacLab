* **Breaking:** Replaced the legacy MJWarp/VBD shared-model penalty coupling with generic staggered proxy
  feedback. One-way and two-way coupling remain available, but trajectories change; retune existing simulations
  and use :class:`~isaaclab_newton.physics.coupling.CouplerProxyCfg` for new configurations. Legacy drivers expose
  the translated configuration through ``driver.cfg`` and keep ``rigid`` and ``soft`` as child-driver accessors.
