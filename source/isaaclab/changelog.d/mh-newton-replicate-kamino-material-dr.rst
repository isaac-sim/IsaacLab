Fixed
^^^^^

* Fixed :class:`~isaaclab.envs.mdp.events.randomize_rigid_body_material` crashing on the Newton
  Kamino solver, which shares contact materials across shapes and environments and rejects
  per-shape overrides. On Kamino it now samples one value per build-time material group and
  broadcasts it to every environment; all other Newton solvers keep per-shape sampling.
