Solver Transitioning
====================

Transitioning to the Newton physics engine introduces new physics solvers that handle simulation using different numerical approaches.
While Newton supports several different solvers, our initial focus for Isaac Lab is on using the MuJoCo-Warp solver from Google DeepMind.

The way the physics scene itself is defined does not change - we continue to use USD as the primary way to set basic parameters of objects and robots in the scene,
and for current environments, the exact same USD files used for the PhysX-based Isaac Lab are used.
In the future, that may change, as new USD schemas are under development that capture additional physics parameters.

What does require change is the way that some solver-specific settings are configured.
Tuning these parameters can have a significant impact on both simulation performance and behaviour.

For now, we will show an example of setting these parameters to help provide a feel for these changes.
Note that the :class:`~isaaclab.sim.NewtonCfg` replaces the :class:`~isaaclab.sim.PhysxCfg` and is used to set everything related to the physical simulation parameters except for the ``dt``:

.. code-block:: python

    from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
    from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg

    solver_cfg = MJWarpSolverCfg(
        nefc_per_env=35,
        ls_iterations=10,
        cone="pyramidal",
        ls_parallel=True,
        impratio=1,
    )
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,
    )
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, newton_cfg=newton_cfg)


Here is a very brief explanation of some of the key parameters above:

* ``nefc_per_env``: This is the size of the buffer constraints we want MuJoCo warp to
  pre-allocate for a given environment. A large value will slow down the simulation,
  while a too small value may lead to some contacts being missed.

* ``ls_iterations``: The number of line searches performed by the MuJoCo Warp solver.
  Line searches are used to find an optimal step size, and for each solver step,
  at most ``ls_iterations`` line searches will be performed. Keeping this number low
  is important for performance. This number is also an upper bound when
  ``ls_parallel`` is not set.

* ``cone``: This parameter provides a choice between pyramidal and elliptic
  approximations for the friction cone used in contact handling. Please see the
  MuJoCo documentation for additional information on contact:
  https://mujoco.readthedocs.io/en/stable/computation/index.html#contact

* ``ls_parallel``: This switches line searches from iterative to parallel execution.
  Enabling ``ls_parallel`` provides a performance boost, but at the cost of some
  simulation stability. To ensure good simulation behaviour when enabled, a higher
  ``ls_iterations`` setting is required. Usually an increase of approximately 50% is
  best over the ``ls_iterations`` setting when ``ls_parallel`` is disabled.

* ``impratio``: This is the frictional-to-normal constraint impedance ratio that
  enables finer-grained control of the significance of the tangential forces
  compared to the normal forces. Larger values signify more emphasis on harder
  frictional constraints to avoid slip. More on how to tune this parameter (and
  cone) can be found in the MuJoCo documentation here:
  https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-impratio

* ``num_substeps``: The number of substeps to perform when running the simulation.
  Setting this to a number larger than one allows to decimate the simulation
  without requiring Isaac Lab to process data between two substeps. This can be
  of value when using implicit actuators, for example.


A more detailed transition guide covering the full set of available parameters and describing tuning approaches will follow in an upcoming release.
