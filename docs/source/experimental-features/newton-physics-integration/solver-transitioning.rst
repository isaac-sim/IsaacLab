Solver Transitioning
====================

Transitioning to the Newton physics engine introduces new physics solvers that handle simulation using different numerical approaches.
While Newton supports several different solvers, our initial focus for Isaac Lab is on using the
MuJoCo-Warp solver from Google DeepMind. Isaac Lab also includes beta support for the Kamino
solver on selected classic tasks. Kamino is selected through a physics preset rather than as a
separate backend; see :ref:`hydra-backend-solver-presets`.

.. note::

    Kamino support is experimental and currently depends on assets being structured
    in a way that Kamino can consume. Assets that work with MuJoCo-Warp or PhysX
    may still require model-structure updates before they work with Kamino.

The way the physics scene itself is defined does not change - we continue to use USD as the primary way to set basic parameters of objects and robots in the scene,
and for current environments, the exact same USD files used for the PhysX-based Isaac Lab are used.
In the future, that may change, as new USD schemas are under development that capture additional physics parameters.

What does require change is the way that some solver-specific settings are configured.
Tuning these parameters can have a significant impact on both simulation performance and behaviour.

For now, we will show an example of setting these parameters to help provide a feel for these changes.
Note that the :class:`~isaaclab_newton.physics.NewtonCfg` replaces
:class:`~isaaclab_physx.physics.PhysxCfg` and is used to set everything related to the physical
simulation parameters except for the ``dt``:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    solver_cfg = MJWarpSolverCfg(
        njmax=35,
        nconmax=20,
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
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=newton_cfg)


Here is a very brief explanation of some of the key parameters above:

* ``njmax``: This is the number of constraint rows MuJoCo-Warp pre-allocates for a
  given environment. A large value will slow down the simulation, while a too small
  value may lead to missing constraints.

* ``nconmax``: This is the maximum number of contact points MuJoCo-Warp pre-allocates
  for a given environment. Set it high enough for the expected contact count.

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
