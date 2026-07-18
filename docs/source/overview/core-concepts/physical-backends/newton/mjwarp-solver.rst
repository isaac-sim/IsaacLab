MJWarp Solver
=============

The MuJoCo-Warp solver from Google DeepMind is the primary, validated solver
for the Newton backend in Isaac Lab. It is enabled by setting
:attr:`~isaaclab_newton.physics.NewtonCfg.solver_cfg` to a
:class:`~isaaclab_newton.physics.MJWarpSolverCfg`, usually exposed as the
``newton_mjwarp`` physics preset on a task configuration. Newton ships with
beta support for an alternative Kamino solver — see :doc:`kamino-solver` and
:ref:`hydra-backend-solver-presets` for how presets are selected. For details
on how solver-specific managers are implemented, or how to add a new solver
manager, see :doc:`newton-manager-abstraction`.

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
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

    solver_cfg = MJWarpSolverCfg(
        use_mujoco_contacts=False,
        njmax=35,
        nconmax=20,
        ls_iterations=10,
        cone="pyramidal",
        impratio=1,
    )
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        collision_cfg=NewtonCollisionPipelineCfg(),
        num_substeps=1,
        debug_mode=False,
    )
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, physics=newton_cfg)


Here is a very brief explanation of some of the key parameters above:

* ``use_mujoco_contacts``: Selects contact ownership. First-party
  configurations set this to ``False`` so Newton's ``CollisionPipeline``
  generates contacts and MJWarp resolves them. The public default remains
  ``True`` during the deprecation window so
  existing external configurations can temporarily retain MuJoCo's internal
  collision path. New and migrated configurations should always set it to
  ``False``. Standalone and coupled MJWarp managers call
  :meth:`~isaaclab_newton.physics.MJWarpSolverCfg.validate_contact_mode`
  before constructing the solver, so custom manager integrations should do the
  same after applying any configuration overrides.

* ``collision_cfg``: Configures Newton's contact generation, including broad
  phase, contact reduction, and generation buffers. Move collision-specific
  tuning here when migrating from MuJoCo's internal collision path. Its
  ``rigid_contact_max`` field controls the first contact-capacity stage.

* ``njmax``: This is the number of constraint rows MuJoCo-Warp pre-allocates for a
  given environment. A large value will slow down the simulation, while a too small
  value may lead to missing constraints.

* ``nconmax``: This is the contact capacity MJWarp accepts from Newton. It does
  not size Newton's generation buffer; use ``collision_cfg.rigid_contact_max``
  for that stage. Both limits must be large enough to avoid overflow warnings.

* ``ccd_iterations``: Tunes GJK/EPA only on the deprecated MuJoCo internal
  collision path. It does not configure Newton's collision pipeline.

* ``ls_iterations``: The number of iterative line searches performed by the
  MuJoCo Warp solver. Line searches are used to find an acceptable step size,
  and the solver stops early when convergence is reached. Tune this cap together
  with ``iterations`` and ``tolerance`` when balancing convergence and runtime.

* ``cone``: This parameter provides a choice between pyramidal and elliptic
  approximations for the friction cone used in contact handling. Please see the
  MuJoCo documentation for additional information on contact:
  https://mujoco.readthedocs.io/en/stable/computation/index.html#contact

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
