:orphan:

.. _how-to-export-environment-to-usd:

Exporting an Environment to USD
-------------------------------

Isaac Lab can export one initialized environment as a USD file that the physics engine's own import turns back
into the same physics, without Isaac Lab. The file keeps the scene's USD, with assets as references to their
source files, and adds the engine values that the import would not reproduce, such as actuator gains and the
initial joint state.

Setting up the export
^^^^^^^^^^^^^^^^^^^^^

One field turns the export on: :attr:`~isaaclab.scene.InteractiveSceneCfg.export_usd_path`. With it set, the
environment writes environment 0 when it is created:

* after the assets initialize, so the file holds the values the engine runs with, and before startup events
  (randomization);
* for the physics backend the environment runs on: the file reproduces that engine's physics in its own import;
* to the given ``.usda`` file, with the files the export copies, such as re-saved actuator networks, in
  ``<name>_files/`` next to it.

The path is part of the scene configuration because on PhysX and OVPhysX the scene reads what the engine loaded
before the assets write their configuration, and the export writes the difference.

From training, ``--export_deployment_usd`` sets it to ``deployment.usda`` in the run's log directory (torch
frontend, rank 0):

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD --export_deployment_usd

From a script, set it before creating the environment:

.. code-block:: python

   env_cfg.scene.export_usd_path = "/tmp/anymal_d.usda"
   env = gym.make("Isaac-Velocity-Flat-AnymalD", cfg=env_cfg)

Assets keep the path Isaac Lab loaded them from, a URL by default. Isaac Sim and OVPhysX fetch URLs; Newton opens
the file with OpenUSD's asset resolver, which in plain OpenUSD reads local files only. For it, export with a local
asset root (see :ref:`manage-asset-downloads`).

Loading the file
^^^^^^^^^^^^^^^^

Each engine imports the file from its path:

.. tab-set::

   .. tab-item:: PhysX

      .. code-block:: python

         from isaacsim import SimulationApp

         app = SimulationApp({"headless": True})

         import omni.usd
         from isaacsim.core.simulation_manager import SimulationManager

         omni.usd.get_context().open_stage(path)
         SimulationManager.initialize_physics()

   .. tab-item:: OVPhysX

      .. code-block:: python

         import ovphysx
         import ovstage
         from ovphysx.samples.python_samples.schema_support import physx_schema_paths

         ovstage.population.register_usd_schemas([*map(str, physx_schema_paths()), str(ovphysx.newton_schema_root())])
         physx = ovphysx.PhysX()
         stage = ovstage.Stage("export")
         ovstage.population.open_usd(stage, path, ordinal=1, domains=ovstage.PopulationDomain.ALL)
         stage.advance_write_floor(ordinal=1).wait()
         physx.wait_task(physx.attach_ovstage(stage, read_ordinal=1))

   .. tab-item:: Newton

      .. code-block:: python

         import newton
         from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

         builder = newton.ModelBuilder()
         newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
         result = builder.add_usd(
             path,
             schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
             allow_unsafe_torch_checkpoints=True,  # network actuators, such as ANYmal-D's LSTM
         )
         model = builder.finalize()

      For Kamino, register ``newton.solvers.SolverKamino``'s attributes instead and leave out ``SchemaResolverMjc``.

To simulate it as Isaac Lab did:

* Step at the file's physics step. Isaac Sim reads it from the file; on OVPhysX it is
  ``1 / physxScene:timeStepsPerSecond`` of the physics scene; Newton's import returns it as
  ``result["physics_dt"]``.
* On Newton, create the solver from the model, such as ``newton.solvers.SolverMuJoCo(model)``, which reads the
  solver options from the file. Kamino takes contacts from Newton's collision pipeline, as Isaac Lab does by
  default.
* On Newton, step the actuators in ``model.actuators``, such as networks, before the solver; see
  `Newton's actuators <https://newton-physics.github.io/newton/latest/concepts/actuators.html>`__.

What is not exported
^^^^^^^^^^^^^^^^^^^^

* Startup and reset events (randomization), controllers, and state other than the initial joint and
  deformable state.
* Terrain that Isaac Lab collides as a heightfield imports as its mesh.
* Actuators without a USD drive equivalent, such as networks, stay ``NewtonActuator`` prims, which only
  Newton's import reads; the export warns.
* The export warns where Newton's import cannot reproduce a value:

  * settings of solvers other than MuJoCo, beyond Kamino's constraint gains;
  * collisions every solver step instead of once per physics step, when the physics step has substeps;
  * joint target modes that the gains cannot tell apart;
  * the initial joint state of articulations with ball joints;
  * values of the joints, bodies and shapes Newton's import generates, such as cable segments.

* The export raises where an import would build something else:

  * Newton deformable objects;
  * bodies, joints or colliders added outside the USD, such as by a world builder hook;
  * a joint with different values per axis, or a material shared by colliders with different values.
