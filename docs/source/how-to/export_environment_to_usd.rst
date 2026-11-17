:orphan:

.. _how-to-export-environment-to-usd:

Exporting an Environment to USD
-------------------------------

Isaac Lab can export one initialized environment as a USD file that each backend's own import (Isaac Sim,
OVPhysX or Newton) turns back into the same physics, without Isaac Lab. The file keeps the scene's USD, with
assets as references to their source files, and adds the engine values that the import would not reproduce,
such as actuator gains and the initial joint state.

Exporting from training
^^^^^^^^^^^^^^^^^^^^^^^

``--export_deployment_usd`` writes ``deployment.usda`` to the run's log directory, after the assets
initialize and before startup events:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-AnymalD --export_deployment_usd

Exporting from a script
^^^^^^^^^^^^^^^^^^^^^^^

Set :attr:`~isaaclab.scene.InteractiveSceneCfg.export_usd_path` and the environment exports environment 0
when it is created:

.. code-block:: python

   env_cfg.scene.export_usd_path = "/tmp/anymal_d.usda"
   env = gym.make("Isaac-Velocity-Flat-AnymalD", cfg=env_cfg)

To export from a scene directly, call :meth:`~isaaclab.scene.InteractiveScene.export_to_usd` after the
simulation is reset and before the first step. On PhysX and OVPhysX the scene must still be created with
``export_usd_path`` set: the export compares the engine with what it loaded, which the scene reads only then.

Loading the file
^^^^^^^^^^^^^^^^

Each engine reads the file with its own USD import. Isaac Sim fetches the remote assets the file references; for
OVPhysX and Newton, :meth:`~isaaclab.sim.usd_export.BaseUsdExporter.open_stage` downloads them as Isaac Lab does
when spawning them. Files the export copied, such as re-saved actuator networks, are next to the file in
``<name>_files/``.

.. tab-set::

   .. tab-item:: Isaac Sim

      Open the file (**File > Open**) and press **Play**, or from a script:

      .. code-block:: python

         from isaacsim import SimulationApp

         app = SimulationApp({"headless": True})

         import omni.timeline
         import omni.usd

         omni.usd.get_context().open_stage(path)
         omni.timeline.get_timeline_interface().play()
         app.update()

   .. tab-item:: OVPhysX

      .. code-block:: python

         import ovphysx
         import ovstage
         from ovphysx.samples.python_samples.schema_support import physx_schema_paths
         from pxr import UsdPhysics

         from isaaclab.sim.usd_export import BaseUsdExporter

         usd = BaseUsdExporter.open_stage(path)
         scene = next(prim for prim in usd.Traverse() if prim.IsA(UsdPhysics.Scene))
         dt = 1 / scene.GetAttribute("physxScene:timeStepsPerSecond").Get()

         ovstage.population.register_usd_schemas([*map(str, physx_schema_paths()), str(ovphysx.newton_schema_root())])
         physx = ovphysx.PhysX()
         stage = ovstage.Stage("export")
         source = usd.Flatten().ExportToString()
         ovstage.population.open_usd_from_string(stage, source, ordinal=1, domains=ovstage.PopulationDomain.ALL)
         stage.advance_write_floor(ordinal=1).wait()
         physx.wait_task(physx.attach_ovstage(stage, read_ordinal=1))
         physx.step_sync(dt)

   .. tab-item:: Newton

      .. code-block:: python

         import newton
         from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

         from isaaclab.sim.usd_export import BaseUsdExporter

         builder = newton.ModelBuilder()
         newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
         result = builder.add_usd(
             BaseUsdExporter.open_stage(path),
             schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
             allow_unsafe_torch_checkpoints=True,  # network actuators, such as ANYmal-D's LSTM
         )
         model = builder.finalize()
         solver = newton.solvers.SolverMuJoCo(model)  # with the solver options the file states
         dt = result["physics_dt"]

      For Kamino, register ``newton.solvers.SolverKamino``'s attributes instead and leave out ``SchemaResolverMjc``.

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
