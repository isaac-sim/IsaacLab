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

Isaac Sim and OVPhysX open the file as any USD stage. Newton imports it with the solver's custom attributes and
schema resolvers:

.. code-block:: python

   import newton
   from isaaclab_newton.physics.newton_builder import register_builder_attributes, usd_schema_resolvers

   from isaaclab.sim.usd_export import BaseUsdExporter

   builder = newton.ModelBuilder()
   register_builder_attributes(builder, solver_cfg)
   builder.add_usd(
       BaseUsdExporter.open_stage(path),
       schema_resolvers=usd_schema_resolvers(solver_cfg),
       allow_unsafe_torch_checkpoints=True,  # network actuators, such as ANYmal-D's LSTM
   )
   model = builder.finalize()

:meth:`~isaaclab.sim.usd_export.BaseUsdExporter.open_stage` downloads referenced remote assets as Isaac Lab does
when spawning them; a plain ``Usd.Stage.Open`` needs a resolver that fetches them. Files the export copied, such
as re-saved actuator networks, are next to the file in ``<name>_files/``.

What is not exported
^^^^^^^^^^^^^^^^^^^^

* Startup and reset events (randomization), controllers, and state other than the initial joint and
  deformable state.
* Values the import cannot read, such as Newton's collision pipeline and substeps: the export warns.
* Entities the import does not build alike, such as bodies a Newton world builder hook adds: the export raises.
