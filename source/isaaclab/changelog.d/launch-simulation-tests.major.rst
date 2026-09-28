Changed
^^^^^^^

* Changed the tests to start the simulation runtime with :func:`isaaclab.test.utils.launch_test_simulation`,
  and made the core ``isaaclab`` tests backend-agnostic; tests of Kit, PhysX, or Isaac Sim behavior moved to
  ``isaaclab_physx``.
* **Breaking:** Changed :func:`isaaclab.test.utils.launch_test_simulation` to take an optional simulation
  config and start the runtime that config needs (default :class:`~isaaclab.sim.SimulationCfg`) instead of
  always requiring Kit, and to return ``None`` instead of the Kit application. Pass ``require_kit=True`` to
  keep requiring Kit, and use ``omni.kit.app.get_app()`` where a test needs the Kit application.
