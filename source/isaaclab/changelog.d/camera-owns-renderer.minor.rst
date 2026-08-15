Changed
^^^^^^^

* Moved renderer backend creation into :meth:`~isaaclab.sensors.camera.Camera.__init__`, which runs
  during scene construction. A backend's ``__init__`` is its pre-physics phase and must complete
  before :meth:`~isaaclab.sim.SimulationContext.reset`, whereas sensor initialization only runs on
  ``PhysicsEvent.PHYSICS_READY``. Backend construction order still follows sensor registration
  order, and configs that compare equal still share one backend.

Removed
^^^^^^^

* **Breaking:** Removed ``InteractiveScene.initialize_renderers``. It only pre-created the backends
  that each camera now creates for itself, and its return value was unused. Delete the call; no
  replacement is needed.
