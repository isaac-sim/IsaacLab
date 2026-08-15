Added
^^^^^

* Added :meth:`~isaaclab.sim.SimulationContext.get_interactive_scene`, the public accessor for the
  scene passed to :meth:`~isaaclab.sim.SimulationContext.register_interactive_scene`.

Changed
^^^^^^^

* :class:`~isaaclab.scene_data.SceneDataProvider` now takes the owning
  :class:`~isaaclab.sim.SimulationContext` and reads the stage and the active scene through it,
  instead of resolving the singleton and holding a scene reference pushed to it at registration
  time. A provider built without a simulation context reports no scene.

Removed
^^^^^^^

* **Breaking:** Removed ``SceneDataProvider.set_interactive_scene``. The provider follows the
  simulation context it was constructed with, so registering the scene with
  :meth:`~isaaclab.sim.SimulationContext.register_interactive_scene` is enough. Delete the call.
