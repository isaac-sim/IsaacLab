Added
^^^^^

* Added :mod:`isaaclab.scene_data` sub-package consolidating
  :class:`~isaaclab.scene_data.SceneDataProvider`,
  :class:`~isaaclab.scene_data.SceneDataBackend`, and
  :class:`~isaaclab.scene_data.SceneDataFormat` in a single import location.

Changed
^^^^^^^

* **Breaking:** Moved :class:`~isaaclab.scene_data.SceneDataProvider` from
  :mod:`isaaclab.scene.scene_data_provider` and
  :class:`~isaaclab.scene_data.SceneDataBackend` /
  :class:`~isaaclab.scene_data.SceneDataFormat` from :mod:`isaaclab.physics`
  to the new :mod:`isaaclab.scene_data` sub-package. Update imports::

      # before
      from isaaclab.scene.scene_data_provider import SceneDataProvider
      from isaaclab.physics import SceneDataBackend, SceneDataFormat

      # after
      from isaaclab.scene_data import SceneDataProvider, SceneDataBackend, SceneDataFormat
