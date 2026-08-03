Fixed
^^^^^

* Fixed :class:`~isaaclab.scene_data.SceneDataProvider` transform mapping stalling
  at high rigid-body counts, which delayed setup by minutes in scenes with
  thousands of environments.
