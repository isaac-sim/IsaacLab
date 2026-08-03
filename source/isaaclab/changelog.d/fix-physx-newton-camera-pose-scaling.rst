Fixed
^^^^^

* Fixed :class:`~isaaclab.scene_data.SceneDataProvider` transform mapping stalling
  at high rigid-body counts, which delayed setup by minutes in scenes with
  thousands of environments.

Added
^^^^^

* Added :meth:`~isaaclab.sim.views.BaseFrameView.close` to release backend state
  authored by a frame view. Backends also release best-effort on garbage
  collection, but only an explicit close is deterministic.
