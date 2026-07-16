Fixed
^^^^^

* Fixed :class:`~isaaclab.scene.InteractiveScene` raising ``RuntimeError: Clone planning did
  not assign spawn_path`` for assets that clone combinations claim but never activate in any
  environment (e.g. heterogeneous scenes where a zero-weight
  :class:`~isaaclab.cloner.InclusionSet` claims unused asset slots). Such assets are now
  skipped instead of constructed.
