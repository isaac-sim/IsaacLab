Added
^^^^^

* Added articulation ordering utilities and optional :class:`~isaaclab.assets.ArticulationCfg`
  fields for public joint/body ordering presets.

Changed
^^^^^^^

* Changed custom :class:`~isaaclab.assets.BaseArticulation` backends to expose
  backend joint/body names and ordering maps. Existing backends continue to
  work through deprecated fallbacks; override ``backend_joint_names`` and
  ``backend_body_names`` before these properties become abstract in a future
  release.
