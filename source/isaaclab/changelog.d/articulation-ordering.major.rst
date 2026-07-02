Added
^^^^^

* Added articulation ordering utilities and optional :class:`~isaaclab.assets.ArticulationCfg`
  fields for public joint/body ordering presets.

Changed
^^^^^^^

* Changed custom :class:`~isaaclab.assets.BaseArticulation` backends to expose
  backend joint/body names and ordering maps. Existing backends continue to
  work through deprecated fallbacks; override
  :attr:`~isaaclab.assets.BaseArticulation.backend_joint_names` and
  :attr:`~isaaclab.assets.BaseArticulation.backend_body_names` before these
  properties become abstract in a future release.
