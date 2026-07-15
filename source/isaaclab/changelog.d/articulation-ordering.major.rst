Added
^^^^^

* Added articulation ordering utilities and optional :class:`~isaaclab.assets.ArticulationCfg`
  fields for public joint/body ordering presets.
* Added :meth:`~isaaclab.assets.Articulation.map_body_ids_to_backend` and
  :meth:`~isaaclab.assets.Articulation.map_joint_ids_to_backend` to translate
  public-order body/joint indices into backend view order for raw-view interop.
* Added the ``__backend_native_orderings__`` class attribute on
  :class:`~isaaclab.assets.BaseArticulation` so backends declare which symbolic
  ordering conventions their native order satisfies, enabling the identity
  fast path without editing the core resolvers.

Changed
^^^^^^^

* Changed custom :class:`~isaaclab.assets.BaseArticulation` backends to expose
  backend joint/body names and ordering maps. Existing backends continue to
  work through deprecated fallbacks; override
  :attr:`~isaaclab.assets.BaseArticulation.backend_joint_names` and
  :attr:`~isaaclab.assets.BaseArticulation.backend_body_names` before these
  properties become abstract in a future release.
