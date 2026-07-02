Changed
^^^^^^^

* Changed the automate and factory environments to apply the armature inertia offset through
  :meth:`~isaaclab.assets.Articulation.set_inertias_index` instead of the raw tensor view, so
  :attr:`~isaaclab.assets.ArticulationData.body_inertia` stays coherent.

Fixed
^^^^^

* Fixed the Forge environment's force-sensor body index to resolve against
  :attr:`~isaaclab.assets.Articulation.backend_body_names`, since it indexes the
  backend-order ``root_view.get_link_incoming_joint_force()`` array. With a
  non-identity :attr:`~isaaclab.assets.ArticulationCfg.body_ordering`, the previous
  public-order index silently read another link's wrench.
