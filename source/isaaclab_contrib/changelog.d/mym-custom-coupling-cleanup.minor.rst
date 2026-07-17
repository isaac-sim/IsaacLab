Added
^^^^^

* Added the opt-in :mod:`isaaclab_contrib.custom_coupling` example. Import the
  module explicitly to register ``IsaacContrib-Lift-Soft-Franka-Custom-Coupling``.

Deprecated
^^^^^^^^^^

* Deprecated the MJWarp and Featherstone coupling configurations and managers
  in :mod:`isaaclab_contrib.deformable`. Use
  :class:`~isaaclab_contrib.coupling.CouplerProxyCfg` for supported
  MJWarp and VBD tasks, or import :mod:`isaaclab_contrib.custom_coupling`
  explicitly for the manual MJWarp and VBD example. Featherstone users must
  switch the rigid solver to MJWarp before migrating, or retain the deprecated
  Featherstone path.
