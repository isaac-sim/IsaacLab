Fixed
^^^^^

* Fixed :class:`~isaaclab_newton.physics.NewtonManager` building the PhysX-backend
  shadow Newton visualization model (used by Newton-native visualizers/renderers such
  as viser, rerun, and Newton GL/RTX) with USD-authored self-collision filter pairs.
  These pairs scale with the number of cloned environments and could reach billions
  of entries, causing ``ModelBuilder.finalize()`` to run out of memory. The shadow
  model never runs collision detection, so its collision filter pairs are now cleared
  before finalization.
