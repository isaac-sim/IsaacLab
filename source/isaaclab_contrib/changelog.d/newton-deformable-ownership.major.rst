Removed
^^^^^^^

* **Breaking:** Removed ``isaaclab_contrib.deformable`` after moving the implementation to Newton.
  Import ``DeformableObject`` and ``DeformableObjectData`` from :mod:`isaaclab_newton.assets` instead,
  or use the backend-independent :class:`isaaclab.assets.DeformableObject`.
