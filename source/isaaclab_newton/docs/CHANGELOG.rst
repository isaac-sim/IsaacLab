Changelog
---------

0.2.1 (2026-02-25)

Removed
^^^^^^^

imgui-bundle dependency.

0.2.0 (2026-02-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_newton.assets` module containing Newton-specific asset implementations:

  * :class:`~isaaclab_newton.assets.Articulation` and :class:`~isaaclab_newton.assets.ArticulationData`:
    Newton-specific implementation for articulated rigid body systems (e.g., robots). Extends
    :class:`~isaaclab.assets.BaseArticulation` with Newton's ``ArticulationView`` API for
    GPU-accelerated simulation of multi-joint systems.

  * :class:`~isaaclab_newton.assets.RigidObject` and :class:`~isaaclab_newton.assets.RigidObjectData`:
    Newton-specific implementation for single rigid body assets. Extends
    :class:`~isaaclab.assets.BaseRigidObject` with Newton's simulation API for efficient
    rigid body state queries and writes.

* Added warp kernel modules for fused GPU computations:

  * :mod:`isaaclab_newton.assets.kernels` — shared kernels for root state extraction,
    velocity transforms, COM/link frame conversions, and data write-back.
  * :mod:`isaaclab_newton.assets.articulation.kernels` — articulation-specific kernels
    for joint state, soft limits, actuator state updates, and friction properties.

* All ``.data.*`` properties use ``wp.array`` with structured warp types
  (``wp.vec3f``, ``wp.quatf``, ``wp.transformf``, ``wp.spatial_vectorf``),
  matching the same convention used by ``isaaclab_physx``.

* All write methods follow the ``_index`` / ``_mask`` split for explicit
  sparse-index vs. boolean-mask semantics.


0.1.0 (2026-02-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added empty package
