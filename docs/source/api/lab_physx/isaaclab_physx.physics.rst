isaaclab\_physx.physics
=======================

.. automodule:: isaaclab_physx.physics

  .. rubric:: Classes

  .. autosummary::

    PhysxManager
    PhysxCfg
    SurfaceVelocity
    PhysxSurfaceVelocityTwist

.. currentmodule:: isaaclab_physx.physics

Physics Manager
---------------

.. autoclass:: PhysxManager
  :members: get_physics_sim_view
  :show-inheritance:

Physics Configuration
---------------------

.. autoclass:: PhysxCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__

Surface Velocity
----------------

.. autoclass:: SurfaceVelocity
  :members:
  :show-inheritance:

.. autoclass:: PhysxSurfaceVelocityTwist
  :members:

.. autofunction:: apply_surface_velocity_api

.. autofunction:: compute_surface_velocity_twist

.. autofunction:: resolve_surface_velocity_paths

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab_physx.physics` API.

.. currentmodule:: isaaclab_physx.physics

.. autosummary::
   :nosignatures:

   IsaacEvents

.. py:class:: IsaacEvents

   PhysX-specific simulation lifecycle events.
