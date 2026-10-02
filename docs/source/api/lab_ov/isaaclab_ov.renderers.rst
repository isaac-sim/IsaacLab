isaaclab\_ov.renderers
======================

.. automodule:: isaaclab_ov.renderers

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab_ov.renderers` API.

.. currentmodule:: isaaclab_ov.renderers

.. autosummary::
   :nosignatures:

   OVRTXRenderer
   OVRTXRendererCfg
   OVRTXBackendCfg

.. autoclass:: OVRTXRenderer
   :show-inheritance:

.. autoclass:: OVRTXRendererCfg
   :show-inheritance:

.. autoclass:: OVRTXBackendCfg
   :show-inheritance:

Simulation-owned stages
-----------------------

The simulation owns these detached stages and releases them after their consumers close. Each stage's
configuration selects the populated USD domains, and :class:`~isaaclab_ov.cloner.OvstageReplicateContext`
prepares its copies. Independent renderer and visualizer stages use the rendering domain. With
``ISAAC_LAB_OVRTX_USE_OVSTAGE=1``, OVRTX borrows its rendering stage from the simulation registry.
OVPhysX retains a separate physics stage and native replication path.

.. autoclass:: isaaclab_ov.stage.OvstageBackendCfg
   :show-inheritance:

.. autoclass:: isaaclab_ov.stage.OvstageBackend
   :members: populate, query, commit, reset, close
