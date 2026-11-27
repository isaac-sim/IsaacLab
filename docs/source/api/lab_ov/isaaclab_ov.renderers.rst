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

Scene backend
-------------

OVRTX renders with its Hydra scene backend by default. ``ISAAC_LAB_OVRTX_USE_HERCULES=1`` selects its Hercules
scene backend instead and ``0`` selects Hydra, on both the internal-stage and OVStage paths. OVRTX reads the choice
when each renderer is created and keeps it for the rest of the process, so set the flag explicitly whenever
renderers in one process should use different backends. Leaving it unset keeps the current selection.

Simulation-owned stages
-----------------------

The simulation owns these detached stages and releases them after their consumers close. Each stage's
configuration selects the populated USD domains, and :class:`~isaaclab_ov.cloner.OvstageReplicateContext`
prepares its copies. OVRTX cameras and RTX visualizers use the renderer's stage in the rendering domain.
:ref:`Clone routing <ov-clone-routing>` selects stage ownership and environment overrides before
backend initialization. OVPhysX retains a separate physics stage and native replication path by default.

.. autoclass:: isaaclab_ov.stage.OvstageBackendCfg
   :show-inheritance:

.. autoclass:: isaaclab_ov.stage.OvstageBackend
   :members: populate, query, commit, reset, close
