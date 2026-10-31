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
   OvLibrariesRenderLease

.. autoclass:: OVRTXRenderer
   :show-inheritance:
   :members: acquire_ov_libraries_render_lease

.. autoclass:: OVRTXRendererCfg
   :show-inheritance:

.. autoclass:: OVRTXBackendCfg
   :show-inheritance:

Renderer leases
---------------

An external consumer, such as the OVXR teleoperation adapter, can borrow one exact OVRTX
renderer and its private OVStage with :meth:`OVRTXRenderer.acquire_ov_libraries_render_lease`
before clone preparation. While the lease is active, ordinary camera rendering, scene writes and
renderer/stage destruction are rejected; eager scene updates are left to the lease's scene
publisher. A failed leased publication poisons publication only; the borrower can still retire
the pair, after which the renderer must be recreated.

.. autoclass:: OvLibrariesRenderLease
   :members:

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
