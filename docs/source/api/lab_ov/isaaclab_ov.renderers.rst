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

Borrowed stages
---------------

The simulation owns these detached stages and releases them after their visualizer consumers close.

.. autoclass:: isaaclab_ov.stage.OvstageBackendCfg
   :show-inheritance:

.. autoclass:: isaaclab_ov.stage.OvstageBackend
   :members: populate, close
