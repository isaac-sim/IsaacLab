isaaclab.visualizers
====================

.. automodule:: isaaclab.visualizers

Custom visualizers implement ``initialize(sim, *, cameras)`` and ``reset(soft=False)``.
Initialization receives the simulation owner explicitly. Call ``super().initialize(sim, cameras=cameras)``
to retain the simulation as ``self._sim`` and bind scene data. Resolve backend-specific resources in
the implementation; core lifecycle dispatch does not import or construct them. Hard resets can reacquire
resources through ``self._sim.get_or_create_backend(...)`` and access the stage through ``self._sim.stage``.
Rendering uses the resolved resources.

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.visualizers` API.

.. currentmodule:: isaaclab.visualizers

.. autosummary::
   :nosignatures:

   BaseVisualizer
   KeyEventSource
   KeyboardSubscription
   CheckpointGhostCheckpoint
   CheckpointGhostController
   CheckpointGhostLayout
   CheckpointGhostSlot
   VisualizerCfg
   VisualizerBackendCfg
   WindowCfg
   PerspectiveCameraCfg
   SceneCameraCfg
   ImageViewCfg
   ImageView

.. autoclass:: BaseVisualizer
   :show-inheritance:

.. autoclass:: KeyEventSource
   :members: closed, add_key_listener, close

.. autoclass:: KeyboardSubscription
   :members: closed, close

.. autofunction:: add_key_callbacks

.. autoclass:: CheckpointGhostCheckpoint
   :members:
   :show-inheritance:

.. autoclass:: CheckpointGhostController
   :members:
   :show-inheritance:

.. autoclass:: CheckpointGhostLayout
   :members:
   :show-inheritance:

.. autoclass:: CheckpointGhostSlot
   :members:
   :show-inheritance:

.. autodata:: CHECKPOINT_GHOST_COLORS
   :no-value:

.. autoclass:: VisualizerCfg
   :show-inheritance:

.. autoclass:: VisualizerBackendCfg
   :show-inheritance:

.. autoclass:: PerspectiveCameraCfg
   :show-inheritance:

.. autoclass:: SceneCameraCfg
   :show-inheritance:

.. autoclass:: ImageViewCfg
   :show-inheritance:

.. autoclass:: ImageView
   :members: read, read_rgb, invalidate

.. autoclass:: WindowCfg
   :show-inheritance:
