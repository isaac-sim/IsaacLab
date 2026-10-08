isaaclab.visualizers
====================

.. automodule:: isaaclab.visualizers

Custom visualizers implement ``initialize(sim, *, cameras)`` and ``reset(soft=False)``.
Initialization receives the simulation owner explicitly. Call ``super().initialize(sim, cameras=cameras)``
to bind scene data, the authored stage, and the resource registry. Resolve backend-specific resources in
the implementation; core lifecycle dispatch does not import or construct them. Hard resets can reacquire
resources through the bound ``self._get_backend`` callable. Rendering uses the resolved resources.

Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.visualizers` API.

.. currentmodule:: isaaclab.visualizers

.. autosummary::
   :nosignatures:

   BaseVisualizer
   VisualizerCfg
   PerspectiveCameraCfg
   SceneCameraCfg

.. autoclass:: BaseVisualizer
   :show-inheritance:

.. autoclass:: VisualizerCfg
   :show-inheritance:

.. autoclass:: PerspectiveCameraCfg
   :show-inheritance:

.. autoclass:: SceneCameraCfg
   :show-inheritance:
