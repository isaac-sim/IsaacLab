.. currentmodule:: isaaclab.envs.reset

isaaclab.envs.reset
===================

Native storage and mask-based selection for prepared reset catalogs. All shared
reset utilities live here; tasks own collection criteria and physical restoration.
See :doc:`/source/how-to/reset_state_buffers` for graph capture, default-state
broadcasting, custom metrics, and the existing Lift event integration.

.. autosummary::

   ResetStateBuffer
   ResetStateSampler
   ResetStateSamplerCfg
   SamplingScoreCfg
   BetaSamplingScoreCfg

.. autoclass:: ResetStateBuffer
   :members:
   :exclude-members: __init__

.. autoclass:: ResetStateSampler
   :members:
   :exclude-members: __init__

.. autoclass:: ResetStateSamplerCfg
   :members:
   :exclude-members: __init__

.. autoclass:: SamplingScoreCfg
   :members:
   :exclude-members: __init__

.. autoclass:: BetaSamplingScoreCfg
   :members:
   :exclude-members: __init__
