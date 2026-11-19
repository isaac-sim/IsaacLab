isaaclab_tasks.utils
====================

.. automodule:: isaaclab_tasks.utils
   :members:
   :imported-members:

Reset-state sampling
--------------------

:class:`~isaaclab_tasks.utils.ResetStateSampler` assigns reset rows using recent
episode outcomes tracked by :class:`~isaaclab_tasks.utils.SuccessMonitor`. Its
configurable sampling strategies support cached or procedural state catalogs.

See :doc:`/source/how-to/reset_state_buffers` for buffer creation, validation,
physical replay, outcome attribution, and evaluation from complete task starts.

Sampling strategies
-------------------

.. automodule:: isaaclab_tasks.utils.sampling
   :members:
   :imported-members:
