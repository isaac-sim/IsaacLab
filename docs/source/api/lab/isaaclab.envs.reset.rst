isaaclab.envs.reset
===================

.. automodule:: isaaclab.envs.reset

.. currentmodule:: isaaclab.envs.reset

.. autosummary::

   ResetStateBuffer

:class:`ResetStateBuffer` owns a fixed tensor tree supplied at construction.
It copies complete states through ``states`` or selected dense rows through
``get_state``. Preparation, metadata, sampling, and physical restoration belong
to the caller. Optional collection and selection helpers live in
:mod:`isaaclab_tasks.utils`.

See :doc:`/source/how-to/reset_state_buffers` for preparation, validation, loading
compatible task artifacts, selection, and replay.

.. autoclass:: ResetStateBuffer
   :members:
