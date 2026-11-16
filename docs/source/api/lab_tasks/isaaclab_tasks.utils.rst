isaaclab_tasks.utils
====================

.. automodule:: isaaclab_tasks.utils
   :members:
   :imported-members:

Reset-state preparation and selection
-------------------------------------

:func:`~isaaclab_tasks.utils.collect_reset_states` optionally prepares validated
tensor states with bounded attempts configured by
:class:`~isaaclab_tasks.utils.ResetStateCollectionCfg`. It returns states, group
indices, and diagnostics; tasks own the metadata and any tensor buffer.

:class:`~isaaclab_tasks.utils.ResetStateSampler` assigns sampled or explicitly
requested catalog IDs and records outcomes against previous assignments. Catalog
IDs may identify tensor rows, procedural recipes, or cached states. Additional
metric tensors are borrowed from the caller and updated in place; ``success_rate``
is supplied by :class:`~isaaclab_tasks.utils.SuccessMonitor`.

See :doc:`/source/how-to/reset_state_buffers` for buffer creation, validation,
physical replay, outcome attribution, and evaluation from complete task starts.

Sampling scores
---------------

:class:`~isaaclab_tasks.utils.sampling.Sampler` composes weighted scores from named
per-item metrics. :class:`~isaaclab_tasks.utils.sampling.TargetSuccessScoreCfg`
has an analytic unit peak independent of concentration, while
:class:`~isaaclab_tasks.utils.sampling.UniformScoreCfg` requires no metrics.
Custom configurations derive from
:class:`~isaaclab_tasks.utils.sampling.SamplingScoreCfg` and provide a scorer
implementing ``score(metrics, out)``. Existing Beta/uniform strategies and
success-rate-bound constructor calls retain their previous behavior.

.. automodule:: isaaclab_tasks.utils.sampling
   :members:
   :imported-members:
