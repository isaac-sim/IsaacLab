Changed
^^^^^^^

* **Breaking:** Routed clone backends through one :class:`~isaaclab.cloner.ClonePlan` using its
  ``context_rows`` mapping. Replace ``UsdReplicateContext.queue(...)`` or
  ``queue_mapping(...)`` followed by ``replicate()`` with
  ``UsdReplicateContext.replicate(plan)``. Standalone callers can use
  :func:`~isaaclab.cloner.usd_replicate`. Removed the former ``stage`` argument from
  :func:`~isaaclab.cloner.replicate` and :class:`~isaaclab.cloner.ReplicateSession`.
  Custom ``cloning_contexts`` must be registered with
  ``sim.get_or_create_backend(ContextType, ...)`` before dispatch; built-in physics managers
  register their own context. Changed clone-plan arrays, grid transforms, and clone strategies
  to NumPy and removed their ``device`` arguments, including ``CloneCfg.device``. Changed
  :func:`~isaaclab.cloner.usd_replicate` to accept NumPy arrays instead of tensors. Convert an
  array to a runtime tensor only where its consuming component requires one.
