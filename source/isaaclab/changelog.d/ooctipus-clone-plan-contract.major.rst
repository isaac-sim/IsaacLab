Changed
^^^^^^^

* **Breaking:** Routed clone backends through one :class:`~isaaclab.cloner.ClonePlan` using its
  new ``context_rows`` mapping. Replace ``UsdReplicateContext.queue(...)`` or
  ``queue_mapping(...)`` followed by ``replicate()`` with
  ``UsdReplicateContext.replicate(plan)``. Standalone callers may continue to use
  :func:`~isaaclab.cloner.usd_replicate`. Remove the former ``stage`` argument from
  :func:`~isaaclab.cloner.replicate` and :class:`~isaaclab.cloner.ReplicateSession`.
  Register custom ``cloning_contexts`` with ``sim.get_or_create_backend(ContextType, ...)``
  before dispatch; built-in physics managers register their own context.
