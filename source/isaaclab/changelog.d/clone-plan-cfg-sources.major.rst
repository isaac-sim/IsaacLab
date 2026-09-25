Changed
^^^^^^^

* **Breaking:** Changed ``ClonePlan.sources`` to retain asset and sensor configurations,
  and ``destinations`` to select a spawner variant per configuration and environment
  (``-1`` for no replicated instance). Replaced ``cfg_rows`` and ``context_rows`` with
  direct configuration references and ``context_source_indices``. Use
  ``cloner.query.replication_mapping(plan)`` to derive source paths, destination templates,
  and the boolean mapping previously stored in ``clone_mask``.
* **Breaking:** Derived ``ClonePlan.global_paths`` from shared configurations. Pass those
  configurations to ``make_clone_plan`` or ``ReplicateSession`` instead of a separate
  ``global_paths`` argument. Raw ``usd_replicate`` and native backend replication APIs
  retained their path-based inputs.
