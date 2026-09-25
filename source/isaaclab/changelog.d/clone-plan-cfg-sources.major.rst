Changed
^^^^^^^

* **Breaking:** Replaced the clone-plan source/environment matrix with reusable
  ``asset_prototypes``, packed ``world_prototypes`` and ``world_prototype_starts``, and
  one world-prototype index per ``destinations`` entry. Repeated memberships now represented
  repeated asset instances. Shared assets occupied the leading world ``-1`` slice.
  Use ``make_clone_plan(asset_prototypes, world_prototypes, num_worlds, shared_assets=...)``
  for pure topology, or let ``InteractiveScene`` manage planning and replication.
* **Breaking:** Moved USD names and placement out of ``ClonePlan``. Path queries now accepted
  the native mapping tuples in ``sim.clone_contexts[UsdReplicateContext].instances``;
  topology queries used ``iter_worlds(plan)``. Removed cached configuration/context routing
  maps. Raw USD and native backend replication functions retained their path-based inputs.
* **Breaking:** Changed clone strategies to select world-prototype indices from relative
  weights. The sequential strategy allocated contiguous world groups; use the random
  strategy for independent sampling. ``ReplicateSession`` accepted ragged
  ``world_prototypes`` and ``weights`` instead of ``valid_set``.
