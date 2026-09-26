Changed
^^^^^^^

* **Breaking:** Replaced the clone-plan source/environment matrix with reusable
  ``asset_prototypes``, packed ``world_prototypes`` and ``world_prototype_starts``, and
  one world-prototype index per ``world_prototype_layout`` entry. Repeated memberships now represented
  repeated asset instances. Shared assets occupied the leading world ``-1`` slice.
  Use ``make_clone_plan(asset_prototypes, world_prototypes, num_worlds, shared_assets=...)``
  for pure topology, or let ``InteractiveScene`` manage planning and replication.
* **Breaking:** Moved USD names and placement out of ``ClonePlan``. Path queries now accepted
  the native mapping tuples in ``sim.clone_contexts[UsdReplicateContext].instances``;
  topology queries used ``get_world_prototypes(plan)``. Renamed ``iter_sources`` to
  ``get_matched_sources``; both ``get_*`` queries returned lists instead of generators.
  Removed cached configuration/context routing maps. Raw USD and native backend replication
  functions retained their path-based inputs.
  Clone contexts consumed destination world IDs directly without allocating dense masks.
* **Breaking:** Changed clone strategies to select world-prototype indices from relative
  weights. The sequential strategy allocated contiguous world groups; use the random
  strategy for independent sampling. ``ReplicateSession`` accepted ragged
  ``world_prototypes`` and ``weights`` instead of ``valid_set``.
