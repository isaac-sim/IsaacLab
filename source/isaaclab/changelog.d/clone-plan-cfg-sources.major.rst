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
  ``get_asset_prototypes(plan, path_expr=None)`` and ``get_world_prototypes(plan, path_expr=None)``
  returned prototype IDs as 1-D NumPy ``int32`` arrays, optionally filtered by declared cfg paths.
  Read cfgs and world memberships from the plan instead of unpacking topology-query results.
  Added ``get_asset_prototype_world_index(plan, asset_prototype, unique=False)`` and
  ``get_world_prototype_world_index(plan, world_prototype)`` for destination world indices.
  Asset selection accepted an index or declared-path expression and returned
  ``(world_indices, world_starts)`` with one entry per instance and shared-world-first offsets.
  Unpack both arrays for per-world instance ranges, or use ``unique=True`` to return only unique world indices.
  Renamed native-path lookup ``iter_sources`` to ``get_matched_sources``, returning a list.
  Removed cached configuration/context routing maps. Raw USD and native backend replication
  functions retained their path-based inputs.
  Clone contexts consumed destination world IDs directly without allocating dense masks.
* **Breaking:** Changed clone strategies to select world-prototype indices from relative
  weights. The sequential strategy allocated contiguous world groups; use the random
  strategy for independent sampling. ``ReplicateSession`` accepted ragged
  ``world_prototypes`` and ``weights`` instead of ``valid_set``.
