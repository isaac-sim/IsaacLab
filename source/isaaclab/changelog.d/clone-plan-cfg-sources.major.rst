Changed
^^^^^^^

* **Breaking:** Replaced the clone-plan source/environment matrix with ``plan.topology``,
  a ``PrototypeWorldTopology`` holding reusable ``asset_prototypes``, packed
  ``world_prototypes`` and ``world_prototype_starts``, and
  one world-prototype index per ``world_prototype_layout`` entry. Repeated memberships now represented
  repeated asset instances. Shared assets occupied the leading world ``-1`` slice.
  Use ``make_clone_plan(asset_prototypes, world_prototypes, num_worlds, shared_assets=...)``
  to construct a plan, or let ``InteractiveScene`` manage planning and replication.
  Kept world origins on ``plan.positions``, separate from topology; pass ``positions=...``
  when constructing a plan instead of storing placement on ``UsdReplicateContext``.
* **Breaking:** Moved USD names out of ``ClonePlan``. Native bindings remained
  in ``sim.clone_contexts[UsdReplicateContext].instances``;
  ``cloner.path.get_asset_prototypes(plan.topology, path_expr=None)`` and
  ``cloner.path.get_world_prototypes(plan.topology, path_expr=None)``
  returned prototype IDs as 1-D NumPy ``int32`` arrays, optionally filtered by declared cfg paths.
  Read cfgs and world memberships from the plan instead of unpacking topology-query results.
  Added ``get_asset_prototype_world_index(plan.topology, asset_prototype)`` returning
  ``(world_indices, world_starts)`` with one entry per instance and shared-world-first offsets.
  Use ``get_asset_prototype_unique_world_index(plan.topology, asset_prototype)`` or
  ``get_world_prototype_world_index(plan.topology, world_prototype)`` for unique destination world indices.
  All three accepted numeric IDs only and returned flat ``int32`` indices with ``int64``
  boundaries shaped ``[num_queries, num_worlds + 2]``. Resolve expressions with ``cloner.path``
  before querying. Repeated input IDs retained separate results; a scalar formed a one-query batch.
  Removed ``iter_sources``, ``path_env_ids``, ``path_to_clone``, and ``path_to_source`` from ``cloner.query``;
  compose topology queries and ``cloner.path`` primitives with existing native bindings
  for authored paths and destination names. Subtree-copy selection belonged to USD replication,
  not the topology query API.
  Removed cached configuration/context routing maps. Raw USD and native backend replication
  functions retained their path-based inputs.
  Clone contexts consumed destination world IDs directly without allocating dense masks.
* Added explicit ``cloner.to_warp(plan.topology, device)`` materialization of ``PrototypeWorldTopology``.
  Retained asset cfg references on the host and materialized only the three numeric arrays.
  NumPy planning remained on the host; Warp queries used preallocated outputs and supported
  CUDA graph replay without implicit uploads or readbacks. Materialize once during initialization
  and share the returned topology; no automatic device cache or mutable mirror was introduced.
* **Breaking:** Changed clone strategies to select world-prototype indices from relative
  weights. The sequential strategy allocated contiguous world groups; use the random
  strategy for independent sampling. ``ReplicateSession`` accepted ragged
  ``world_prototypes`` and ``weights`` instead of ``valid_set``.
