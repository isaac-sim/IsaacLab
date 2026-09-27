Changed
^^^^^^^

* **Breaking:** Replaced the clone-plan source/environment matrix with ``plan.topology``,
  a ``PrototypeWorldTopology`` holding the asset-prototype count, packed
  ``world_prototypes`` and ``world_prototype_starts``, and
  one world-prototype index per ``world_prototype_layout`` entry. Repeated memberships now represented
  repeated asset instances. Shared assets occupied the leading world ``-1`` slice.
  Use ``make_clone_plan(asset_cfgs, world_prototypes, num_worlds, shared_assets=...)``
  to construct a plan, or let ``InteractiveScene`` manage planning and replication.
  Kept host declarations, naming, and placement on ``plan.asset_cfgs``, ``plan.env_template``,
  and ``plan.positions``, outside numeric topology. Pass ``env_template=...`` and
  ``positions=...`` when constructing a plan.
* **Breaking:** Removed consumer access to USD clone contexts. Read authored paths with
  ``cloner.path.get_asset_prototype_paths(plan)`` and destination templates with
  ``cloner.path.get_world_prototype_asset_templates(plan)``. The latter returned templates
  aligned with world memberships and the existing ``world_prototype_starts`` array;
  ``include_world_indices=True`` also returned destination IDs and their per-prototype starts.
  ``cloner.path.get_asset_prototypes(plan, path_expr=None)`` and
  ``cloner.path.get_world_prototypes(plan, path_expr=None)``
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
  compose topology queries and ``cloner.path`` primitives for authored paths and destination
  names. Moved subtree-copy policy into clone backends; ``cloner.path.get_parent_indices(paths)``
  supplied strict ancestry without copying policy. Removed ``under``, ``relativize``, and ``split``;
  use ``relative_to(...) is not None``, ``match(...).suffix``, and ``template.partition("{}")``.
  Consolidated ``path`` and ``query`` into stateless namespaces in ``clone_plan.py``.
  Import them from ``isaaclab.cloner``; calls through ``cloner.path`` and ``cloner.query`` stayed unchanged.
  Removed cached configuration/context routing maps. Raw USD and native backend replication
  functions retained their path-based inputs.
  Clone contexts consumed destination world IDs directly without allocating dense masks.
* Added explicit ``cloner.to_warp(plan.topology, device)`` materialization of ``PrototypeWorldTopology``.
  Kept asset cfg references on the host plan and materialized only the three numeric arrays.
  NumPy planning remained on the host; Warp queries used preallocated outputs and supported
  CUDA graph replay without implicit uploads or readbacks. Materialize once during initialization
  and share the returned topology; no automatic device cache or mutable mirror was introduced.
* **Breaking:** Changed clone strategies to select world-prototype indices from relative
  weights. The sequential strategy allocated contiguous world groups; use the random
  strategy for independent sampling. ``ReplicateSession`` accepted ragged
  ``world_prototypes`` and ``weights`` instead of ``valid_set``.
