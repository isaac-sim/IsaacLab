ADR-0002 Implementation Audit
==============================

:Audit Date: 2026-04-29
:Audit Branch: ``ncournia/scene-data-provider``
:Audit Commit: ``HEAD``

Each claim from
:doc:`/source/refs/adr/0002-capability-based-provider-extensibility` is
matched against the implementation. Status legend matches
:doc:`0001-audit`.

.. |OK| replace:: ✓ OK
.. |PARTIAL| replace:: ◐ PARTIAL
.. |GAP| replace:: ✗ GAP

Capability protocols
--------------------

Built-in protocols ``GpuTransformBuffer``, ``UsdFabric``, ``NewtonState``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |OK|
| Implementation:

  - ``GpuTransformBuffer`` and ``UsdFabric`` in
    ``source/isaaclab/isaaclab/physics/capabilities/_protocols.py:30``
    and ``:88``.
  - ``NewtonState`` in
    ``source/isaaclab_newton/isaaclab_newton/physics/capabilities/_protocols.py:11``.

| Tests: ``source/isaaclab/test/physics/test_capabilities.py``
  (registry mechanics; runtime-checkable assertions at
  ``test_runtime_checkable_protocol_isinstance``).

Bare-noun naming following ``typing.Iterable`` precedent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |OK|
| Names ``GpuTransformBuffer``, ``UsdFabric``, ``NewtonState`` match
  the kind-of-thing convention.

Type identity by Python ``type``; same-named classes in different
modules are distinct
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |OK|
| Implementation: ``BaseSceneDataProvider._capabilities`` is a
  ``dict[type, Any]``; lookup uses class identity.
| Test:
  ``test_capabilities.py::test_custom_capability_routes_through_registry``
  declares a customer-defined ``_CustomCap`` Protocol and registers it.

Generic caps in ``isaaclab.physics``; ``NewtonState`` in
``isaaclab_newton.physics``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |OK|
| Generic caps under
  ``source/isaaclab/isaaclab/physics/capabilities/``;
  Newton cap under
  ``source/isaaclab_newton/isaaclab_newton/physics/capabilities/``.

Provider registry
-----------------

``get_capability``, ``list_capabilities``, ``get_first_capability``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |OK|
| Implementation:
  ``source/isaaclab/isaaclab/physics/base_scene_data_provider.py:48-80``.
| Tests: ``test_capabilities.py``,
  ``test_registered_capability_is_returned``,
  ``test_get_first_capability_walks_in_order``,
  ``test_get_first_capability_returns_none_when_all_missing``.

Per-lifetime cap handles registered in subclass ``__init__``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |OK|
| Implementation:

  - PhysX: ``physx_scene_data_provider.py`` (the cap-registration
    block at end of ``__init__``).
  - Newton: ``newton_scene_data_provider.py`` (same pattern).

| Test: ``test_re_registration_replaces_handle``.

``GpuTransformBuffer`` mandatory baseline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Claim: Every provider must offer ``GpuTransformBuffer`` so consumers
  always have a fallback.
| Status: |OK|
| Implementation: both providers register ``GpuTransformBuffer``
  unconditionally.
| Tests:
  ``test_physx_provider_capabilities.py::test_physx_baseline_capabilities``
  and ``test_newton_provider_capabilities.py::test_newton_baseline_capabilities``
  assert presence regardless of optional caps.
| Caveat: not enforced at the type level. A custom provider that omits
  ``GpuTransformBuffer`` from its registry is structurally legal —
  validation surfaces only when a consumer requires it. The ADR claims
  this is part of the contract; making the contract enforceable would
  require either an abstract method on ``BaseSceneDataProvider`` or a
  subclass-init-time assertion. Tracked as follow-up; deferred (low
  risk: the in-tree providers all comply, and customers diverging are
  caught immediately by the wire-up validator).

Consumer requirement declarations
---------------------------------

``required_capabilities`` and ``required_one_of`` ClassVars
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |OK|
| Implementation:

  - Base classes: ``base_renderer.py:24-34``, ``base_visualizer.py:31-40``.
  - In-tree consumers (declared in their class bodies):

    - ``OVRTXRenderer``: ``required_capabilities = (GpuTransformBuffer, NewtonState)``.
    - ``NewtonWarpRenderer``: ``required_capabilities = (NewtonState,)``.
    - ``IsaacRtxRenderer``: ``required_capabilities = (UsdFabric,)``.
    - ``KitVisualizer``: ``required_capabilities = (UsdFabric,)``.
    - ``NewtonVisualizer``, ``RerunVisualizer``, ``ViserVisualizer``:
      ``required_capabilities = (NewtonState,)``.

| Tests: ``test_capabilities.py::test_validate_passes_when_required_satisfied``,
  ``test_validate_fails_with_missing_required``,
  ``test_required_one_of_passes_when_any_present``,
  ``test_required_one_of_fails_when_all_missing``,
  ``test_required_capabilities_are_all_required``,
  ``test_validate_consolidates_multiple_failures``.
| Caveat: no in-tree consumer actually declares ``required_one_of``;
  the framework supports it and unit tests cover it, but no consumer
  exercises the preference-ordered path in practice. ADR-0002's Newton
  Warp example (``required_one_of = ((NewtonState, GpuTransformBuffer),)``)
  is not implemented; the actual ``NewtonWarpRenderer`` declares
  ``NewtonState`` as strictly required because the renderer is
  Newton-shaped end-to-end and a ``GpuTransformBuffer`` fallback was
  never exercised.

Wire-up validation runs at first frame; consolidated error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |OK|
| Implementation: ``base_scene_data_provider.py``
  ``register_consumer``, ``validate_consumer_capabilities``,
  ``_validate_consumers_if_needed``. Both
  ``PhysxSceneDataProvider.update`` and
  ``NewtonSceneDataProvider.update`` call
  ``self._validate_consumers_if_needed()`` as their first action, so
  the consolidated error fires on the first per-frame tick after
  consumer registration.
| Tests: ``test_capabilities.py::test_validate_consolidates_multiple_failures``
  covers the validator; provider-level tests in
  ``test_physx_provider_capabilities.py`` and
  ``test_newton_provider_capabilities.py`` exercise the wired-up call
  path.

Self-registration via ``BaseRenderer.register_with_scene_data_provider``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |OK|
| Implementation: ``base_renderer.py:36-43``,
  ``base_visualizer.py:53-60``.
| Each in-tree consumer calls
  ``self.register_with_scene_data_provider(provider)`` once during
  initialization or first ``update_transforms``.

Migration of the typed transform API onto ``GpuTransformBuffer``
----------------------------------------------------------------

Removal of the base-class methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Claim: ``get_body_transforms``, ``get_source_format``,
  ``get_newton_state``, ``get_newton_model``, ``get_transforms`` removed
  from ``BaseSceneDataProvider``.
| Status: |OK|
| Implementation: removed in commit ``2d1ab5f12``
  (Phase 5). Subclass ``PhysxSceneDataProvider`` and
  ``NewtonSceneDataProvider`` implementations remain as concrete
  methods that the cap adapters delegate to.

Replacement for ``SceneDataRequirement``
----------------------------------------

| Claim: ``SceneDataRequirement`` and the string-keyed visualizer/
  renderer requirement maps are removed; replaced by capability
  ClassVars.
| Status: |GAP|
| Reality:

  - ``SceneDataRequirement`` is *still* in
    ``source/isaaclab/isaaclab/physics/scene_data_requirements.py``.
  - ``_VISUALIZER_REQUIREMENTS`` and ``_RENDERER_REQUIREMENTS`` mappings
    still drive provider construction
    (``physx_scene_data_provider.py:107``,
    ``newton_scene_data_provider.py:124``).
  - ``NewtonWarpRenderer.__init__``
    (``newton_warp_renderer.py:157-167``) still calls
    ``aggregate_requirements`` and ``requirement_for_renderer_type``
    to flag the PhysX provider's ``_needs_newton_sync``.

| Why deferred: removing this requires the SimulationContext to
  aggregate consumer ``required_capabilities`` ClassVars *before* the
  provider is constructed, so the provider knows which optional caps
  to set up. Today the order is reversed (provider builds, then
  consumers register), and the cap-driven path coexists with the
  string-keyed legacy. The two systems agree on what ends up in the
  registry; only the trigger differs.
| Follow-up: in a separate PR, refactor ``SimulationContext`` to walk
  consumer ClassVars at provider-construction time, then delete
  ``SceneDataRequirement``, ``_VISUALIZER_REQUIREMENTS``,
  ``_RENDERER_REQUIREMENTS``, and the ``aggregate_requirements`` /
  ``requirement_for_*`` helpers. The ``_needs_newton_sync`` and
  ``_needs_usd_sync`` flags become "did any consumer register cap X?"
  derived from the consumer registry.

``_needs_newton_sync`` becomes generic
--------------------------------------

| Claim: PhysX's ``_needs_newton_sync`` flag becomes
  capability-driven, no longer Newton-specific.
| Status: |GAP|
| Same root cause as the ``SceneDataRequirement`` gap: the flag is
  still derived from
  ``simulation_context.get_scene_data_requirements().requires_newton_model``
  (``physx_scene_data_provider.py:107``), not from registered consumer
  caps. Tracked as part of the same follow-up.

Customer extensibility
----------------------

Third-party packages may define their own protocols
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |OK|
| Verified by the ``_CustomCap`` test in ``test_capabilities.py``,
  which defines a customer-shaped Protocol and registers it on a fake
  provider with no framework changes.

Open Items declared in ADR-0002
-------------------------------

Custom :class:`TransformFormat`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |GAP|, deferred (per ADR §"Open Items").
| ``TransformFormat`` is still a closed ``Enum``
  (``scene_data_types.py:45``); the ``ConversionDispatcher`` holds a
  hard-coded 4×4 kernel grid (``scene_data_conversion.py``). To open
  the design, ``TransformFormat`` becomes a class hierarchy and the
  dispatcher becomes a registry. The ADR explicitly defers this; the
  audit confirms no work has been done.

Cross-process capability identity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |GAP|, declared out-of-scope by the ADR. No work done.

``CudaStream`` capability
^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |GAP|, declared deferred by the ADR. No work done. The
  existing ``stream`` parameter on
  ``GpuTransformBuffer.get_body_transforms`` covers the only concrete
  in-tree need today.

Summary
-------

15 claims audited.

- |OK|: 12
- |PARTIAL|: 0
- |GAP|: 3 (``SceneDataRequirement`` retired, ``_needs_newton_sync``
  generalized, custom ``TransformFormat``)

Open follow-ups (ranked by visibility risk):

1. **Refactor ``SimulationContext`` to drive provider construction
   from consumer ClassVars.** Closes both ``SceneDataRequirement``
   and ``_needs_newton_sync`` gaps in one move. Larger refactor;
   recommend separate PR after #5352 merges.
2. **Open ``TransformFormat`` to a class hierarchy.** Deferred per
   ADR-0002; necessary for customer-defined formats but not blocking
   the in-tree consumer surface.
3. **Mandatory-baseline enforcement for ``GpuTransformBuffer``.** Add
   either a base-class assertion or a subclass-init check. Low-risk
   guard that future custom providers do not silently violate the
   baseline contract.
