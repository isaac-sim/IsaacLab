ADR-0001 Implementation Audit
==============================

:Audit Date: 2026-04-29
:Audit Branch: ``ncournia/scene-data-provider``
:Audit Commit: ``HEAD``

Each claim from :doc:`/source/refs/adr/0001-scene-data-provider-redesign`
is matched against the implementation. Status legend:

- |OK| — implemented as described.
- |PARTIAL| — implemented with caveats noted.
- |GAP| — claimed but not implemented (with proposed follow-up).
- |MOVED| — relocated by ADR-0002.

.. |OK| replace:: ✓ OK
.. |PARTIAL| replace:: ◐ PARTIAL
.. |GAP| replace:: ✗ GAP
.. |MOVED| replace:: ↪ MOVED

Decision claims
---------------

Transform format types
^^^^^^^^^^^^^^^^^^^^^^

| Claim: ``Vec3QuatTransforms``, ``Vec3Mat33Transforms``,
  ``TransformArrayData``, ``Mat44Transforms`` dataclasses with
  ``QuaternionConvention`` and ``MatrixLayout`` metadata.
| Status: |OK|
| Implementation: ``source/isaaclab/isaaclab/physics/scene_data_types.py:65-152``.
| Tests: ``source/isaaclab/test/physics/test_scene_data_types.py``.

Centralized GPU conversion kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Claim: ``ConversionDispatcher`` selects and launches Warp kernels for
  any ``(source_format, target_format)`` pair. All 16 paths plus
  quaternion swizzle and matrix-layout transpose, with optional
  ``index_map`` for subset writes.
| Status: |OK|
| Implementation: ``source/isaaclab/isaaclab/physics/scene_data_conversion.py:39``
  (``ConversionDispatcher``); kernels in
  ``source/isaaclab/isaaclab/physics/scene_data_kernels.py`` (576 lines).
| Tests: ``source/isaaclab/test/physics/test_scene_data_kernels.py`` and
  ``test_scene_data_conversion.py``.

Buffer pool with generation-based caching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Claim: ``TransformBufferPool`` with three fast paths — format-match
  passthrough, same-frame conversion cache, cross-frame buffer reuse.
| Status: |OK|
| Implementation: ``source/isaaclab/isaaclab/physics/scene_data_buffers.py:74``
  (``TransformBufferPool``).
| Tests: ``source/isaaclab/test/physics/test_scene_data_buffers.py``.

CUDA stream support
^^^^^^^^^^^^^^^^^^^

| Claim: ``get_body_transforms()`` accepts an optional ``stream``
  parameter; conversion kernels launch via ``wp.ScopedStream``.
| Status: |PARTIAL|
| Implementation: parameter present on
  ``GpuTransformBuffer.get_body_transforms`` (capabilities/_protocols.py)
  and on the provider implementations
  (``physx_scene_data_provider.py``,
  ``newton_scene_data_provider.py``); ``ConversionDispatcher`` accepts
  the stream and uses ``wp.ScopedStream``.
| Caveat: the existing 56 unit tests do not exercise the stream
  parameter — every test path uses the default stream. The contract
  works in the abstract but is not under regression coverage.
| Follow-up: add a stream test in
  ``source/isaaclab/test/physics/test_scene_data_conversion.py`` that
  asserts the dispatcher submits work to a non-default stream when one
  is provided.

Enhanced base class — ``get_body_transforms()`` and ``get_source_format()``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Claim: ``BaseSceneDataProvider`` gains
  ``get_body_transforms(target_format, ...)`` and
  ``get_source_format()``.
| Status: |MOVED|
| Note: the methods were retained on the base class through Phase 4 of
  the ADR-0002 migration, then removed in Phase 5
  (commit ``2d1ab5f12``). The same API now lives on the
  :class:`~isaaclab.physics.GpuTransformBuffer` capability protocol;
  consumers acquire it via
  ``provider.get_capability(GpuTransformBuffer)``.

Consequences claims
-------------------

Zero-copy GPU-only path
^^^^^^^^^^^^^^^^^^^^^^^

| Claim: When formats match, data flows from simulator to consumer with
  no GPU copies or kernel launches.
| Status: |PARTIAL|
| Implementation: Newton provider's ``Vec3QuatTransforms`` /
  ``TransformArrayData`` paths return source buffers directly when
  ``allow_passthrough=True``.
| Caveat: no automated test asserts pointer identity between the
  consumer's buffer and the simulator's source buffer. The
  Newton-renderer migration in Phase 4 sets ``allow_passthrough=True``
  but does not verify the no-copy property at runtime.
| Follow-up: add the snapshot regression tests proposed in the migration
  plan, and include a ``data_ptr`` identity assertion for the
  Newton-sim + Newton-renderer combo.

No duplicated conversions
^^^^^^^^^^^^^^^^^^^^^^^^^

| Claim: All format conversion logic lives in
  ``scene_data_kernels.py``.
| Status: |OK|
| Audit: ``grep -rn "convert_camera_frame_orientation_convention\|sync_newton_transforms"``
  shows the only remaining per-consumer conversion is OVRTX's
  ``create_camera_transforms_kernel`` (camera-pose convention conversion,
  outside transform-format scope) and Newton's
  ``RenderData._update_transforms`` (camera-only). The
  ``sync_newton_transforms_kernel`` referenced as duplicated logic in
  the ADR was removed in commit ``2d1ab5f12``.

Type safety
^^^^^^^^^^^

| Claim: Consumers receive concrete ``TransformData`` subclasses; format
  mismatches caught at call time.
| Status: |OK|
| Implementation: ``GpuTransformBuffer.get_body_transforms`` returns
  ``TransformData | None``; concrete return types are
  ``Vec3QuatTransforms`` / ``Vec3Mat33Transforms`` /
  ``TransformArrayData`` / ``Mat44Transforms``.
| Caveat: type narrowing relies on consumers checking
  ``isinstance(td, Mat44Transforms)`` (etc.) before reading
  format-specific attributes. Documented in the dataclass docstrings
  but not enforced.

Performance — pre-allocated buffers, generation cache, streams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |PARTIAL|
| Same caveat as the CUDA stream claim: behaviour is implemented but
  no benchmark or regression test pins it. A multi-frame test asserting
  zero allocations after warm-up would close the gap.

Extensibility — new format requires one enum, one dataclass, kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Claim: Adding a new format requires one ``TransformFormat`` enum
  variant, one dataclass, and conversion kernels to/from existing
  formats.
| Status: |GAP|
| ``TransformFormat`` is currently a closed Python ``Enum``
  (``scene_data_types.py:45``); customers cannot add a new variant
  without modifying the framework. The ``ConversionDispatcher``
  similarly hard-codes a 4×4 grid.
| Follow-up: per ADR-0002 Open Items §"Custom TransformFormat", convert
  the enum to an open class hierarchy with a kernel registry. Deferred
  to a follow-up PR after #5352 merges.

Negative consequences claims
----------------------------

Interface change — subclasses with default ``None`` returns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |MOVED|
| The default-``None`` shim on ``BaseSceneDataProvider`` was removed in
  Phase 5 of the ADR-0002 migration. New consumer code goes through the
  capability registry; the back-compat path is no longer needed.

Kernel maintenance — 16 paths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Status: |OK|
| 16 kernel functions live in ``scene_data_kernels.py``; each is small
  (~10–20 lines). Maintenance cost matches the ADR's prediction.

Passthrough ownership semantics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| Claim: Documented but not enforced at runtime.
| Status: |OK|
| Implementation: ``GpuTransformBuffer.get_body_transforms`` docstring
  in ``capabilities/_protocols.py`` documents the no-mutate contract;
  ``Vec3QuatTransforms`` etc. expose mutable Warp arrays with no
  runtime guard. As designed.

Post-merge clarifications added in this audit
---------------------------------------------

The 2026-04-29 update to ADR-0001 documents three components that landed
without ADR coverage:

- ``SceneDataRequirement`` and ``VisualizerPrebuiltArtifacts``
  (``scene_data_requirements.py``).
- The PhysX→Newton sync bridge in
  ``PhysxSceneDataProvider`` (``physx_scene_data_provider.py:107, 144-146``).
- The ``SceneDataProvider`` factory (``scene_data_provider.py:20``).

ADR-0002 schedules ``SceneDataRequirement`` and
``_needs_newton_sync`` for replacement once the capability framework
becomes the sole driver of provider service registration; that
transition is tracked in :doc:`0002-audit`.

Summary
-------

13 claims audited.

- |OK|: 6
- |PARTIAL|: 4 (CUDA stream, zero-copy, performance, type safety)
- |GAP|: 1 (custom ``TransformFormat``)
- |MOVED|: 2 (typed-API methods, default-``None`` consequence)

Open follow-ups:

1. Stream-aware test in ``test_scene_data_conversion.py``.
2. ``data_ptr`` identity assertion for the Newton passthrough path.
3. ``TransformFormat`` open-class refactor (deferred to follow-up PR).
4. Allocation/benchmarks regression coverage.
