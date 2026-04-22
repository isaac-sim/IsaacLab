ADR-0001: Scene Data Provider Redesign
=======================================

:Status: Accepted
:Date: 2026-04-22
:Authors: Nathan Cournia, Daniela Hasenbring

Context
-------

Isaac Lab bridges multiple physics simulators (PhysX, Newton) with multiple
renderers (OVRTX, Newton Warp, Isaac RTX) and visualizers (Kit, Newton, Rerun,
Viser). The Scene Data Provider (SDP) is the component responsible for
transferring simulation data—primarily rigid-body transforms—between these
subsystems.

The existing SDP has several problems:

1. **Untyped interface.** ``BaseSceneDataProvider.get_transforms()`` returns
   ``dict[str, Any] | None``. Consumers must guess the format of the returned
   data and perform ad-hoc conversions.

2. **Duplicated conversion logic.** Each consumer implements its own format
   conversions. The OVRTX renderer converts ``transformf`` → ``mat44d``, the
   PhysX provider converts ``vec3 + quat`` → ``transformf``, and so on.
   Adding a new consumer means writing another set of conversions.

3. **No zero-copy guarantees.** Data is copied unnecessarily when the
   simulator's native format already matches what the consumer needs.

4. **No CUDA stream support.** All GPU work runs synchronously on the default
   stream. Renderers like OVRTX that manage their own streams cannot enqueue
   deferred work.

5. **No buffer lifetime management.** Buffers are allocated per-call with no
   pre-allocation, no reuse across frames, and no change detection to skip
   redundant conversions.

Decision
--------

We introduce a **typed format-negotiation system** for the SDP. The key
components are:

Transform format types
^^^^^^^^^^^^^^^^^^^^^^

A set of concrete dataclasses representing the four transform representations
used in Isaac Lab:

- ``Vec3QuatTransforms`` — separate position (``wp.vec3``) and quaternion
  (``wp.quatf``) arrays. Native to PhysX.
- ``Vec3Mat33Transforms`` — separate position and 3×3 rotation matrix arrays.
- ``TransformArrayData`` — packed 7-float transforms (``wp.transformf``).
  Native to Newton.
- ``Mat44Transforms`` — 4×4 homogeneous matrices (``wp.mat44f`` or
  ``wp.mat44d``). Native to OVRTX.

Each type carries metadata: ``QuaternionConvention`` (XYZW or WXYZ) and
``MatrixLayout`` (row-major or column-major).

Centralized GPU conversion kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A ``ConversionDispatcher`` selects and launches the appropriate Warp kernel
for any ``(source_format, target_format)`` pair. All 16 kernel paths
(4 × 4 grid) are implemented, plus quaternion convention swizzle and matrix
layout transpose kernels. All kernels accept an optional ``index_map`` for
subset scatter writes.

This consolidates conversion logic that was previously duplicated across
``physx_scene_data_provider.py``, ``ovrtx_renderer_kernels.py``, and
individual visualizers.

Buffer pool with generation-based caching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TransformBufferPool`` pre-allocates GPU output buffers and reuses them
across frames. A generation counter (incremented each ``update()`` call)
enables three fast paths:

1. **Format-match passthrough (zero work).** When the consumer's requested
   format matches the simulator's native format, the simulator's own GPU
   buffer is returned directly. No copy, no kernel launch, no allocation.
   Example: Newton backend → Newton Warp renderer, both using
   ``TransformFormat.TRANSFORM``.

2. **Same-frame conversion cache.** If the same conversion was already
   performed this frame (same generation), the cached result is returned.
   Example: two consumers both request ``MAT44`` → kernel runs once.

3. **Cross-frame buffer reuse.** When generation changes, the conversion
   kernel re-runs but writes into the same pre-allocated buffer—no
   allocation.

CUDA stream support
^^^^^^^^^^^^^^^^^^^

``get_body_transforms()`` accepts an optional ``stream`` parameter. When
provided, conversion kernels are launched on that stream via
``wp.ScopedStream``, enabling deferred execution. This follows the pattern
established in ``NewtonManager`` and aligns with OVRTX's stream-based
rendering pipeline.

Enhanced base class
^^^^^^^^^^^^^^^^^^^

``BaseSceneDataProvider`` gains two new methods:

- ``get_body_transforms(target_format, ...)`` — the primary typed API.
  Consumers declare the format they need; the SDP converts from the
  simulator's native format.
- ``get_source_format()`` — returns the simulator's native transform format,
  allowing consumers to request a passthrough-compatible format.

Both methods have default implementations returning ``None`` for backward
compatibility.

Consequences
------------

Positive
^^^^^^^^

- **Zero-copy GPU-only path.** When formats match, data flows from simulator
  to consumer without any GPU memory copies or kernel launches.
- **No duplicated conversions.** All format conversion logic lives in one
  place (``scene_data_kernels.py``). Adding a new consumer requires zero
  conversion code.
- **Type safety.** Consumers receive concrete ``TransformData`` subclasses
  instead of untyped dicts. Format mismatches are caught at call time.
- **Performance.** Pre-allocated buffers, generation-based caching, and
  CUDA stream support eliminate per-frame allocation overhead and enable
  asynchronous execution.
- **Extensibility.** Adding a new format requires adding one enum variant,
  one dataclass, and the conversion kernels to/from existing formats.

Negative
^^^^^^^^

- **Interface change.** ``BaseSceneDataProvider`` gains new methods. Existing
  subclasses continue to work (default implementations return ``None``), but
  must implement the new methods to participate in the typed API.
- **Kernel maintenance.** The 4×4 conversion grid means 16 kernel paths to
  maintain. This is mitigated by each kernel being small (~10 lines) and
  using Warp's built-in transform functions.
- **Passthrough ownership semantics.** When ``allow_passthrough=True``,
  consumers receive a reference to the simulator's buffer and must not
  mutate it. This is documented but not enforced at runtime.

Alternatives Considered
-----------------------

Per-consumer conversion library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each renderer/visualizer could use a shared utility library for conversions
but manage its own buffer allocation and caching. This was rejected because
it still duplicates buffer management logic and makes the zero-copy
passthrough optimization harder to implement centrally.

Opaque binary buffers with layout descriptors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The SDP could pass opaque byte buffers with runtime layout descriptors,
making the SDP fully format-agnostic. This was rejected as overly complex
for the four well-known formats used in Isaac Lab, and it would prevent
compile-time type checking of Warp kernel inputs.

References
----------

- Proof-of-concept implementation:
  ``daniela-hase/IsaacLab`` branch ``dev/scene-data-provider-api``
- Design meeting (2026-03-09): ``isaac-lab-render-standup.20260309.md``
- Design meeting (2026-04-02): ``new-scene-data-provider-transcript.md``
