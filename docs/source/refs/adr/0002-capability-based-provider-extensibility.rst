ADR-0002: Capability-Based Provider Extensibility
==================================================

:Status: Accepted
:Date: 2026-04-28
:Authors: Nathan Cournia
:Supersedes (in part): ADR-0001 — channel and consumer-requirement framing

Context
-------

ADR-0001 introduced a typed format-negotiation API on the Scene Data
Provider (SDP). It defined four transform formats, a conversion dispatcher,
and a buffer pool with format-match passthrough. That work assumed a single
implicit transport surface: a GPU buffer of typed ``TransformData``.

Two extensibility gaps surfaced after the typed API landed.

1. **Transport surfaces are hard-coded to "GPU buffer of TransformData".**

   Several in-tree consumers do not read transforms from a GPU buffer:

   - :class:`~isaaclab_physx.renderers.IsaacRtxRenderer` reads transforms
     from USD Fabric directly. Its ``update_transforms`` is a no-op.
   - The Kit visualizer reads transforms from USD Fabric via Hydra.

   These are valid fast paths — when PhysX writes Fabric natively, no
   SDP work is needed at all. But the ADR-0001 design has no name for
   them. Their fast paths exist only as accidental no-ops in
   ``update_transforms`` implementations.

   Customer extensions push this further. A customer authoring a vertically
   integrated provider+consumer pair (e.g. shared-memory transport between
   a custom physics backend and a custom remote viewer) must be able to
   register a fast path of their own without modifying framework code. An
   enum-keyed transport type does not allow this.

2. **Consumer requirements are stringly-typed and validated implicitly.**

   :class:`~isaaclab.physics.scene_data_requirements.SceneDataRequirement`
   tracks per-consumer needs as boolean flags
   (``requires_newton_model``, ``requires_usd_stage``) keyed by visualizer
   or renderer type-name strings. This works for a fixed in-tree consumer
   set but is not extensible — a customer-authored consumer cannot declare
   a new requirement type without modifying the framework's lookup tables.

   Failure modes are also implicit. When a consumer's needs are not met,
   the failure surfaces as a ``None`` return deep in an update loop or an
   ``AttributeError`` on a missing method — never as a coherent
   wire-up-time error.

Decision
--------

We introduce a **capability-based extensibility model** for the SDP.

Capability protocols
^^^^^^^^^^^^^^^^^^^^

Each transport surface or provider service is expressed as a
``runtime_checkable`` ``typing.Protocol``. The four built-in capabilities are:

- ``GpuTransformBuffer`` — typed GPU buffer pull. Hosts the
  :meth:`get_body_transforms` and :meth:`get_source_format` API previously
  on :class:`BaseSceneDataProvider` (ADR-0001).
- ``UsdFabric`` — USD prim attribute storage with explicit freshness.
  Exposes ``ensure_current(stream=None) -> None``.
- ``NewtonState`` — direct access to the active Newton ``State`` and
  ``Model`` objects. Read-only. Replaces the legacy
  :meth:`BaseSceneDataProvider.get_newton_state` and
  :meth:`get_newton_model` methods.

Capabilities are identified by Python ``type`` identity. Two protocols with
the same name in different modules are distinct. This is intentional and
matches the ABI risk: code that imports ``acme.shm.SharedMemoryChannel``
expects compatibility with other code importing the same class, not with
an unrelated class that happens to share a name.

Generic capabilities (``GpuTransformBuffer``, ``UsdFabric``) live in
``isaaclab.physics.capabilities``. Backend-specific capabilities live in
their owning extension package (``NewtonState`` lives in
``isaaclab_newton.physics.capabilities``).

Provider registry
^^^^^^^^^^^^^^^^^

:class:`BaseSceneDataProvider` gains:

.. code-block:: python

    def get_capability(self, cap_type: type[T]) -> T | None: ...
    def list_capabilities(self) -> frozenset[type]: ...
    def get_first_capability(
        self, *cap_types: type
    ) -> tuple[type, Any] | None: ...

Providers register handles in ``__init__`` by populating an internal
``{type: handle}`` mapping. Handles are per-provider singletons, valid for
the SDP's lifetime. ``get_capability`` returns the registered handle or
``None``; ``list_capabilities`` returns all registered types;
``get_first_capability`` walks an ordered preference list and returns the
first match (used by consumers that prefer one cap but accept a fallback).

The ``GpuTransformBuffer`` capability is **mandatory** for every provider.
This guarantees every consumer can fall back to typed buffer pull when its
preferred cap is unavailable, and makes ``GpuTransformBuffer`` the universal
lingua franca across mixed provider/consumer pairs.

Consumer requirement declarations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Renderer and visualizer base classes gain two ``ClassVar`` attributes:

.. code-block:: python

    class NewtonWarpRenderer(BaseRenderer):
        required_capabilities: ClassVar[tuple[type, ...]] = ()
        required_one_of: ClassVar[tuple[tuple[type, ...], ...]] = (
            (NewtonState, GpuTransformBuffer),
        )

- ``required_capabilities`` lists capabilities all of which must be
  present.
- ``required_one_of`` lists groups; from each group at least one entry must
  be present.

Validation runs once at SDP wire-up after all consumers are constructed.
A single consolidated error lists every consumer's unmet requirements.
Consumers register themselves with the SDP from
:class:`~isaaclab.renderers.BaseRenderer.__init__` and
:class:`~isaaclab_visualizers.base.BaseVisualizer.__init__`.

Migration of the typed transform API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The typed transform API introduced in ADR-0001 moves off
:class:`BaseSceneDataProvider` and onto the ``GpuTransformBuffer`` protocol.

Consumer code changes from:

.. code-block:: python

    transforms = provider.get_body_transforms(TransformFormat.MAT44, ...)

to:

.. code-block:: python

    cap = provider.get_capability(GpuTransformBuffer)
    transforms = cap.get_body_transforms(TransformFormat.MAT44, ...)

The base-class methods ``get_body_transforms``, ``get_source_format``,
``get_newton_state``, ``get_newton_model``, and ``get_transforms`` are
removed. Per ``AGENTS.md``, the deprecation requirement applies only to
released API; the entire SDP stack — including the legacy methods
introduced by PR #4797 — is on this branch and unreleased, so removal is
clean.

Replacement for ``SceneDataRequirement``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The string-keyed requirement mechanism in
``isaaclab.physics.scene_data_requirements`` is removed. The new
declarative attributes on consumer base classes replace it cleanly:

- ``requires_usd_stage=True`` becomes
  ``required_capabilities = (UsdFabric,)``.
- ``requires_newton_model=True`` becomes
  ``required_capabilities = (NewtonState,)``.
- ``preferred_transform_formats`` moves into the consumer's own logic — the
  consumer chooses its target format when it calls
  ``GpuTransformBuffer.get_body_transforms``.

Consumer protocol
^^^^^^^^^^^^^^^^^

Each consumer's per-frame update follows this shape:

.. code-block:: python

    def update_transforms(self) -> None:
        # Optional: ensure native channel is current. No-op when the
        # provider writes the channel natively.
        if self._fabric is not None:
            self._fabric.ensure_current()
            return

        # Typed buffer-pull path.
        transforms = self._gpu.get_body_transforms(
            self._target_format, ...
        )
        if transforms is not None:
            self._copy(transforms)

Consequences
------------

Positive
^^^^^^^^

- **Customer extensibility.** Third-party packages may define their own
  capability protocols and ship matched provider/consumer pairs. The SDP
  forwards opaquely; the framework never needs to know what the customer's
  cap means.

- **Native-channel fast paths are first-class.** PhysX→Fabric→Isaac RTX is
  no longer an accidental ``update_transforms`` no-op; ``UsdFabric``
  formalizes the contract and surfaces it for testing and introspection.

- **Failure mode is startup-time and explicit.** A consumer with unmet
  requirements fails at SDP wire-up with one consolidated message listing
  every missing capability across every consumer.

- **Two parallel mechanisms collapse into one.** ``SceneDataRequirement``
  and the implicit "this method may exist on the base class" pattern both
  disappear. The capability registry is the single source of truth.

- **The PhysX provider's ``_needs_newton_sync`` flag becomes generic.**
  Today it answers "is any consumer asking for Newton state?" tied
  specifically to the Newton sync bridge. Generalised: "did any consumer
  query a non-native cap?" — keyed by capability type, not Newton-specific.

- **Introspection and debug.** ``provider.list_capabilities()`` is a
  useful debugging tool. When a consumer's expected fast path silently
  doesn't engage, the registry shows what is actually offered.

Negative
^^^^^^^^

- **One layer of indirection.** Consumers go through ``get_capability()``
  rather than calling the base class directly. The cost is a single dict
  lookup per cap, performed once and cached for the SDP lifetime, so
  per-frame overhead is zero.

- **Type-identity coupling.** A protocol class is the lookup key. Two
  implementations of "a shared-memory channel" in different packages will
  not match unless they import the same class. This is correct behaviour
  but customers must understand it; a versioned cap (``MyCapV2``) is
  introduced as a new class, not by mutating the existing one.

- **No semantic schema enforcement.** A provider claiming to satisfy a
  protocol is checked structurally by ``runtime_checkable``. Missing
  methods are caught; semantically wrong implementations are not.

- **Wire-up registration adds machinery.** Renderers and visualizers must
  self-register with the SDP. The base classes do this in ``__init__`` so
  individual subclasses are unaffected, but it does mean the SDP holds
  weak references to consumers for the validator to walk.

Alternatives Considered
-----------------------

String-keyed capability registry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Providers expose a ``dict[str, Any]`` keyed by capability name. Rejected:
no static type checking, no IDE autocomplete on returned handles, and
naming collisions across packages are unprotected.

Enum extension at import time
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Keep ``TransportChannel`` as an ``Enum`` and let customers register
additional values. Rejected: Python enums are intentionally closed;
extending them is a known anti-pattern and the value-object would still
have nowhere to host its behavioural contract.

Constructor injection of capabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework wires required capability handles into each consumer at
construction time, removing the runtime ``get_capability`` query.
Rejected: heavier machinery without commensurate benefit; consumers can
still cache capability handles in ``__init__``. The query pattern can
evolve into injection if a future use case demands it.

Open Items
----------

- **Custom :class:`TransformFormat`.** Customers will eventually want to
  add new transform formats and conversion kernels. The current
  ``TransformFormat`` ``Enum`` is closed; opening it requires migrating
  to a class hierarchy with a kernel-registry dispatcher. This is
  architecturally compatible with the capability model
  (``GpuTransformBuffer.get_body_transforms`` already accepts a format
  parameter) but is deferred to a follow-up after this PR merges.

- **Cross-process capability identity.** Python ``type`` identity is
  stable in-process but not across pickling or RPC. If the SDP is ever
  used across processes, capabilities will need stable URN-style
  identifiers (``"acme.shm.SharedMemoryChannel.v1"``). Out of scope.

- **CudaStream capability.** Originally proposed alongside the four
  primary capabilities. Deferred — no consumer concretely needs it yet.
  When OVRTX or another stream-aware consumer surfaces a need, the
  protocol will be added without disturbing the framework.

References
----------

- :doc:`0001-scene-data-provider-redesign` — the typed format-negotiation
  API this ADR builds on.
- ``AGENTS.md`` — deprecation policy (waived here because the entire SDP
  stack is unreleased on the branch).
