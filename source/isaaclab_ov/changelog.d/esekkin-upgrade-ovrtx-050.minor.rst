Changed
^^^^^^^

* Updated the optional OV runtimes to ``ovrtx==0.5.0.377615``, ``ovphysx==0.6.3``, and
  ``ovstage==0.2.0.377349``, which must be installed together. OvPhysX 0.6.3 declares
  ``ovstage==0.2.0.377349``, and OVRTX 0.5 requires OVStage 0.2, so no mixed combination of
  these runtimes is supported. Reinstall with ``uv sync --inexact --extra ov`` (or
  ``--extra ovrtx`` / ``--extra ovphysx``). Pinning any one of the three to an older release
  now fails at runtime rather than degrading.
* Changed the OmniClient dependency to require ``omniverseclient==2.74.0``, matching the client
  library that the OVStage 0.2 and OVRTX 0.5 native plugins bundle. An environment left on
  ``omniverseclient==2.72.3`` fails plugin initialization with
  ``Initialization failed, minor version 74 > 72``.
* Writes to OvPhysX CPU-only property bindings, such as deformable material attributes, now
  require host-resident ``indices`` and ``mask`` buffers. OvPhysX 0.6 rejects device tensors on
  these bindings instead of staging them, so move selection buffers to the host before writing.

Deprecated
^^^^^^^^^^

* OvPhysX 0.6.3 deprecated the tensor-binding API that :class:`~isaaclab_ov.sim.views.OvPhysxView`
  is built on, in favor of unified read/write APIs that expose Warp arrays directly. The bindings
  still work in 0.6.3; no migration is required yet.

Known Issues
^^^^^^^^^^^^

* OVRTX 0.5 changes rendering defaults relative to 0.4, so captured images differ from images
  produced by the previous runtime. Rendering-correctness comparisons against goldens recorded
  under OVRTX 0.4 are expected to report differences.
