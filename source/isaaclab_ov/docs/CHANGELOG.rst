Changelog
---------

0.1.7 (2026-05-12)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Construct the underlying OVRTX ``Renderer`` in
  :class:`~isaaclab_ov.renderers.OVRTXRenderer` ``__init__`` instead of
  during :meth:`~isaaclab_ov.renderers.OVRTXRenderer.prepare_stage`. This
  pairs with the new pre-physics ``__init__`` /
  post-physics :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.initialize`
  lifecycle: when invoked eagerly via
  :meth:`~isaaclab.scene.InteractiveScene.initialize_renderers`, the OVRTX
  ``Renderer`` is created before
  :meth:`~isaaclab.sim.SimulationContext.reset` (and therefore before
  ovphysx initialises), which OVRTX 0.3 requires.
* Replaced an ``assert`` on the OVRTX ``Renderer`` construction with an
  explicit :class:`RuntimeError` so the failure is reported even when
  Python is run with ``-O``.
* Renamed the internal ``OVRTXRenderer.initialize(spec)`` helper to
  ``_initialize_from_spec(spec)`` to avoid shadowing the new
  no-arg :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.initialize`
  lifecycle hook.


0.1.6 (2026-05-09)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Set ``keep_system_alive=True`` on the internal OVRTX ``RendererConfig`` in
  :class:`~isaaclab_ov.renderers.ovrtx_renderer.OVRTXRenderer` so the renderer
  system is not torn down prematurely during pytest sessions.
* Initialize Warp runtime for OvRTX renderer.


0.1.5 (2026-05-08)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Bumped Newton pin to ``v1.2.0rc2``. Pulls in IsaacLab-relevant fixes from
  `newton-physics/newton#2678 <https://github.com/newton-physics/newton/pull/2678>`_
  and `newton-physics/newton#2720
  <https://github.com/newton-physics/newton/pull/2720>`_ (``SolverKamino``
  reset under ``world_mask``), the upstream tendon-scoping fix from
  `newton-physics/newton#2659
  <https://github.com/newton-physics/newton/pull/2659>`_ ("Scope USD
  custom-frequency parsing"), and a VRAM-leak fix on example reset
  (`newton-physics/newton#2710
  <https://github.com/newton-physics/newton/pull/2710>`_).
* Newton ``v1.2.0rc2`` requires ``warp-lang==1.13.0``, ``mujoco==3.8.0``,
  and ``mujoco-warp==3.8.0.1``. ``warp-lang``/``mujoco``/``mujoco-warp``
  pins live in :mod:`isaaclab` and ``tools/wheel_builder/res/python_packages.toml``;
  the Newton pin is mirrored across :mod:`isaaclab_newton`,
  :mod:`isaaclab_visualizers` (3×), :mod:`isaaclab_physx` (``[newton]``
  extra), and the wheel-builder TOML.
* Updated ``wp.math.transform_to_matrix`` to ``wp.transform_to_matrix`` in
  :mod:`~isaaclab_newton.physics.newton_manager` and
  :mod:`~isaaclab_ov.renderers.ovrtx_renderer_kernels` to match the
  ``warp-lang`` 1.13 API (the ``wp.math`` namespace was removed).


0.1.4 (2026-05-08)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Modified the OVRTX renderer to use the new patterns from renderer/camera decoupling.

Fixed
^^^^^

* Fixed ``AttributeError: 'Renderer' object has no attribute 'add_usd'`` in
  :class:`~isaaclab_ov.renderers.OVRTXRenderer` when using ``ovrtx`` 0.3.0 or
  newer. The renderer now calls :meth:`ovrtx.Renderer.open_usd` on 0.3.0+ and
  falls back to ``Renderer.add_usd`` on older versions.


0.1.3 (2026-04-30)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Simple-shading outputs, with RTX Minimal mode resolved from the requested camera data types and written on
  the injected render product in USD.
* Expanded unit tests for OVRTX Warp kernels in ``test_ovrtx_renderer_kernels.py``.

Changed
^^^^^^^

* OVRTX integration now branches ``read_gpu_transforms``, depth tile extraction, and semantic ID coloring kernels on
  ovrtx **0.3.0** vs older versions so tiled buffers and transforms stay correct across ovrtx versions.
* RGB tiling reads ``LdrColor`` and supports both 3- and 4-channel buffers.

Removed
^^^^^^^

* Removed ``OVRTXRendererCfg.simple_shading_mode``. Request simple shading via the simple-shading data types on the
  camera instead; the renderer derives RTX minimal mode from the data types.

0.1.2 (2026-03-23)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Semantic segmentation in :class:`~isaaclab_ov.renderers.OVRTXRenderer` maps
  semantic instance IDs to RGBA using the same pseudo-random per-ID HSV scheme as the
  Isaac Sim RTX render backend, so OVRTX and Isaac RTX produce matching colors for the
  same IDs. Numeric IDs ``0`` (BACKGROUND) and ``1`` (UNLABELLED) use fixed RGBA.

0.1.1 (2026-03-07)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added ``ovrtx>=0.2.0,<0.3.0`` as a declared dependency, installable from the
  public NVIDIA package index (``pypi.nvidia.com``).
* Added ``ov`` to the list of valid sub-packages for selective installation via
  ``./isaaclab.sh -i ov``.

0.1.0 (2026-03-04)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :mod:`isaaclab_ov.renderers` module with OVRTX renderer for tiled camera
  rendering:

  * :class:`~isaaclab_ov.renderers.OVRTXRenderer` and
    :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`: RTX-based rendering via the
    ovrtx library, with stage export, USD cloning, and camera/object bindings.

  * :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.prepare_stage`: Base
    interface hook for stage preprocessing before create_render_data (OVRTX
    exports USD stage; Isaac RTX and Newton Warp use no-op implementations).
