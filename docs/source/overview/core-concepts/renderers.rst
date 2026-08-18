.. _overview_renderers:

Renderers
=========

Isaac Lab uses a pluggable renderer architecture to support different rendering backends for camera sensors.
The :class:`~isaaclab.renderers.BaseRenderer` abstract base class defines the interface that all renderer
implementations must follow.

Isaac Lab supports three rendering backends:

- **Isaac RTX renderer** (``IsaacRtxRendererCfg``) — NVIDIA's Omniverse RTX rendering pipeline
  running inside Isaac Sim. Requires Isaac Sim. Best for photorealistic rendering, full camera
  sensor support (RGB, depth, semantic segmentation, etc.), and production quality outputs.
- **OVRTX renderer** (``OVRTXRendererCfg``) — A standalone RTX path-tracing renderer provided by
  the ``isaaclab_ov`` extension. Delivers RTX-quality rendering.
- **Newton Warp renderer** (``NewtonWarpRendererCfg``) — A lightweight GPU-accelerated renderer
  built on NVIDIA Warp. Works with the Newton physics backend and does **not** require Isaac Sim
  (kit-less mode). Ideal for training workflows where full RTX fidelity is not needed.

Choosing a renderer backend
----------------------------

+---------------------+-------------------------------+---------------------------------+
| Backend             | Requires Isaac Sim?           | Best For                        |
+=====================+===============================+=================================+
| Isaac RTX           | Yes                           | Full sensor fidelity, RTX       |
|                     |                               | photorealism, PhysX backend     |
+---------------------+-------------------------------+---------------------------------+
| OVRTX               | No (kit-less; needs           | RTX-quality rendering without   |
|                     | ``isaaclab_ov`` + ``ovrtx``)  | requiring Isaac Sim             |
+---------------------+-------------------------------+---------------------------------+
| Newton Warp         | No (kit-less)                 | Newton backend, fast training   |
+---------------------+-------------------------------+---------------------------------+

.. note::

   Visualization markers are not yet supported by Newton-based renderer backends,
   including the Newton Warp renderer. Use an RTX-based renderer, such as the
   Isaac RTX renderer or OVRTX renderer, when marker visualization is needed.

.. note::
   **Temporal information for camera-based RL.** Unlike RTX modes with temporal
   anti-aliasing (DLSS, DLAA, TAA), the Newton Warp renderer does not inject
   prior-frame information into the current image. Camera-control tasks that depend
   on velocity-like visual cues should add explicit temporal observations
   (e.g. task-local frame stacking) rather than relying on renderer-specific artifacts.

Per-environment Isaac RTX scene partitioning
---------------------------------------------

The Isaac RTX renderer enables per-environment scene partitioning by default. It assigns
matching scene-partition tokens to each ``/World/envs/env_<index>`` hierarchy and its
camera so tiled views render only that environment's geometry.

Configure the behavior through :class:`~isaaclab_physx.renderers.IsaacRtxRendererCfg`:

.. code-block:: python

   from isaaclab_physx.renderers import IsaacRtxRendererCfg

   renderer_cfg = IsaacRtxRendererCfg(enable_scene_partitioning=False)

Scene partitioning and the all-environment spectator view are separate controls.
:class:`~isaaclab.app.AppLauncher` enables spectator support before RTX startup only
when the Kit viewport is enabled or Kit visualization, recording, livestreaming, or XR
is requested. Regular headless training and camera-sensor runs keep it disabled so
tiled cameras are not exposed to the spectator mode's world-space layout constraints.

``global_settings.show_all_partitions_by_default`` maps to that same process-global RTX
setting; it is not a separate feature. Its default value of ``None`` preserves the
launch-time choice made by :class:`~isaaclab.app.AppLauncher`. An explicit value overrides
that setting when the Isaac RTX renderer is constructed. When enabled, environments must
remain spatially separated because overlapping partition bounds can make content leak into
another environment or disappear. When disabled, the Kit viewport displays only the
selected environment.

This setting does not affect OVRTX, which always partitions multi-environment scenes.

Prims outside the environment hierarchies remain in the shared background partition.
Environment-owned ``PointInstancer`` markers can carry one matching scene-partition
token per instance; markers without that ownership information remain shared.

Architecture Overview
---------------------

The renderer system consists of:

1. **BaseRenderer** — Abstract base class defining the rendering lifecycle and interface
2. **Renderer** — Factory that instantiates the appropriate backend based on renderer configuration class
3. **RendererCfg** — Base configuration; each backend extends it with backend-specific options
4. **Concrete implementations** — Backend-specific renderers in extension packages
5. **RenderContext** — A management class for instantiating and accessing renderer instances using a **RendererCfg**.
   After instantiation, a config can then be used to acquire the instance of the renderer as needed.

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.renderers import BaseRenderer
   from isaaclab_newton.renderers import NewtonWarpRendererCfg

   # Create a Newton Warp renderer (no Isaac Sim required)
   sim_ctx = sim_utils.SimulationContext.instance()
   # RenderContext.get_renderer will instantiate the renderer backend
   # or return an existing renderer with a matching config
   renderer: BaseRenderer = sim_ctx.render_context.get_renderer(NewtonWarpRendererCfg())
   assert isinstance(renderer, BaseRenderer)

For the RTX renderer (requires Isaac Sim):

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.renderers import BaseRenderer
   from isaaclab_physx.renderers import IsaacRtxRendererCfg

   # Create an RTX renderer
   sim_ctx = sim_utils.SimulationContext.instance()
   # RenderContext.get_renderer will instantiate the renderer backend
   # or return an existing renderer with a matching config
   renderer: BaseRenderer = sim_ctx.render_context.get_renderer(IsaacRtxRendererCfg())

For RTX renderer settings, see
:doc:`/source/how-to/configure_rendering`.

Core concepts
-------------

- **Use the RenderContext**: Always instantiate renderers via the RenderContext with a renderer-specific config class
  (e.g. ``sim_ctx.render_context.get_renderer(IsaacRtxRendererCfg())``). Do not import or instantiate concrete backend classes
  (e.g. ``IsaacRtxRenderer``, ``OVRTXRenderer``) directly—their names and package locations are
  implementation details and may change without notice.

- **Lightweight config imports**: Importing a renderer configuration class does not pull in backend-specific
  dependencies. The backend is lazily loaded when the renderer is instantiated, and instantiation may fail
  if the backend is not installed.

  .. code-block:: python

     import isaaclab.sim as sim_utils
     from isaaclab.renderers import BaseRenderer
     # Lightweight: does not import OVRTX backend dependencies
     from isaaclab_ov.renderers import OVRTXRendererCfg

     # Lazily loads ovrtx when instantiated; may fail if isaaclab_ov / ovrtx is not installed
     sim_ctx = sim_utils.SimulationContext.instance()
     renderer: BaseRenderer = sim_ctx.render_context.get_renderer(OVRTXRendererCfg())

Installing the OVRTX renderer
------------------------------

The OVRTX renderer is provided by the ``isaaclab_ov`` extension. The extension's
source package ships with the core install, but the renderer's ``ovrtx`` runtime
wheel (the `ovrtx <https://github.com/NVIDIA-Omniverse/ovrtx>`_ package, published
on public PyPI) is **not** installed by default. You must request it
explicitly — OVRTX does **not** require Isaac Sim.

Install via the Isaac Lab CLI using the ``ov[ovrtx]`` token:

.. code-block:: bash

   # Install the ovrtx runtime wheel on top of an existing install
   ./isaaclab.sh -i ov[ovrtx]

.. note::

   The bare ``ov`` token does **not** install any runtime wheel (the source
   packages are already part of the core install). Use ``ov[ovrtx]`` (or ``ov[all]``)
   to pull in the ``ovrtx`` dependency.

Or install the public ``ovrtx`` package directly from PyPI:

.. isaaclab-ovrtx-install::

- **Opaque render data**: The render data object returned by :meth:`~isaaclab.renderers.BaseRenderer.create_render_data` is passed to
  subsequent renderer methods. It should be completely opaque to the caller: inspecting or modifying it
  via get/set attributes is an anti-pattern and breaks the API contract.

.. note::

   The :class:`~isaaclab.renderers.BaseRenderer` class is under active development and may change without notice.

See Also
--------

- :doc:`scene_data_providers` — how scene data flows from physics backends to renderers
- :doc:`/source/overview/core-concepts/visualization` — lightweight visualizer backends for interactive feedback
