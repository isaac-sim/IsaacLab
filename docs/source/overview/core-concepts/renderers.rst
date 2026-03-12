.. _overview_renderers:

Renderers
=========

Isaac Lab uses a pluggable renderer architecture to support different rendering backends for camera sensors.
The :class:`~isaaclab.renderers.BaseRenderer` abstract base class defines the interface that all renderer
implementations must follow.

Isaac Lab supports two rendering backends:

- **RTX renderer** (``IsaacRtxRendererCfg`` / ``OVRTXRendererCfg``) — NVIDIA's Omniverse RTX
  rendering pipeline. Requires Isaac Sim. Best for photorealistic rendering, full camera sensor
  support (RGB, depth, semantic segmentation, etc.), and production quality outputs.
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
| OVRTX               | Yes (+ ``isaaclab_ov``)       | Alternative RTX pipeline        |
+---------------------+-------------------------------+---------------------------------+
| Newton Warp         | No (kit-less)                 | Newton backend, fast training   |
+---------------------+-------------------------------+---------------------------------+

Architecture Overview
---------------------

The renderer system consists of:

1. **BaseRenderer** — Abstract base class defining the rendering lifecycle and interface
2. **Renderer** — Factory that instantiates the appropriate backend based on renderer configuration class
3. **RendererCfg** — Base configuration; each backend extends it with backend-specific options
4. **Concrete implementations** — Backend-specific renderers in extension packages

.. code-block:: python

   from isaaclab.renderers import BaseRenderer, Renderer
   from isaaclab_newton.renderers import NewtonWarpRendererCfg

   # Create a Newton Warp renderer (no Isaac Sim required)
   renderer: BaseRenderer = Renderer(NewtonWarpRendererCfg())
   assert isinstance(renderer, BaseRenderer)

For the RTX renderer (requires Isaac Sim):

.. code-block:: python

   from isaaclab.renderers import Renderer
   from isaaclab.renderers import IsaacRtxRendererCfg  # or OVRTXRendererCfg

   # Create an RTX renderer
   renderer: BaseRenderer = Renderer(IsaacRtxRendererCfg())

For RTX renderer settings and presets (quality, balanced, performance), see
:doc:`/source/how-to/configure_rendering`.

Core concepts
-------------

- **Use the factory**: Always instantiate renderers via the factory with a renderer-specific config class
  (e.g. ``Renderer(IsaacRtxRendererCfg())``). Do not import or instantiate concrete backend classes
  (e.g. ``IsaacRtxRenderer``, ``OVRTXRenderer``) directly—their names and package locations are
  implementation details and may change without notice.

- **Lightweight config imports**: Importing a renderer configuration class does not pull in backend-specific
  dependencies. The backend is lazily loaded when the renderer is instantiated, and instantiation may fail
  if the backend is not installed.

  .. code-block:: python

     # Lightweight: does not import OVRTX backend dependencies
     from isaaclab_ov.renderers import OVRTXRendererCfg

     # Lazily loads ovrtx when instantiated; may fail if isaaclab_ov / ovrtx is not installed
     renderer: BaseRenderer = Renderer(OVRTXRendererCfg())

Installing the OVRTX renderer
------------------------------

The OVRTX renderer is provided by the ``isaaclab_ov`` extension and requires the
`ovrtx <https://github.com/NVIDIA-Omniverse/ovrtx>`_ package (hosted on
``pypi.nvidia.com``).

Install via the Isaac Lab CLI:

.. code-block:: bash

   # Install isaaclab_ov (and its ovrtx dependency) alongside the core package
   ./isaaclab.sh -i ov

Or install manually with pip:

.. code-block:: bash

   pip install --extra-index-url https://pypi.nvidia.com -e source/isaaclab_ov

- **Opaque render data**: The render data object returned by :meth:`~isaaclab.renderers.BaseRenderer.create_render_data` is passed to
  subsequent renderer methods. It should be completely opaque to the caller: inspecting or modifying it
  via get/set attributes is an anti-pattern and breaks the API contract.

.. note::

   The :class:`~isaaclab.renderers.BaseRenderer` class is under active development and may change without notice.

See Also
--------

- :doc:`multi_backend_architecture` — the factory pattern used by the renderer system
- :doc:`scene_data_providers` — how scene data flows from physics backends to renderers
- :doc:`/source/features/visualization` — lightweight visualizer backends for interactive feedback
