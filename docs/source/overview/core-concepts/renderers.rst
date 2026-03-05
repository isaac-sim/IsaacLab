.. _overview_renderers:

Renderers
=========

Isaac Lab uses a pluggable renderer architecture to support different rendering backends for camera sensors.
The :class:`~isaaclab.renderers.BaseRenderer` abstract base class defines the interface that all renderer
implementations must follow.

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

   # Create a renderer via the factory (returns the appropriate backend instance)
   renderer: BaseRenderer = Renderer(NewtonWarpRendererCfg())
   assert isinstance(renderer, BaseRenderer)

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

- **Opaque render data**: The render data object returned by :meth:`create_render_data` is passed to
  subsequent renderer methods. It should be completely opaque to the caller: inspecting or modifying it
  via get/set attributes is an anti-pattern and breaks the API contract.

.. note::

   The :class:`~isaaclab.renderers.BaseRenderer` class is under active development and may change without notice.
