isaaclab.renderers
==================

.. automodule:: isaaclab.renderers

  .. rubric:: Classes

  .. autosummary::

    BaseRenderer
    RendererCfg

  .. rubric:: Functions

  .. autosummary::

    resolve_async_rendering_enabled
    warn_unsupported_async_rendering

Base Renderer
-------------

.. autoclass:: BaseRenderer
  :members:
  :show-inheritance:
  :exclude-members: __init__

Renderer Configuration
-----------------------

.. autoclass:: RendererCfg
  :members:
  :show-inheritance:
  :exclude-members: __init__


Additional Public Classes
-------------------------

The following classes are part of the public :mod:`isaaclab.renderers` API.

.. currentmodule:: isaaclab.renderers

.. autosummary::
   :nosignatures:

   CameraRenderSpec
   RenderBufferKind
   RenderBufferSpec
   RenderContext

.. autoclass:: CameraRenderSpec
   :show-inheritance:

.. autoclass:: RenderBufferKind
   :show-inheritance:

.. autoclass:: RenderBufferSpec
   :show-inheritance:

.. autoclass:: RenderContext
   :show-inheritance:

Asynchronous Rendering
----------------------

:attr:`RendererCfg.async_rendering` trades one frame of camera latency for pipelined rendering, and
:data:`ASYNC_RENDERING_ENV_VAR` overrides it process-wide. Renderers resolve it through the helpers
below. Only the OVRTX renderer implements the pipelined path.

.. py:data:: ASYNC_RENDERING_ENV_VAR
   :value: "ISAAC_LAB_ASYNC_RENDERING"

   Environment variable overriding :attr:`RendererCfg.async_rendering` for every renderer.

   Accepts boolean spellings: ``0``/``false``/``no``/``off`` or ``1``/``true``/``yes``/``on``. Any
   other value raises ``ValueError``. Set it to exercise the asynchronous path without naming a
   camera that a given task may not define.

.. autofunction:: resolve_async_rendering_enabled

.. autofunction:: warn_unsupported_async_rendering
