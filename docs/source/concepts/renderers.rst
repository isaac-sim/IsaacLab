.. _concepts_renderers:
.. _overview_renderers:

Renderers
=========

Renderers produce camera-sensor buffers for policy observations and synthetic-data workflows.
They are distinct from :doc:`visualizers <visualization>`, which provide
human-facing, interactive views for inspection, debugging, and recording. Isaac Lab uses a
pluggable renderer architecture. All implementations follow the interface defined by
:class:`~isaaclab.renderers.BaseRenderer`.

Isaac Lab supports three rendering backends:

- **Isaac RTX renderer** (``IsaacRtxRendererCfg``)

  - Runs NVIDIA's Omniverse RTX rendering pipeline inside Isaac Sim and pairs with PhysX.
  - Provides RTX Minimal and photo-real rendering, plus the broadest camera-output coverage.
  - Best for full RTX fidelity and workflows that already depend on Isaac Sim or Kit.

- **OVRTX renderer** (``OVRTXRendererCfg``)

  - Provides kit-less RTX rendering through the ``isaaclab_ov`` extension and pairs with Newton.
  - Provides RTX Minimal and photo-real rendering with geometry, motion, and label outputs.
  - Best for RTX image quality without running Isaac Sim.

- **Newton Warp renderer** (``NewtonWarpRendererCfg``)

  - Provides lightweight, kit-less rasterization built on NVIDIA Warp and pairs with Newton.
  - Produces RGB, albedo, depth, normals, and label outputs, but not motion vectors or RTX material
    transport.
  - Best for training workflows where throughput matters more than full RTX fidelity. Its focused
    raster pipeline exposes fewer ground-truth outputs than the RTX renderers, which integrate a
    broader set of RTX and annotator capabilities.

See :ref:`camera-supported-annotators` for the output-by-backend support matrix.

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

.. _renderer-visual-comparison:

Renderer outputs at a glance
----------------------------

The galleries below use the same authored scene, camera, lights, materials, and initial conditions.
Six spheres exercise mirror-like, transparent, semi-transparent, matte, glossy, and emissive
materials. The RGB output is animated to show the spheres falling onto the table; the remaining
outputs are still frames from the same run. Stills use the sixth rendered frame so temporal outputs,
such as motion vectors, show useful motion while the spheres remain near their initial poses.

Treat the images as a qualitative comparison of feature coverage and image character, not a
performance benchmark. The kit-less renderers use Newton physics while Isaac RTX uses PhysX, so
the exact sphere poses can differ. Display-only color maps make scalar, vector, and label outputs
readable here; camera sensors still return their documented raw tensors. Closely related aliases
and distance or ID variants are omitted because they do not add a visually distinct mode.

.. include:: _renderer_gallery.rst

Choosing a rendering capability
--------------------------------

The overview above shows the complete range of visually distinct outputs in one place. The sections
below regroup those outputs by purpose: simplified rendering for throughput-oriented training,
photo-real rendering for full RTX image quality, and advanced outputs for geometry, motion, and
labels. The detailed captions and commands use the suffixless ``Isaac-Cartpole-Camera`` task so
each shown mode can be tried without editing Python.

Simplified rendering
~~~~~~~~~~~~~~~~~~~~

Simplified rendering prioritizes throughput and predictable image formation over full light
transport. Newton Warp provides a lightweight rasterized RGB path. OVRTX and Isaac RTX provide RTX
Minimal mode, which disables indirect lighting and offers three levels of material evaluation:

- **Constant diffuse** uses one constant surface color.
- **Diffuse MDL** is Isaac Lab's stable name for textured diffuse shading.
- **Full MDL** is Isaac Lab's stable name for diffuse, glossy, and emissive material evaluation.

RTX Minimal uses the first distant light in the scene and hard shadows. See the upstream
`OVRTX Minimal mode <https://nvidia-omniverse.github.io/ovrtx/sensors/cameras/render_modes/minimal.html>`_
and `RTX Minimal renderer <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_minimal.html>`_
documentation for the renderer-level settings and limitations.

.. tab-set::

   .. tab-item:: Newton Warp RGB

      .. grid:: 1 2 2 2
         :gutter: 2

         .. grid-item::

            .. figure:: ../_static/overview/sensors/camera-renderer-newton-shadows-disabled.png
               :width: 100%
               :alt: Newton Warp RGB output with directional-light shadows disabled.

               Without shadows (default)

         .. grid-item::

            .. figure:: ../_static/overview/sensors/camera-renderer-newton-shadows-enabled.png
               :width: 100%
               :alt: Newton Warp RGB output with directional-light shadows enabled.

               With shadows

      Newton Warp is the kit-less choice when a lightweight RGB observation is sufficient.

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl \
            --task Isaac-Cartpole-Camera \
            physics=newton_mjwarp renderer=newton_renderer presets=rgb

      Enable directional-light shadows explicitly:

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl \
            --task Isaac-Cartpole-Camera \
            physics=newton_mjwarp renderer=newton_renderer presets=rgb \
            env.scene.tiled_camera.renderer_cfg.enable_shadows=true

   .. tab-item:: OVRTX Minimal

      .. grid:: 1 2 3 3
         :gutter: 2
         :class-container: renderer-preset-grid

         .. grid-item::

            .. figure:: ../_static/overview/sensors/camera-renderer-ovrtx-simple-shading-constant-diffuse.png
               :width: 100%
               :alt: OVRTX constant-diffuse RTX Minimal output.

               Constant diffuse — ``presets=simple_shading_constant_diffuse``

         .. grid-item::

            .. figure:: ../_static/overview/sensors/camera-renderer-ovrtx-simple-shading-diffuse-mdl.png
               :width: 100%
               :alt: OVRTX textured-diffuse RTX Minimal output.

               Diffuse MDL — ``presets=simple_shading_diffuse_mdl``

         .. grid-item::

            .. figure:: ../_static/overview/sensors/camera-renderer-ovrtx-simple-shading-full-mdl.png
               :width: 100%
               :alt: OVRTX full-material RTX Minimal output.

               Full MDL — ``presets=simple_shading_full_mdl``

      **Constant diffuse**

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=newton_mjwarp renderer=ovrtx presets=simple_shading_constant_diffuse

      **Textured diffuse**

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=newton_mjwarp renderer=ovrtx presets=simple_shading_diffuse_mdl

      **Diffuse, glossy, and emissive material evaluation**

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=newton_mjwarp renderer=ovrtx presets=simple_shading_full_mdl

   .. tab-item:: Isaac RTX Minimal

      .. grid:: 1 2 3 3
         :gutter: 2
         :class-container: renderer-preset-grid

         .. grid-item::

            .. figure:: ../_static/overview/sensors/camera-renderer-isaac-rtx-simple-shading-constant-diffuse.png
               :width: 100%
               :alt: Isaac RTX constant-diffuse RTX Minimal output.

               Constant diffuse — ``presets=simple_shading_constant_diffuse``

         .. grid-item::

            .. figure:: ../_static/overview/sensors/camera-renderer-isaac-rtx-simple-shading-diffuse-mdl.png
               :width: 100%
               :alt: Isaac RTX textured-diffuse RTX Minimal output.

               Diffuse MDL — ``presets=simple_shading_diffuse_mdl``

         .. grid-item::

            .. figure:: ../_static/overview/sensors/camera-renderer-isaac-rtx-simple-shading-full-mdl.png
               :width: 100%
               :alt: Isaac RTX full-material RTX Minimal output.

               Full MDL — ``presets=simple_shading_full_mdl``

      **Constant diffuse**

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=isaacsim_physx renderer=isaacsim_rtx presets=simple_shading_constant_diffuse

      **Textured diffuse**

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=isaacsim_physx renderer=isaacsim_rtx presets=simple_shading_diffuse_mdl

      **Diffuse, glossy, and emissive material evaluation**

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=isaacsim_physx renderer=isaacsim_rtx presets=simple_shading_full_mdl

Photo-real rendering
~~~~~~~~~~~~~~~~~~~~

Here, **photo-real rendering** means the regular RGB path from the full RTX Real-Time Path-Tracing
mode; it is a capability grouping, not an Isaac Lab preset name. Choose it when material appearance,
reflections, transparency, lighting, or the accompanying RTX AOVs matter more than the lowest
possible render latency. OVRTX provides this path without Kit, while Isaac RTX provides it inside
Isaac Sim. See the upstream `OVRTX render modes
<https://nvidia-omniverse.github.io/ovrtx/sensors/cameras/render_modes.html>`_ and
`Isaac Sim rendering modes
<https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/rendering_modes.html>`_.

.. tab-set::

   .. tab-item:: OVRTX

      .. figure:: ../_static/overview/sensors/camera-renderer-ovrtx.webp
         :align: center
         :width: 90%
         :alt: Six material spheres falling onto a table in OVRTX RGB output.

         OVRTX photo-real RGB — ``renderer=ovrtx presets=rgb``

      The same renderer also produces the albedo, depth, normals, segmentation, and motion-vector
      outputs shown in the overview gallery.

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=newton_mjwarp renderer=ovrtx presets=rgb

   .. tab-item:: Isaac RTX

      .. figure:: ../_static/overview/sensors/camera-renderer-isaac-rtx.webp
         :align: center
         :width: 90%
         :alt: Six material spheres falling onto a table in Isaac RTX RGB output.

         Isaac RTX photo-real RGB — ``renderer=isaacsim_rtx presets=rgb``

      The same renderer also produces the albedo, depth, normals, segmentation, and motion-vector
      outputs shown in the overview gallery.

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=isaacsim_physx renderer=isaacsim_rtx presets=rgb

Advanced rendering outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond RGB, renderer-produced buffers expose material, geometry, motion, and labeling information:

- **Albedo** isolates the material base color from lighting.
- **Depth** measures optical-axis distance for geometric perception and reconstruction.
- **Normals** encode local surface orientation.
- **Semantic segmentation** groups pixels by class, while **instance segmentation** separates
  individual objects.
- **Motion vectors** encode image-space motion per pixel and require prior-frame history.

Request one or more buffers through :attr:`~sensors.CameraCfg.data_types`. See
:ref:`camera-configuration` for a configuration example and :ref:`camera-output-types` for the
available names, tensor shapes, data types, and meanings. Environment-level ``presets=...``
selectors are task-defined convenience shortcuts, not an exhaustive list of camera outputs.

Output availability differs by backend. Check the :ref:`camera renderer support matrix
<camera-supported-annotators>` before choosing a renderer; for example, OVRTX and Isaac RTX produce
motion vectors, while Newton Warp does not. The :ref:`renderer visual comparison
<renderer-visual-comparison>` above shows these outputs on the same scene.

.. note::

   Visualization markers are debug overlays provided by visualizers, not camera outputs. Their
   support is independent of the camera renderer: the Kit, Newton GL, Rerun, and Viser visualizers
   support markers, while the experimental Newton RTX visualizer does not. See
   :doc:`/source/concepts/visualization` for the visualizer support matrix.

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

.. warning::

   Kit RTX sizes each partition from the bounding boxes of the prims it contains and never
   refreshes the bounding box of an animated ``UsdGeom.BasisCurves`` prim, so cables can be
   culled once they deform beyond their initial extent. See
   :ref:`known-issues-animated-curve-scene-partition` for the workaround.

Architecture Overview
---------------------

The renderer system consists of:

1. **BaseRenderer** — Abstract base class defining the rendering lifecycle and interface
2. **RendererCfg** — Base configuration; each backend extends it with backend-specific options and declares
   its implementation in ``class_type``
3. **Concrete implementations** — Backend-specific renderers in extension packages
4. **RenderContext** — A management class for instantiating and accessing renderer instances using a **RendererCfg**.
   After instantiation, a config can then be used to acquire the instance of the renderer as needed.

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.renderers import BaseRenderer
   from isaaclab_newton.renderers import NewtonWarpRendererCfg

   # Create a Newton Warp renderer (no Isaac Sim required)
   sim_ctx = sim_utils.SimulationContext.instance()
   # RenderContext.get_renderer constructs cfg.class_type(cfg)
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
   # RenderContext.get_renderer constructs cfg.class_type(cfg)
   # or return an existing renderer with a matching config
   renderer: BaseRenderer = sim_ctx.render_context.get_renderer(IsaacRtxRendererCfg())

For RTX renderer settings, see
:doc:`/source/how-to/configure_rendering`.

Core concepts
-------------

- **Use the RenderContext**: Always acquire renderers via the RenderContext with a renderer-specific config class
  (e.g. ``sim_ctx.render_context.get_renderer(IsaacRtxRendererCfg())``). Do not import or instantiate concrete backend classes
  (e.g. ``IsaacRtxRenderer``, ``OVRTXRenderer``) directly—their names and package locations are
  implementation details and may change without notice.

- **Lightweight config imports**: Importing a renderer configuration class does not pull in backend-specific
  dependencies. ``class_type`` is resolved lazily when the renderer is constructed, and construction may fail
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

- :doc:`/source/concepts/scene_data_providers`: how scene data flows from physics backends to renderers
- :doc:`/source/concepts/visualization` — lightweight visualizer backends for interactive feedback
