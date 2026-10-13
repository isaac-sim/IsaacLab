.. _concepts_renderers:
.. _overview_renderers:

Renderers
=========

Renderers produce camera-sensor buffers for policy observations and synthetic-data workflows.
They are distinct from :doc:`visualizers <visualization>`, which provide human-facing, interactive
views for inspection, debugging, and recording. Every renderer implements
:class:`~isaaclab.renderers.BaseRenderer`, and each camera selects one through
:attr:`~isaaclab.sensors.CameraCfg.renderer_cfg`.

.. important::

   OVRTX is the recommended RTX renderer. Isaac RTX is the legacy RTX renderer: it remains available
   for workflows that must run inside Isaac Sim, but new RTX camera work should use OVRTX.

.. _renderer-selection:

Choosing a renderer
-------------------

Each animation shows the same scene and camera in the renderer's ``rgb`` output.

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Newton Warp :bdg-info:`Lightweight`

      .. image:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/camera-renderer-newton.webp
         :alt: Six material spheres falling onto a table in Newton Warp RGB output.
         :width: 100%

      Warp ray tracing with direct lighting. Use it when camera throughput and memory matter more than
      material and lighting fidelity.

      ``renderer=newton_renderer``

   .. grid-item-card:: OVRTX :bdg-success:`Recommended`

      .. image:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/camera-renderer-ovrtx.webp
         :alt: Six material spheres falling onto a table in OVRTX RGB output.
         :width: 100%

      Kit-less RTX rendering with path-traced and Minimal modes. Use it for RTX image quality and
      RTX outputs without Isaac Sim.

      ``renderer=ovrtx``

   .. grid-item-card:: Isaac RTX :bdg-warning:`Legacy`

      .. image:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/camera-renderer-isaac-rtx.webp
         :alt: Six material spheres falling onto a table in Isaac RTX RGB output.
         :width: 100%

      RTX rendering inside Isaac Sim. Use it only when the workflow must run in an Isaac Sim
      process.

      ``renderer=isaacsim_rtx``

.. _renderer-details:

Technical specifications
------------------------

The :ref:`camera renderer support matrix <camera-supported-annotators>` is the authoritative
output-by-renderer reference.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :width: 100%
   :widths: 16 28 28 28
   :class: renderer-table

   * -
     - Newton Warp
     - OVRTX
     - Isaac RTX (legacy)
   * - Status
     - Lightweight renderer
     - Recommended RTX renderer
     - Legacy; maintained for workflows that run inside Isaac Sim
   * - Image formation
     - Warp ray tracing with direct lighting, ambient light, and optional directional-light shadows.
       Shades diffuse color and textures; no indirect lighting, reflections, or refraction.
     - RTX :ref:`Real-Time Path Tracing (RTPT) <renderer-render-modes>` by default; RTX Minimal for
       ``simple_shading_*`` outputs. Full MDL materials.
     - The same RTX modes, rendered through Kit and Replicator. RTX Minimal requires Isaac Sim 6.0
       or later.
   * - Runtime
     - Kit-less; included in the base install
     - Kit-less; ``ovrtx`` or ``ov`` extra (``ovrtx`` |ovrtx_docs_ref|)
     - Isaac Sim and Kit; ``isaacsim`` extra
   * - Physics backends
     - Newton, OvPhysX, Isaac Sim PhysX
     - Newton, OvPhysX
     - Isaac Sim PhysX, Newton
   * - ``renderer=`` selector
     - ``newton_renderer``
     - ``ovrtx``
     - ``isaacsim_rtx``
   * - Configuration
     - :class:`~isaaclab_newton.renderers.NewtonWarpRendererCfg`
     - :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`
     - :class:`~isaaclab_physx.renderers.IsaacRtxRendererCfg`
   * - Outputs
     - ``rgb``, ``rgba``, ``rgb_hdr``, ``albedo``, depth and distances, ``normals``, semantic and
       instance segmentation
     - Newton Warp outputs, plus ``motion_vectors`` and ``simple_shading_*``
     - OVRTX outputs, plus ``instance_id_segmentation_fast``. ``albedo`` and ``simple_shading_*``
       require Isaac Sim 6.0 or later.
   * - Prior-frame history
     - None; each image is independent
     - RTPT denoising reuses prior frames
     - RTPT anti-aliasing and denoising reuse prior frames; DLSS by default
   * - Environment isolation
     - Each environment renders its own Newton world plus shared global geometry
     - Scene partitioning, always on; up to 15625 partitions
     - Scene partitioning, on by default; up to 15625 partitions
   * - Asynchronous rendering
     - No
     - Opt-in through :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.async_rendering`
     - No

``renderer=rtx`` selects OVRTX for kit-less runs and Isaac RTX when the run needs Kit, for example
for Isaac Sim PhysX, the Kit visualizer, or livestreaming. OVRTX cannot share a process with Kit.
Tasks that need velocity-like visual cues should add explicit temporal observations, such as
task-local frame stacking, rather than rely on renderer-specific prior-frame history.

Select a renderer at launch together with a compatible physics backend:

.. tab-set::

   .. tab-item:: Newton Warp

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=newton_mjwarp renderer=newton_renderer presets=rgb

   .. tab-item:: OVRTX

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=newton_mjwarp renderer=ovrtx presets=rgb

   .. tab-item:: Isaac RTX (legacy)

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=isaacsim_physx renderer=isaacsim_rtx presets=rgb

.. _renderer-performance:

Performance
-----------

Values are relative to Newton Warp. Higher is better for throughput; lower is better for memory.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :width: 100%
   :widths: 28 24 24 24

   * - Metric
     - Newton Warp
     - OVRTX
     - Isaac RTX (legacy)
   * - Environment-step throughput
     - 100%
     - 50%
     - 20%
   * - GPU memory (VRAM)
     - 100%
     - 427%
     - 543%
   * - Host memory (RSS)
     - 100%
     - 280%
     - 396%

On the same physics backend, OVRTX delivers about 2.5 times the throughput of Isaac RTX with about
20% less GPU memory and 30% less host memory.

The measurement uses ``Isaac-Reorient-KukaAllegro-Camera`` with 8192 environments,
``physics=newton_mjwarp``, and ``presets=cube,single_camera,rgb64``: one 64 × 64 ``rgb`` camera per
environment, which both RTX renderers render in :ref:`RTPT <renderer-render-modes>`. Each renderer
was measured once on an NVIDIA L40 GPU at Isaac Lab commit ``32fea6be``. Throughput is
``runtime.environment_step_timing.environment_step_fps``, which times ``env.step()`` only. Memory is
the run mean of GPU memory used and of host process RSS. Ratios change with the task, resolution,
outputs, and GPU; see :ref:`developer_tools_benchmarking_run` to measure a specific configuration.

.. _renderer-render-modes:

RTX render modes
----------------

OVRTX and Isaac RTX set the render mode for each camera from its requested outputs. A
``simple_shading_*`` output selects RTX Minimal at the shading level that the output names; other
cameras use Real-Time Path Tracing (RTPT), which is also the default Kit render mode for Isaac RTX.
The mode applies per camera render product, so cameras in one scene can use different modes. Isaac
Lab does not select the progressive Path Tracing mode that OVRTX also documents.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :width: 100%
   :widths: 18 41 41

   * -
     - Real-Time Path Tracing (RTPT)
     - RTX Minimal
   * - Selected by
     - Default for every output except ``simple_shading_*``
     - One ``simple_shading_*`` output
   * - Light transport
     - Path tracing with real-time denoising: global illumination, reflections, and shadows from all
       scene lights
     - Direct lighting from the first ``DistantLight`` only; no indirect lighting, and dome lights
       are ignored. Ambient light is optional.
   * - Materials
     - Full MDL evaluation
     - Set by the output: ``simple_shading_constant_diffuse`` uses one constant color,
       ``simple_shading_diffuse_mdl`` uses textured diffuse color, and ``simple_shading_full_mdl``
       evaluates diffuse, glossy, and emissive terms
   * - Shadows
     - Always traced
     - Hard shadows from the first ``DistantLight``; OVRTX disables them by default
       (:attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.enable_shadows`)
   * - Use for
     - Material appearance, reflections, transparency, and lighting
     - Throughput when simplified color is sufficient
   * - OVRTX reference
     - :ovrtx-docs:`Real-Time Path-Tracing <sensors/cameras/render_modes/real_time_path_tracing.rst>`
     - :ovrtx-docs:`Minimal <sensors/cameras/render_modes/minimal.rst>`
   * - Isaac RTX reference
     - `RTX Real-Time 2.0 <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt.html>`__
     - `RTX Minimal <https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_minimal.html>`__

The OVRTX links open the documentation for ``ovrtx`` |ovrtx_docs_ref|, the version Isaac Lab
installs. The renderers differ when one camera requests RTX Minimal together with regular color:

- OVRTX rejects ``rgb`` or ``rgba`` together with a ``simple_shading_*`` output on one camera, and
  more than one ``simple_shading_*`` output per camera. Request them from separate cameras.
- Isaac RTX keeps such a camera in RTPT. Its ``simple_shading_*`` output is still produced, without
  the RTX Minimal speedup.

For example, select full-MDL RTX Minimal shading on ``Isaac-Cartpole-Camera``:

.. tab-set::

   .. tab-item:: OVRTX

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=newton_mjwarp renderer=ovrtx presets=simple_shading_full_mdl

   .. tab-item:: Isaac RTX (legacy)

      .. code-block:: bash

         uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Camera \
            physics=isaacsim_physx renderer=isaacsim_rtx presets=simple_shading_full_mdl

Use ``presets=simple_shading_constant_diffuse`` or ``presets=simple_shading_diffuse_mdl`` for the
other shading levels, and ``presets=rgb`` for RTPT. The :ref:`output comparison
<renderer-visual-comparison>` shows each shading level next to RTPT ``rgb``.

.. _renderer-visual-comparison:

Output comparison
-----------------

Every image uses the same scene, camera, lights, and materials: six spheres with mirror-like,
transparent, semi-transparent, matte, glossy, and emissive materials. ``rgb`` is animated; the other
outputs are stills from the sixth frame, so motion vectors show motion. Newton Warp and OVRTX use
Newton physics while Isaac RTX uses PhysX, so sphere poses can differ slightly. Scalar, vector, and
label outputs are color-mapped for display; cameras return the raw tensors described in
:ref:`camera-output-types`.

.. include:: _renderer_gallery.rst

Minimal rows are the ``simple_shading_*`` outputs. Depth uses the turbo color map from near (purple)
to far (red), normals map X, Y, and Z to red, green, and blue, and segmentation assigns one color per
class or object. Motion-vector color encodes direction and relative magnitude; arrows show sampled
image-space motion.

Request outputs through :attr:`~isaaclab.sensors.CameraCfg.data_types`; see
:ref:`camera-configuration` for an example. Requesting an output that the renderer does not support
fails during camera initialization. Environment-level ``presets=...`` selectors are task-defined
shortcuts, not an exhaustive list of camera outputs.

.. note::

   Visualization markers are debug overlays drawn by visualizers, not camera outputs, so their
   support does not depend on the camera renderer. See :doc:`/source/concepts/visualization` for the
   visualizer support matrix.

Installing the OVRTX renderer
------------------------------

The OVRTX renderer is provided by the ``isaaclab_ov`` extension. The extension's
source package ships with the core install, but the renderer's ``ovrtx`` runtime
wheel (the `ovrtx <https://github.com/NVIDIA-Omniverse/ovrtx>`_ package, published
on public PyPI) is **not** installed by default. You must request it
explicitly — OVRTX does **not** require Isaac Sim.

Add the ``ovrtx`` extra to the uv environment:

.. code-block:: bash

   # Install the ovrtx runtime wheel on top of an existing install
   uv sync --inexact --extra ovrtx

.. note::

   The ``ov`` extra installs both the OV PhysX and OVRTX runtimes. Use the
   ``ovrtx`` extra to install only the OVRTX runtime. ``--inexact`` preserves
   packages already installed through other extras.

Or install the public ``ovrtx`` package directly from PyPI:

.. isaaclab-ovrtx-install::

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
:class:`~isaaclab_physx.app.KitLauncher` enables spectator support before RTX startup only
when the Kit viewport is enabled or Kit visualization, recording, livestreaming, or XR
is requested. Regular headless training and camera-sensor runs keep it disabled so
tiled cameras are not exposed to the spectator mode's world-space layout constraints.

``global_settings.show_all_partitions_by_default`` maps to that same process-global RTX
setting; it is not a separate feature. Its default value of ``None`` preserves the
launch-time choice made by :class:`~isaaclab_physx.app.KitLauncher`. An explicit value overrides
that setting when the Isaac RTX renderer is constructed. When enabled, environments must
remain spatially separated because overlapping partition bounds can make content leak into
another environment or disappear. When disabled, the Kit viewport displays only the
selected environment.

This setting does not affect OVRTX, which always partitions multi-environment scenes.

Prims outside the environment hierarchies remain in the shared background partition.
Environment-owned ``PointInstancer`` markers can carry one matching scene-partition
token per instance; markers without that ownership information remain shared.

.. warning::

   The Isaac RTX and OVRTX renderers cap the number of scene partitions at 15625.
   Requesting more than 15625 environments with scene partitioning enabled discards
   the additional partitions, and ``rtx.scenedb.plugin`` logs ``SceneDbContext : Maximum
   number of scene partitions (15625) reached. Additional scene partitions will be
   discarded.`` Environments beyond that count then share a partition with another
   environment, so their tiled camera views can show another environment's geometry.
   See :ref:`known-issues-scene-partition-count-cap` for details.

Architecture Overview
---------------------

The renderer system consists of:

1. **BaseRenderer** — Abstract base class defining the rendering lifecycle and interface
2. **RendererCfg** — A ``BackendCfg`` subclass; each backend extends it with backend-specific options and declares
   its implementation in ``class_type``
3. **Concrete implementations** — Backend-specific renderers in extension packages
4. **SimulationContext** — Owns renderer instances in its backend registry and shares them across equal configurations
   of the same concrete type through ``get_or_create_backend(renderer_cfg)``.
5. **RenderContext** — Coordinates initialization, stage preparation, scene updates, and material writers using
   a filtered view of the simulation registry. It does not construct, cache, or close renderer instances.

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.renderers import BaseRenderer
   from isaaclab_newton.renderers import NewtonWarpRendererCfg

   # Create a Newton Warp renderer (no Isaac Sim required)
   sim_ctx = sim_utils.SimulationContext.instance()
   # Reuse a matching renderer or construct one through instantiate(cfg).
   renderer: BaseRenderer = sim_ctx.get_or_create_backend(NewtonWarpRendererCfg())
   assert isinstance(renderer, BaseRenderer)

For the legacy Isaac RTX renderer (requires Isaac Sim):

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.renderers import BaseRenderer
   from isaaclab_physx.renderers import IsaacRtxRendererCfg

   # Create an RTX renderer
   sim_ctx = sim_utils.SimulationContext.instance()
   # Reuse a matching renderer or construct one through instantiate(cfg).
   renderer: BaseRenderer = sim_ctx.get_or_create_backend(IsaacRtxRendererCfg())

For RTX renderer settings, see
:doc:`/source/how-to/configure_rendering`.

.. _renderer-camera-batching:

Batching camera renders
-----------------------

:meth:`~isaaclab.renderers.BaseRenderer.render_batch` accepts a sequence of render-data objects
owned by the same renderer. Prepare the camera poses, intrinsics, and shared scene state before
rendering, then read each camera's output. An empty sequence performs no rendering.

.. code-block:: python

   renderer.render_batch([first_render_data, second_render_data])
   renderer.read_output(first_render_data, first_camera_data)
   renderer.read_output(second_render_data, second_camera_data)

:meth:`~isaaclab.renderers.BaseRenderer.render` continues to accept a single render-data object.
The default ``render_batch()`` implementation calls ``render()`` for each entry, so existing
custom renderers and single-camera callers need no changes. OVRTX steps all registered render
products together but publishes only the requested cameras' observations.

With eager sensor updates (``scene.cfg.lazy_sensor_update=False``), ``scene.update()`` advances
sensor clocks in scene order and collects batch-capable sensors. After the loop, the camera
batch implementation prepares the remaining due captures for submission.
:meth:`~isaaclab.renderers.RenderContext.render_into_cameras` groups these requests by renderer
instance, renders each group, and reads its outputs. The context does not retain a pending-camera
queue. Each camera retains its own update period and reset state.

With lazy sensor updates, reading a camera's ``data`` refreshes only that camera's sensor buffers
and capture timestamps. It does not refresh peer camera sensors sharing the renderer.

Core concepts
-------------

- **Use the simulation registry**: Always acquire renderers with a renderer-specific config class through
  ``sim_ctx.get_or_create_backend(IsaacRtxRendererCfg())``. Do not import or instantiate concrete backend classes
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
     renderer: BaseRenderer = sim_ctx.get_or_create_backend(OVRTXRendererCfg())

- **Opaque render data**: The render data object returned by :meth:`~isaaclab.renderers.BaseRenderer.create_render_data` is passed to
  subsequent renderer methods. It should be completely opaque to the caller: inspecting or modifying it
  via get/set attributes is an anti-pattern and breaks the API contract.

.. note::

   The :class:`~isaaclab.renderers.BaseRenderer` class is under active development and may change without notice.

.. _renderers-async-data-flow:

Asynchronous Rendering
----------------------

:attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.async_rendering` returns a camera's previous capture
while rendering its next image. This allows rendering to overlap simulation and Python work.
The first capture, including after reset, waits for a fresh image. The ovstage path stays synchronous.

Latency is one capture, not necessarily one physics or control step. State observations remain
current, so enabling this option changes the policy's observation timing; training may need to
account for that delay. Physics and policy computation may also use the GPU.

For a camera capturing once per environment step, steady-state operation is:

.. code-block:: text

   Synchronous:   physics k -> submit image k -> wait/read image k -> policy
   Asynchronous:  physics k -> submit image k -> wait/read image k-1 -> policy
                                   |                                 |
                                   +---- image k renders ------------+--> physics k+1

The first image is returned immediately after priming and repeated once on the next capture.

Each camera owns its pending observations. Reading camera A never changes camera B's pixels or
metadata, even when they share a native render submission. Reset discards that camera's pending
observations. A partial environment reset re-primes the whole tiled camera product, without
invalidating other cameras.

Live fields such as ``camera.data.pos_w`` and ``camera.data.intrinsic_matrices`` remain current.
Use the capture metadata when pairing delayed pixels with a pose or calibration, for example
when deprojecting depth:

.. code-block:: python

   data = camera.data
   depth = data.output["distance_to_image_plane"]
   capture = data.info["distance_to_image_plane"]["capture"]
   position, orientation = capture["pos_w"], capture["quat_w_world"]
   intrinsics, frame = capture["intrinsic_matrices"], capture["frame"]

Capture metadata remains unchanged on later captures. Direct renderer callers use
:meth:`~isaaclab.renderers.BaseRenderer.prepare_capture` before submitting a capture and
:meth:`~isaaclab.renderers.BaseRenderer.read_output` to publish it.

SDP fills renderer-owned transform and geometry buffers, including deformable meshes, particles,
and cables. Two buffers alternate per write and are reused only after OVRTX consumes them.
Physics can therefore update its live arrays while the previous capture renders.

See Also
--------

- :doc:`/source/concepts/native-physics-api/scene_data_providers`: how scene data flows from physics backends to renderers
- :doc:`/source/concepts/visualization` — lightweight visualizer backends for interactive feedback
