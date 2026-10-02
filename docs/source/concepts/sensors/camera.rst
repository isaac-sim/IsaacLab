.. _concepts_sensors_camera:
.. _overview_sensors_camera:

.. currentmodule:: isaaclab

Camera
======

A :class:`~sensors.Camera` defines what to capture: camera pose, projection, resolution, sampling
period, and output data types. A renderer defines how those images are produced. Keeping these
responsibilities separate lets one camera configuration work with different physics and rendering
backends.

Camera data is expensive compared with low-dimensional state. Isaac Lab therefore batches the
camera copies from cloned environments into tiled render passes and exposes the de-tiled result as
one device-resident buffer per requested output.

Rendering model
---------------

:attr:`~sensors.CameraCfg.renderer_cfg` selects the renderer. A plain
:class:`~isaaclab.renderers.RendererCfg` requests the runtime default. Use a concrete configuration
when the renderer must be fixed:

.. list-table::
   :header-rows: 1
   :widths: 34 23 43

   * - Renderer configuration
     - Requires Isaac Sim
     - Characteristics
   * - :class:`~isaaclab_physx.renderers.IsaacRtxRendererCfg`
     - Yes
     - Replicator and RTX rendering through Isaac Sim
   * - :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`
     - No
     - Kit-less RTX rendering through ``isaaclab_ov``
   * - :class:`~isaaclab_newton.renderers.NewtonWarpRendererCfg`
     - No
     - Kit-less Warp rasterization through Newton

For an environment that exposes renderer presets, select the renderer at launch instead of editing
the scene configuration:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Camera-Direct renderer=newton_renderer

See :doc:`/source/concepts/backends_and_presets` for preset discovery and
:ref:`renderer-visual-comparison` for a same-scene comparison of the renderer outputs.

With OVRTX, separate camera sensors share the scene and renderer when their renderer
configurations match, while each sensor owns a separate tiled render product. For example,
``env_0/head_camera`` and ``env_1/head_camera`` form one product, while
``env_0/wrist_camera`` and ``env_1/wrist_camera`` form another. The sensors can use different
resolutions and output types, and their poses update independently. Define all camera prims
before initializing the simulation so they are included in the scene exported to OVRTX.

.. _camera-configuration:

Configure a camera
------------------

A camera can spawn a pinhole or fisheye camera prim, or bind to a camera already on the stage.
``offset`` uses the convention declared on :class:`~sensors.CameraCfg.OffsetCfg`:

* ``world``: forward ``+X``, up ``+Z``.
* ``ros``: forward ``+Z``, up ``-Y``.
* ``opengl``: forward ``-Z``, up ``+Y``.

.. code-block:: python

   import isaaclab.sim as sim_utils
   from isaaclab.sensors import CameraCfg
   from isaaclab_newton.renderers import NewtonWarpRendererCfg

   front_camera = CameraCfg(
       prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
       update_period=0.05,
       height=240,
       width=320,
       data_types=["rgb", "depth", "normals"],
       spawn=sim_utils.PinholeCameraCfg(
           focal_length=24.0,
           horizontal_aperture=20.955,
           clipping_range=(0.1, 20.0),
       ),
       offset=CameraCfg.OffsetCfg(
           pos=(0.45, 0.0, 0.1),
           rot=(0.0, 0.0, 0.0, 1.0),
           convention="world",
       ),
       renderer_cfg=NewtonWarpRendererCfg(),
   )

A renderer instance is reused only when cameras use equal renderer configurations of the same
concrete configuration type. Different renderer settings create distinct instances. Camera
configurations, including output types and backgrounds, remain per sensor.

Read camera data
----------------

:attr:`~sensors.CameraData.output` maps each requested name to a
:class:`~isaaclab.utils.warp.ProxyArray`. For ``N`` camera views, height ``H``, width ``W``, and
``C`` channels, each output has shape ``(N, H, W, C)``. Use ``torch`` for a cached zero-copy Torch
view or ``warp`` for the underlying Warp array:

.. code-block:: python

   camera_data = scene["front_camera"].data
   rgb = camera_data.output["rgb"].torch
   depth = camera_data.output["depth"].torch
   intrinsics = camera_data.intrinsic_matrices.torch

Camera pose and intrinsic buffers are also ``ProxyArray`` objects. ``pos_w`` has shape ``(N, 3)``,
``intrinsic_matrices`` has shape ``(N, 3, 3)``, and camera quaternions have shape ``(N, 4)`` in
``(x, y, z, w)`` order. Set ``update_latest_camera_pose=True`` only when current pose data is needed;
updating it adds frame-query overhead.

.. _camera-output-types:

Output types
------------

The camera validates ``data_types`` against the renderer, then allocates the channel count and data
type declared by each :class:`~isaaclab.renderers.RenderBufferSpec`.

.. list-table:: Common output contracts
   :header-rows: 1
   :widths: 34 22 44

   * - Name
     - Channels and type
     - Meaning
   * - ``rgb`` / ``rgba``
     - 3 / 4, ``uint8``
     - Low-dynamic-range color
   * - ``rgb_hdr``
     - 3, ``float32``
     - High-dynamic-range RGB using the active camera settings
   * - ``rgb_radiance``
     - 3, ``float32``
     - Scene-linear RGB before exposure and response, in renderer-relative intensity units
   * - ``albedo``
     - 4, ``uint8``
     - Material base color
   * - ``depth`` / ``distance_to_image_plane``
     - 1, ``float32``
     - Distance [m] along the camera optical axis
   * - ``distance_to_camera``
     - 1, ``float32``
     - Euclidean distance [m] from the optical center
   * - ``normals``
     - 3, ``float32``
     - Local surface normal ``(x, y, z)``
   * - ``motion_vectors``
     - 2, ``float32``
     - Image-space motion; positive ``x`` is left and positive ``y`` is up
   * - ``semantic_segmentation``
     - 4 ``uint8`` or 1 ``int32``
     - Semantic color or ID per pixel
   * - ``instance_segmentation``
     - 4 ``uint8`` or 1 ``int32``
     - Semantically labeled instance color or ID per pixel
   * - ``instance_id_segmentation_fast``
     - 4 ``uint8`` or 1 ``int32``
     - USD-prim instance color or ID per pixel

``depth`` is an alias of ``distance_to_image_plane``. Colorized segmentation uses RGBA ``uint8``;
non-colorized segmentation uses one ``int32`` ID channel. Label and prim-path mappings are stored in
``camera_data.info[output_name]``.

Requesting ``rgb_hdr`` alone preserves the renderer's existing camera settings. Requesting
``rgb_radiance`` makes the renderer prepare its source for processing before exposure and camera
response. If both raw names are requested, they alias the same active HDR source.

.. note::

   On Isaac RTX and OVRTX, ``rgb_radiance`` is not a native renderer output. These renderers derive
   it from HDR color by authoring neutral exposure on the camera prim, so every output rendered from
   that prim, including ``rgb``, ``rgba``, and ``rgb_hdr``, loses the authored exposure. This also
   applies to other camera sensors that share the prim. A single render product cannot return both
   authored-exposure color and ``rgb_radiance``; use a separate camera prim when both are required.
   Newton Warp has no exposure model and is not affected. A native pre-exposure radiance output has
   been requested from the RTX team (NVBug 6858736).

.. figure:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/camera-renderer-isaac-rtx.webp
   :align: center
   :figwidth: 100%
   :alt: RGB camera output

   Isaac RTX RGB output. The animation shows the six material spheres falling onto the table.

.. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-depth.png
   :align: center
   :figwidth: 100%
   :alt: Depth camera output

   Isaac RTX depth output. Display colors encode optical-axis distance; the sensor returns metric
   values [m].

.. _camera-supported-annotators:

Renderer support
~~~~~~~~~~~~~~~~

The common API does not imply that every renderer produces every output. For same-scene examples
across these backends, see the :ref:`renderer visual comparison <renderer-visual-comparison>`. The
current support matrix is:

.. list-table::
   :header-rows: 1
   :widths: 38 20 20 22

   * - Output
     - Isaac RTX
     - OVRTX
     - Newton Warp
   * - ``rgb``, ``rgba``, ``rgb_hdr``, ``rgb_radiance``
     - Yes
     - Yes
     - Yes
   * - ``depth`` and both distance outputs
     - Yes
     - Yes
     - Yes
   * - ``normals``
     - Yes
     - Yes
     - Yes
   * - ``albedo``
     - Isaac Sim 6.0+
     - Yes
     - Yes
   * - ``motion_vectors``
     - Yes
     - Yes
     - No
   * - semantic and instance segmentation
     - Yes
     - Yes
     - Yes
   * - ``instance_id_segmentation_fast``
     - Yes
     - No
     - No
   * - ``simple_shading_*`` modes
     - Isaac Sim 6.0+
     - Yes
     - No

Querying an unsupported output fails during camera initialization. Renderer configuration controls
semantic filters, segmentation colorization, and depth clipping where those options are
backend-specific.

.. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-normals.png
   :align: center
   :figwidth: 100%
   :alt: Camera surface-normal output

   Isaac RTX normals output. Red, green, and blue encode the surface normal X, Y, and Z components.

.. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-semantic-segmentation.png
   :align: center
   :figwidth: 100%
   :alt: Semantic segmentation output

   Isaac RTX semantic segmentation. One color represents each class: the six spheres share one
   class, while the table and backdrop use separate classes.

.. figure:: ../../_static/overview/sensors/camera-renderer-isaac-rtx-instance-segmentation.png
   :align: center
   :figwidth: 100%
   :alt: Instance segmentation output

   Isaac RTX instance segmentation. Each sphere receives its own color, distinguishing objects
   that share the same semantic class.

Background color
----------------

When :attr:`~sensors.CameraCfg.background_color` is ``None``, each renderer uses its default
background. Set a normalized RGB tuple to use a solid color for pixels that miss all geometry:

.. code-block:: python

   from isaaclab.utils import replace

   mask_camera = replace(front_camera, background_color=(0.0, 0.0, 0.0))

The setting is per camera. Cameras with renderer-default and solid backgrounds can coexist in one
scene.

.. _camera-post-processing:

Process camera observations
---------------------------

:class:`~envs.mdp.visual_observations.processed_image` configures an ordered image-processing chain as an
observation term:

.. code-block:: text

   renderer -> camera buffers -> processor 1 -> processor 2 -> observation

Each observation term owns its processor state. Separate terms can consume the same camera with
independent processing, and cameras can share a renderer. Processors declare
their required inputs and produced outputs with :class:`~isaaclab.renderers.RenderBufferSpec`:
channel count, Warp dtype, layout, device, and color space. The term resolves these requirements
before renderer setup and requests the camera inputs it needs. Inputs may come from the camera or
an earlier processor. An unavailable input or incompatible declaration raises an initialization error;
the chain performs no implicit
layout, dtype, device, or color conversions.

Buffers use ``NHWC`` layout on the camera device. A processor can use private intermediate names
without adding them to ``CameraCfg.data_types``. The chain allocates its intermediate and output
buffers once. Its ``rgb`` and ``rgba`` outputs share storage; ``rgb`` is a strided view of the first
three ``rgba`` channels. ``CameraData.output`` continues to expose the configured camera outputs;
the observation term returns its processed result.

PPISP
~~~~~

:class:`~isaaclab_ppisp.PpispProcessorCfg` configures PPISP (Physically Plausible Image Signal
Processing). It applies responsivity, exposure, vignetting, color correction, and a camera response
function to ``rgb_radiance``, producing ``rgb`` and ``rgba``. The input is scene-linear RGB before
exposure and response, expressed in renderer-relative intensity units. The same processor works
with Isaac RTX, OVRTX, and Newton Warp.

PPISP declares ``color_space="camera_response"`` because its configurable response curve does not
guarantee an sRGB transfer function. Downstream operations can require that encoding or leave
``color_space=None`` when they accept any encoding.

.. code-block:: python

   from isaaclab.envs import mdp
   from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
   from isaaclab_ppisp import PpispCfg, PpispProcessorCfg

   camera_image = ObservationTermCfg(
       func=mdp.processed_image,
       params={
           "sensor_cfg": SceneEntityCfg("front_camera"),
           "processors": [PpispProcessorCfg(isp_cfg=PpispCfg(inputs={"exposureOffset": 1.5}))],
           "data_type": "rgb",
           "normalize": False,
       },
   )

Add this term to an :class:`~managers.ObservationGroupCfg` in the environment's observation
configuration. ``normalize=False`` is the default and returns the persistent ``uint8`` RGB result.
``normalize=True`` supports RGB/RGBA and follows :func:`~envs.mdp.observations.image`: it converts to
``float32``, divides by 255, and subtracts the spatial mean for each view and channel. Its output
and mean buffers are reused. ``permute=True`` returns a cached ``NCHW`` view; the default is ``NHWC``.

For automatic discovery, use ``PpispProcessorCfg()`` or explicitly pass
``isp_cfg=PpispDiscoveryMode.AUTO_CAMERA``. ``AUTO_CAMERA`` reads ``ppisp:*`` attributes from the
first matched camera prim. ``AUTO_ANY`` also falls back to the first camera with PPISP attributes
on the stage. Import ``PpispDiscoveryMode`` from ``isaaclab_ppisp``. Discovery runs once while preparing the
observation term; an unsuccessful automatic lookup disables that processor. A static
configuration is shared by all cloned views in one camera batch. Controller weights can predict
per-view exposure and color parameters, while the remaining coefficients stay shared.

The observation term resolves PPISP's ``rgb_radiance`` input independently of ``CameraCfg.data_types``.
If an earlier processor produces that signal, PPISP consumes its result. Otherwise the term requests
it from the camera, and the renderer selects the required exposure setup. For example, requesting
only ``rgb`` as camera output keeps the radiance intermediate private. A subsequent processor can
consume PPISP's RGB result; a future visual domain randomization processor can be added in the same way.

.. important::

   Supplying ``rgb_radiance`` from Isaac RTX or OVRTX disables RTX auto-exposure, authors neutral
   ``exposure:*`` values, and applies the
   ``OmniRtxCameraAutoExposureAPI_1`` and ``OmniRtxCameraExposureAPI_1`` schemas on every matched
   camera prim. This retains the existing PPISP camera-wide exposure behavior: raw outputs and
   every observation using that camera reflect the resulting renderer settings. ``rgb_hdr`` and
   ``rgb_radiance`` share that source when both are requested. Use separate cameras for separate
   exposure settings. When an earlier processor supplies ``rgb_radiance``, PPISP adds no renderer
   request for that signal.

   OVRTX also configures each HDR-consuming render product to route Gaussian pixels through
   ``HdrColor``. This applies to public ``rgb_hdr`` and private ``rgb_radiance`` requests;
   no external Gaussian tone-mapping setting is required.

.. important::

   **Breaking change:** ``CameraCfg.isp_cfg`` and ``isaaclab.sensors.camera.CameraISPMode`` were removed.
   To migrate, remove the ``isp_cfg`` argument from the camera configuration and pass its previous
   value as ``processors=[PpispProcessorCfg(isp_cfg=existing_cfg)]`` in the ``params`` of an
   ``ObservationTermCfg(func=mdp.processed_image, ...)``, as shown above. Replace discovery enum
   imports with ``from isaaclab_ppisp import PpispDiscoveryMode``; ``AUTO_CAMERA`` and ``AUTO_ANY``
   retain their discovery behavior.

   Read the processed image from the environment's observations. ``camera.data.output`` contains
   the renderer's raw camera outputs and no longer returns PPISP results. To access the image by
   its term name within an observation group, set that group's ``concatenate_terms=False``.

Using the generic pipeline without PPISP does not import the optional ``isaaclab_ppisp`` package.

Run the ``ppisp-camera`` example for a PPISP workflow using the observation API:

.. code-block:: bash

   uv run --extra isaacsim isaaclab example ppisp-camera \
      --renderer newton_renderer --max_steps 60

Add a processor
~~~~~~~~~~~~~~~

A :class:`~sensors.SensorPostProcessorCfg` factory receives the configuration and a
:class:`~sensors.CameraPostProcessorContext` containing the stage, camera paths, image
dimensions, and device. It returns a :class:`~sensors.SensorPostProcessor` with buffer
declarations and callbacks, or ``None`` to disable the stage. Keep state inside that factory so
observation terms remain independent.

For example, this stateless processor inverts RGB values after PPISP. Save the kernel in a Python
module so Warp can inspect its source:

.. code-block:: python

   import warp as wp

   from isaaclab.renderers import RenderBufferSpec
   from isaaclab.sensors.post_processing import SensorPostProcessor, SensorPostProcessorCfg


   @wp.kernel
   def invert_rgb(source: wp.array4d(dtype=wp.uint8), output: wp.array4d(dtype=wp.uint8)):
       view, row, column, channel = wp.tid()
       output[view, row, column, channel] = wp.uint8(255) - source[view, row, column, channel]


   def make_invert_processor(cfg, context):
       source = output = None

       def initialize(inputs, outputs):
           nonlocal source, output
           source = inputs["rgb"].warp
           output = outputs["rgb"].warp

       def process(mask):
           wp.launch(
               invert_rgb,
               dim=(context.num_views, context.height, context.width, 3),
               inputs=[source, output],
               device=context.device,
           )

       return SensorPostProcessor(
           inputs=cfg.inputs,
           outputs=cfg.outputs,
           initialize=initialize,
           process=process,
           in_place=True,
       )


   rgb_spec = RenderBufferSpec(3, wp.uint8, color_space="camera_response")
   inverted_image = ObservationTermCfg(
       func=mdp.processed_image,
       params={
           "sensor_cfg": SceneEntityCfg("front_camera"),
           "processors": [
               PpispProcessorCfg(isp_cfg=PpispCfg()),
               SensorPostProcessorCfg(
                   func=make_invert_processor,
                   inputs={"rgb": rgb_spec},
                   outputs={"rgb": rgb_spec},
               ),
           ],
           "data_type": "rgb",
       },
   )

``initialize(inputs, outputs)`` receives persistent ``ProxyArray`` bindings once. Allocate scratch
storage there and write into the supplied outputs in ``process(mask)``. Set ``in_place=True`` only
when the operation supports shared input/output storage; this permits reuse of a preceding stage's
output. Camera inputs are borrowed read-only, so callbacks must also handle separate input/output
buffers. Replacing bound arrays would invalidate renderer and downstream bindings.

Processing runs when the observation term consumes a fresh camera frame, on the current Warp
stream. If a processor uses Torch or another stream, it must establish stream dependencies before
returning. Repeated observation reads of a cached camera frame do not execute the chain again.
The boolean device mask identifies updated views; use it to advance temporal state only for those
views. Renderers refresh the image batch, so
an image transform may still need to process all views to keep its outputs coherent.

Stateful processors can provide ``reset(mask)`` for partial environment resets and ``close()`` for
cleanup. The observation manager forwards its reset selection and closes its terms when the
environment closes. Cleanup must also tolerate partial initialization. These callbacks and
declarations are sufficient to add another processor without changing renderer code.

The observation manager takes its usual snapshot before applying modifiers, noise, clipping,
scaling, delay, and history. Code that calls the term directly should clone its result before modifying
it or retaining it beyond the next rendered frame.

Preparation and ownership
~~~~~~~~~~~~~~~~~~~~~~~~~

``processed_image`` uses :meth:`~managers.ManagerTermBase.prepare_scene` after scene spawning and
before the first simulation reset. This lets PPISP discover USD camera settings and resolve its
``rgb_radiance`` input before renderer setup or stage export. The observation manager adopts this
prepared term, retaining its processor state and allocated buffers for its lifetime.

The camera provides rendered inputs and a frame generation counter. Other consumers can call
``Camera.request_render_inputs(("rgb_radiance",))`` before camera initialization, then read
``render_outputs`` for persistent raw buffers. The ``render_generation`` attribute changes when a new image is
published. The per-view ``render_frame`` counters identify updated views, including delayed asynchronous captures;
the existing ``frame`` counters describe live camera updates. ``render_buffer_specs`` and ``camera_prim_paths`` are
available during preparation.

At simulation startup, cameras prepare their configured and requested signals before
any camera initializes render data. This ensures shared renderer exports include every camera's settings.

Processing state, ordering, normalization, and observation caching belong to the term. A compatible
additional processor can therefore be composed in the observation configuration without editing
the camera or renderer.

Bindings persist for the environment's lifetime. If simulation stop/start recreates camera buffers,
recreate the environment before reading processed observations again; existing terms reject the
replaced buffers.

Performance and validation
--------------------------

Image memory and rendering cost scale with the number of environments, resolution, channel count,
and requested outputs. Start camera-based tasks with a small environment count, verify shapes and
renderer support, and then scale while monitoring GPU memory. Avoid requesting buffers that the task
does not consume.

Tiled rendering batches the cloned views into shared render passes, but it does not remove the memory
cost of the de-tiled outputs or downstream vision models. The camera follows the shared sensor
``update_period`` contract; choose a period that matches the observation cadence instead of rendering
at every physics step by default.

A runnable camera example is available as ``camera``:

.. code-block:: bash

   uv run --extra isaacsim isaaclab example camera

For saving output to disk, see :doc:`/source/how-to/save_camera_output`. For renderer selection
and customization, see :doc:`/source/how-to/configure_rendering`.
