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

Camera processing uses the same :class:`~isaaclab.utils.modifiers.ModifierCfg` list as other
observations. The observation manager passes ``(env, data)`` to each modifier in order. A stateful
modifier can own buffers and implement ``reset(env_ids)`` and ``close()``. Each modifier takes and
returns one tensor; the manager remains independent of camera types and the number of cameras.

PPISP
~~~~~

:class:`~isaaclab_ppisp.PpispModifierCfg` applies PPISP to scene-linear ``rgb_radiance`` from a
camera, producing uint8 ``rgb`` and ``rgba``. Add it to an image observation's modifiers:

.. code-block:: python

   from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
   from isaaclab_ppisp import PpispCfg, PpispModifierCfg, ppisp_camera_input

   ppisp = PpispModifierCfg(sensor_cfg=SceneEntityCfg("front_camera"),
                            isp_cfg=PpispCfg(inputs={"exposureOffset": 1.5}))
   camera_image = ObservationTermCfg(
       func=ppisp_camera_input,
       params={"modifier_cfg": ppisp},
       modifiers=[ppisp],
   )

The modifier discovers USD ``ppisp:*`` attributes during observation preparation, before simulation
reset, and requests private ``rgb_radiance`` from the camera when needed. ``AUTO_CAMERA`` reads the
first matching camera prim; ``AUTO_ANY`` also searches the stage. Discovery without a match passes
through the camera's raw RGB/RGBA. With ``input_source="previous"``, supply an observation source and
earlier modifier that produce scene-linear radiance instead of using ``ppisp_camera_input``; PPISP then
makes no renderer request. Set ``output="rgba"``
to select RGBA, or use a separate observation term for each color output. Separate observation
terms own separate PPISP buffers. ``normalize=True`` divides by 255 and subtracts the per-image spatial
mean, and ``permute=True`` returns NCHW instead of NHWC.

The camera publishes its capture frame in ``CameraData.info[name]["capture"]["frame"]``. PPISP uses
the published frame object to reuse its result on repeated observation reads. Delayed captures retain
the frame metadata that belongs to their image. Partial environment resets reach the modifier through
``reset(env_ids)``; PPISP itself has no temporal state.

For use outside observations, create ``PpispModifierCfg``, call ``cfg.func.prepare_scene(cfg, env)`` after camera
spawning and before ``sim.reset()``, then construct ``cfg.func(cfg, image_shape, env=env)``. Call the
modifier with ``(env, ppisp_camera_input(env, cfg))``, read its returned tensor,
and call ``close()`` when done. The returned buffer is reused on a later capture; clone it if a caller
must retain the previous image.

.. important::

   ``CameraCfg.isp_cfg`` and ``CameraISPMode`` were removed. Put ``PpispModifierCfg`` in the
   observation term's ``modifiers`` instead. Camera ``data.output`` contains raw renderer results.
   Requesting ``rgb_radiance`` on Isaac RTX or OVRTX neutralizes exposure for all outputs from that
   camera prim. Use a separate prim if the authored-exposure output is also needed. OVRTX routes
   Gaussian pixels through ``HdrColor`` for this input; Newton Warp does not change exposure setup.

At simulation startup, cameras prepare configured and requested signals before initializing render
data. This includes inputs requested by modifiers during observation preparation. Camera buffers
remain borrowed read-only, while each modifier owns its output buffers.

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
