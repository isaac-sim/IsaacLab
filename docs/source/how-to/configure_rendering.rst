:orphan:

Select and configure a Renderer
===============================

Renderers produce camera-sensor observations. They are distinct from visualizers, which provide
interactive views for people. Select a renderer for the images your policy or data pipeline needs,
then tune only the options that affect that workflow.

Choose a renderer
-----------------

For tasks that advertise renderer presets, choose a compatible physics and renderer pair at launch.
Use ``--task <task-name> --help`` to see the presets a task actually supports; renderer availability
is task-specific.

.. list-table:: Renderer choices
   :header-rows: 1
   :widths: 20 25 30 25

   * - Renderer
     - Choose it when
     - Trade-off
     - Typical command
   * - Newton Warp
     - You need the lowest VRAM use and high camera throughput for Newton training.
     - Lightweight rasterization; it has a smaller output set and does not provide motion vectors
       or full RTX material transport.
     - ``physics=newton_mjwarp renderer=newton_renderer presets=rgb``
   * - OVRTX
     - You need scalable kit-less RTX rendering and higher visual fidelity.
     - Uses more VRAM than Newton Warp. Choose RTX Minimal outputs when throughput matters more
       than photo-real appearance.
     - ``physics=newton_mjwarp renderer=ovrtx presets=rgb``
   * - Isaac RTX (legacy)
     - A workflow must run through Isaac Sim/Kit or needs its broad RTX and Replicator output set.
     - Requires Isaac Sim and PhysX; do not use it as the default performance path for new work.
     - ``physics=isaacsim_physx renderer=isaacsim_rtx presets=rgb``

For example, start a camera task with the low-VRAM Newton renderer:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Camera \
      physics=newton_mjwarp renderer=newton_renderer presets=rgb

Switch the same supported task to the higher-fidelity kit-less RTX renderer:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Camera \
      physics=newton_mjwarp renderer=ovrtx presets=rgb

The :ref:`renderer details <renderer-details>` and the
:ref:`camera renderer support matrix <camera-supported-annotators>` are the authoritative references
for output availability and runtime requirements. Do not compare the renderer choices through a
camera-count heuristic: measure the complete task and observation configuration you intend to train.

Customize Newton Warp
---------------------

Newton Warp is the throughput-oriented choice. Begin with its defaults, then enable only the image
features that matter to the policy. Shadows, textures, ambient lighting, traversal order, and tile
dimensions are controlled by :class:`~isaaclab_newton.renderers.NewtonWarpRendererCfg`.

For a task with a camera renderer configuration, enable directional-light shadows with an override:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Camera \
      physics=newton_mjwarp renderer=newton_renderer presets=rgb \
      env.scene.tiled_camera.renderer_cfg.enable_shadows=true

When defining a camera in Python, configure the renderer directly:

.. code-block:: python

   from isaaclab_newton.renderers import NewtonWarpRendererCfg

   renderer_cfg = NewtonWarpRendererCfg(
       enable_textures=True,
       enable_shadows=True,
       render_order="tiled",
   )

Use ``render_order`` and the tile dimensions only after profiling a representative scene; they are
implementation-level throughput controls, not visual-quality settings.

Customize OVRTX
---------------

OVRTX provides RTX Minimal and photo-real paths without Isaac Sim. Use the regular ``rgb`` output
when material appearance, reflections, transparency, or the broader RTX output set matter. For
training that only needs simplified color, select a ``simple_shading_*`` preset instead:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl \
      --task Isaac-Cartpole-Camera \
      physics=newton_mjwarp renderer=ovrtx \
      presets=simple_shading_diffuse_mdl

In RTX Minimal mode, :class:`~isaaclab_ov.renderers.OVRTXRendererCfg.enable_shadows` controls
directional-light shadow rays. They improve visual faithfulness but cost render time. Path-traced
OVRTX outputs always cast shadows, so this option does not affect regular ``rgb`` or other AOVs.

.. code-block:: python

   from isaaclab_ov.renderers import OVRTXRendererCfg

   renderer_cfg = OVRTXRendererCfg(enable_shadows=True)

Customize Isaac RTX
-------------------

Isaac RTX remains available for Isaac Sim and PhysX workflows. The settings below are specific to
that legacy renderer; use Newton Warp or OVRTX for new Newton and kit-less workloads.

.. note::

   Requesting one of the ``simple_shading_*`` camera data types without a regular color output
   switches that camera's render product to RTX Minimal mode; the data type selects the shading
   level. The switch applies per render product, so other cameras and the Kit viewport keep their
   configured render mode. RTX Minimal uses only the first ``DistantLight`` prim, ignores
   ``DomeLight`` prims, and may also use configured ambient lighting. When ``rgb``, ``rgba``, or
   ``rgb_hdr`` is requested from the same render product, it retains its configured render mode to
   preserve the color output; the ``simple_shading_*`` output remains available but does not receive
   the RTX Minimal performance improvement.

Overriding Specific Rendering Settings
--------------------------------------

RTX rendering settings can be overridden via
:class:`~isaaclab_physx.renderers.IsaacRtxRendererGlobalSettingsCfg`.

There are 2 ways to provide settings that override the defaults.

1. :class:`~isaaclab_physx.renderers.IsaacRtxRendererGlobalSettingsCfg`
   supports overriding specific settings via user-friendly setting names that
   map to underlying RTX settings.
   For example:

   .. code-block:: python

      global_settings = IsaacRtxRendererGlobalSettingsCfg(
         # user-friendly setting overrides
         enable_translucency=True,  # render glass / transmissive surfaces
         enable_reflections=True,  # render reflections
         dlss_mode=3,  # 0 (Performance), 1 (Balanced), 2 (Quality, the default), 3 (Auto)
      )

   List of user-friendly settings.

   .. table::
      :widths: 25 75

      +----------------------------+--------------------------------------------------------------------------+
      | enable_translucency        | Bool. Enables translucency for specular transmissive surfaces such as    |
      |                            | glass at the cost of some performance.                                   |
      +----------------------------+--------------------------------------------------------------------------+
      | enable_reflections         | Bool. Enables reflections at the cost of some performance.               |
      +----------------------------+--------------------------------------------------------------------------+
      | enable_global_illumination | Bool. Enables Diffused Global Illumination at the cost of some           |
      |                            | performance.                                                             |
      +----------------------------+--------------------------------------------------------------------------+
      | antialiasing_mode          | Literal["Off", "FXAA", "DLSS", "TAA", "DLAA"].                           |
      |                            |                                                                          |
      |                            | DLSS: Boosts performance by using AI to output higher resolution frames  |
      |                            | from a lower resolution input. DLSS samples multiple lower resolution    |
      |                            | images and uses motion data and feedback from prior frames to reconstruct|
      |                            | native quality images.                                                   |
      |                            | DLAA: Provides higher image quality with an AI-based anti-aliasing       |
      |                            | technique. DLAA uses the same Super Resolution technology developed for  |
      |                            | DLSS, reconstructing a native resolution image to maximize image quality.|
      +----------------------------+--------------------------------------------------------------------------+
      | enable_dlssg               | Bool. Enables the use of DLSS-G. DLSS Frame Generation boosts performance|
      |                            | by using AI to generate more frames. This feature requires an Ada        |
      |                            | Lovelace architecture GPU and can hurt performance due to additional     |
      |                            | thread-related activities.                                               |
      +----------------------------+--------------------------------------------------------------------------+
      | enable_dl_denoiser         | Bool. Enables the use of a DL denoiser, which improves the quality of    |
      |                            | renders at the cost of performance.                                      |
      +----------------------------+--------------------------------------------------------------------------+
      | dlss_mode                  | Literal[0, 1, 2, 3]. For DLSS anti-aliasing, selects the performance/    |
      |                            | quality tradeoff mode. Valid values are 0 (Performance), 1 (Balanced),   |
      |                            | 2 (Quality), or 3 (Auto).                                                |
      +----------------------------+--------------------------------------------------------------------------+
      | enable_direct_lighting     | Bool. Enable direct light contributions from lights.                     |
      +----------------------------+--------------------------------------------------------------------------+
      | samples_per_pixel          | Int. Defines the Direct Lighting samples per pixel. Higher values        |
      |                            | increase the direct lighting quality at the cost of performance.         |
      +----------------------------+--------------------------------------------------------------------------+
      | enable_shadows             | Bool. Enables shadows at the cost of performance. When disabled, lights  |
      |                            | will not cast shadows.                                                   |
      +----------------------------+--------------------------------------------------------------------------+
      | enable_ambient_occlusion   | Bool. Enables ambient occlusion at the cost of some performance.         |
      +----------------------------+--------------------------------------------------------------------------+


2. For more control,
   :class:`~isaaclab_physx.renderers.IsaacRtxRendererGlobalSettingsCfg`
   allows you to override any RTX setting by using the ``carb_settings``
   argument.

   The full NVIDIA RTX renderer documentation can be found at
   https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer.html.

   An example usage of ``carb_settings``.

   .. code-block:: python

      global_settings = IsaacRtxRendererGlobalSettingsCfg(
         # raw carb setting overrides
         carb_settings={
            "rtx.translucency.enabled": False,
            "rtx.reflections.enabled": False,
            "rtx.domeLight.upperLowerStrategy": 3,
         }
      )


Current Limitations
-------------------

For performance reasons, we default to using DLSS for denoising, which generally provides better performance.
This may result in renders of lower quality, which may be especially evident at lower resolutions.
Due to this, we recommend using per-tile or per-camera resolution of at least 100 x 100.
For renders at lower resolutions, we advice setting the ``antialiasing_mode`` attribute in
:class:`~isaaclab_physx.renderers.IsaacRtxRendererGlobalSettingsCfg` to
``DLAA``, and also potentially enabling ``enable_dl_denoiser``. Both of these settings should help improve render
quality, but also comes at a cost of performance. Additional rendering parameters can also be specified in
:class:`~isaaclab_physx.renderers.IsaacRtxRendererGlobalSettingsCfg`.


If you observe visual artifacts such as ghosting or disocclusion issues when using tiled rendering, you can try
adjusting the ``disocclusionScale`` parameter. This setting controls how aggressively the renderer handles
areas that become newly visible between frames:

.. code-block:: python

   global_settings = IsaacRtxRendererGlobalSettingsCfg(
      carb_settings={
         "/rtx/aovConverter/disocclusionScale": 10000,
      }
   )

.. note::

   This parameter is not commonly exposed as it may have side effects in certain scenarios.
   Only use it as a last resort if other quality settings do not resolve the visual artifacts.
   The value can be adjusted to a very high value to reduce disocclusion artifacts.


Rendering UsdVol 3D Gaussian Scenes in Multiple Environments
------------------------------------------------------------

When using UsdVol volumes with 3D Gaussian particles (e.g. exported from
`3DGRUT <https://github.com/nv-tlabs/3dgrut?tab=readme-ov-file#exporting-usdz-for-use-in-omniverse-and-isaac-sim>`_)
in **multiple environments**, you must set the following so the renderer uses the correct compositing path:

.. code-block:: python

   global_settings = IsaacRtxRendererGlobalSettingsCfg(
      carb_settings={
         "omni.rtx.nre.compositing.rendererHints": 3,
      }
   )

.. warning::

   With multiple environments, each environment holds its own copy of the scene, increasing device memory use,
   and environments are rendered one after another, which can substantially slow down rendering.
