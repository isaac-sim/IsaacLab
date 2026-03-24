Configuring RTX Rendering Settings
====================================

.. note::

   This guide covers the **RTX renderer** settings, which are used when running Isaac Lab with
   Isaac Sim. The RTX renderer is based on NVIDIA's Omniverse RTX rendering pipeline and is
   available for all camera sensors in the PhysX backend.

   For the **Newton renderer** (used with the Newton backend or in kit-less mode), see
   :ref:`overview_renderers` for the pluggable renderer architecture and available backends.

Isaac Lab's RTX renderer offers 3 preset rendering modes: performance, balanced, and quality.
You can select a mode via a command line argument or from within a script, and customize settings as needed.
Adjust and fine-tune rendering to achieve the ideal balance for your workflow.

Selecting a Rendering Mode Profile
----------------------------------

Rendering mode can be selected in two ways:

1. Set the visualizer profile selector field ``rendering_mode``, which selects an entry from
   :attr:`~sim.SimulationCfg.rendering_mode_cfgs`.

   .. code-block:: python

      import isaaclab.sim as sim_utils
      from isaaclab_physx.visualizers import KitVisualizerCfg

      sim_cfg = sim_utils.SimulationCfg(
          visualizer_cfgs=[
              KitVisualizerCfg(
                  rendering_mode="performance",
              ),
          ],
      )

2. Use the ``--rendering_mode`` CLI argument, which takes precedence over
   ``visualizer_cfg.rendering_mode``.

   .. code-block:: bash

      ./isaaclab.sh -p scripts/tutorials/00_sim/set_rendering_mode.py --rendering_mode {performance/balanced/quality}

Notes:

* If ``rendering_mode=None`` for a visualizer, Isaac Lab does not apply rendering overrides
  for that visualizer, and backend/native defaults (for Kit, USD-authored settings) are used.
* ``--rendering_mode`` is the supported CLI entry point.

Example renders from the ``set_rendering_mode.py`` script.
To help assess rendering, the example scene includes reflections, translucency,
direct and ambient lighting, and several material types.

-  Quality Mode

   .. image:: ../_static/how-to/howto_rendering_example_quality.jpg
      :width: 100%
      :alt: Quality Rendering Mode Example

-  Balanced Mode

   .. image:: ../_static/how-to/howto_rendering_example_balanced.jpg
      :width: 100%
      :alt: Balanced Rendering Mode Example

-  Performance Mode

   .. image:: ../_static/how-to/howto_rendering_example_performance.jpg
      :width: 100%
      :alt: Performance Rendering Mode Example

Overriding Rendering Mode Settings
----------------------------------

Preset rendering settings can be overwritten via
:class:`~isaaclab.rendering_mode.RenderingModeCfg`.

There are two ways to provide settings that overwrite presets:

1. :class:`~isaaclab.rendering_mode.RenderingModeCfg` supports overwriting specific settings via explicit
   ``kit_*`` fields that map to underlying RTX settings.

   .. code-block:: python

      import isaaclab.sim as sim_utils
      from isaaclab.rendering_mode import RenderingModeCfg

      mode_cfg = RenderingModeCfg(
          rendering_mode_preset="performance",
          # explicit field overrides
          kit_enable_translucency=True,  # defaults to False in performance mode
          kit_enable_reflections=True,   # defaults to False in performance mode
          kit_dlss_mode=3,               # defaults to 0 in performance mode
      )

   List of Kit settings.

   .. table::
      :widths: 25 75

      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_translucency            | Bool. Enables translucency for specular transmissive surfaces such as   |
      |                                    | glass at the cost of some performance.                                  |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_reflections             | Bool. Enables reflections at the cost of some performance.              |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_global_illumination     | Bool. Enables Diffuse Global Illumination at the cost of some           |
      |                                    | performance.                                                            |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_antialiasing_mode              | Literal["Off", "FXAA", "DLSS", "TAA", "DLAA"].                          |
      |                                    | DLSS boosts performance by reconstructing higher-resolution frames.      |
      |                                    | DLAA prioritizes image quality using the same SR technology as DLSS.    |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_dlssg                   | Bool. Enables DLSS-G frame generation (Ada Lovelace GPU required).      |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_dl_denoiser             | Bool. Enables DL denoiser (quality up, performance down).               |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_dlss_mode                      | Literal[0, 1, 2, 3] = Performance, Balanced, Quality, Auto.             |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_direct_lighting         | Bool. Enables direct light contributions from lights.                   |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_samples_per_pixel              | Int. Direct lighting samples-per-pixel (higher = better, slower).       |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_shadows                 | Bool. Enables shadows at performance cost.                              |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_ambient_occlusion       | Bool. Enables ambient occlusion at performance cost.                    |
      +------------------------------------+-------------------------------------------------------------------------+


2. If you need a custom profile, define your own named entry in
   :attr:`~sim.SimulationCfg.rendering_mode_cfgs` and select it from each visualizer.

   .. code-block:: python

      import isaaclab.sim as sim_utils
      from isaaclab.rendering_mode import RenderingModeCfg
      from isaaclab_physx.visualizers import KitVisualizerCfg

      sim_cfg = sim_utils.SimulationCfg(
          rendering_mode_cfgs={
              "my_profile": RenderingModeCfg(
                  rendering_mode_preset="balanced",
                  kit_enable_reflections=True,
                  kit_dlss_mode=2,
              ),
          },
          visualizer_cfgs=[KitVisualizerCfg(rendering_mode="my_profile")],
      )

Current Limitations
-------------------

For performance reasons, we default to using DLSS for denoising, which generally provides better performance.
This may result in renders of lower quality, which may be especially evident at lower resolutions.
Due to this, we recommend using per-tile or per-camera resolution of at least 100 x 100.
For renders at lower resolutions, we advise setting
``kit_antialiasing_mode="DLAA"`` in
:class:`~isaaclab.rendering_mode.RenderingModeCfg`, and potentially enabling
``kit_enable_dl_denoiser=True``. Both settings can improve quality at a cost
of performance.


If you observe visual artifacts such as ghosting or disocclusion issues when using tiled rendering, you can try
adjusting the ``disocclusionScale`` parameter. This setting controls how aggressively the renderer handles
areas that become newly visible between frames:

.. note::

   Low-level carb rendering settings (for example,
   ``/rtx/aovConverter/disocclusionScale``) are not currently exposed through
   :class:`~isaaclab.rendering_mode.RenderingModeCfg`.

.. note::

   This parameter is not commonly exposed as it may have side effects in certain scenarios.
   Only use it as a last resort if other quality settings do not resolve the visual artifacts.
   The value can be adjusted to a very high value to reduce disocclusion artifacts.


Rendering UsdVol 3D Gaussian Scenes in Multiple Environments
------------------------------------------------------------

When using UsdVol volumes with 3D Gaussian particles (e.g. exported from
`3DGRUT <https://github.com/nv-tlabs/3dgrut?tab=readme-ov-file#exporting-usdz-for-use-in-omniverse-and-isaac-sim>`_)
in **multiple environments**, you must set the following so the renderer uses the correct compositing path:

.. note::

   This setting is not currently exposed through
   :class:`~isaaclab.rendering_mode.RenderingModeCfg`.

.. warning::

   With multiple environments, each environment holds its own copy of the scene, increasing device memory use,
   and environments are rendered one after another, which can substantially slow down rendering.
