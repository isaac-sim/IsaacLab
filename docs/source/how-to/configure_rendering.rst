Configuring Rendering Settings
==============================

Isaac Lab offers 3 preset rendering quality profiles: ``performance``, ``balanced``, and ``high``.
You can select a profile via a command line argument or from within a script, and customize settings as needed.
Adjust and fine-tune rendering to achieve the ideal balance for your workflow.

Selecting a Rendering Quality Profile
-------------------------------------

Rendering quality can be selected in 2 ways.

1. setting ``rendering_quality`` in a visualizer config, which selects an entry from
   :attr:`~sim.SimulationCfg.rendering_quality_cfgs`

   .. code-block:: python

     import isaaclab.sim as sim_utils
     from isaaclab.visualizers import KitVisualizerCfg

     sim_cfg = sim_utils.SimulationCfg(
         visualizer_cfgs=[
             KitVisualizerCfg(
                 rendering_quality="performance",
             ),
         ],
     )

2. using the ``--rendering_quality`` CLI argument, which takes precedence over
   ``visualizer_cfg.rendering_quality``

   .. code-block:: bash

     ./isaaclab.sh -p scripts/tutorials/00_sim/set_rendering_mode.py --rendering_quality {performance/balanced/high}


Notes:

* If ``rendering_quality=None`` for a visualizer, Isaac Lab does not apply rendering overrides
  for that visualizer, and backend/native defaults (for Kit, USD-authored settings) are used.
* ``--rendering_mode`` is deprecated. If used, it maps to ``--rendering_quality``
  (legacy ``quality`` maps to ``high``).


Example renders from the ``set_rendering_mode.py`` script.
To help assess rendering, the example scene includes some reflections, translucency, direct and ambient lighting, and several material types.

-  High Mode

   .. image:: ../_static/how-to/howto_rendering_example_quality.jpg
      :width: 100%
      :alt: High Rendering Mode Example

-  Balanced Mode

   .. image:: ../_static/how-to/howto_rendering_example_balanced.jpg
      :width: 100%
      :alt: Balanced Rendering Mode Example

-  Performance Mode

   .. image:: ../_static/how-to/howto_rendering_example_performance.jpg
      :width: 100%
      :alt: Performance Rendering Mode Example

Overwriting Specific Rendering Quality Settings
-----------------------------------------------

Preset rendering settings can be overwritten via :class:`~sim.RenderingQualityCfg`.

There are 2 ways to provide settings that overwrite presets.

1. :class:`~sim.RenderingQualityCfg` supports overwriting specific settings via explicit
   ``kit_*`` fields that map to underlying RTX settings.
   For example:

   .. code-block:: python

      quality_cfg = sim_utils.RenderingQualityCfg(
         rendering_mode_preset="performance",
         # explicit field overrides
         kit_enable_translucency=True,  # defaults to False in performance mode
         kit_enable_reflections=True,  # defaults to False in performance mode
         kit_dlss_mode=3,  # defaults to 0 in performance mode
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
      | kit_enable_global_illumination     | Bool. Enables Diffused Global Illumination at the cost of some          |
      |                                    | performance.                                                            |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_antialiasing_mode              | Literal["Off", "FXAA", "DLSS", "TAA", "DLAA"].                          |
      |                            |                                                                          |
      |                            | DLSS: Boosts performance by using AI to output higher resolution frames  |
      |                            | from a lower resolution input. DLSS samples multiple lower resolution    |
      |                            | images and uses motion data and feedback from prior frames to reconstruct|
      |                            | native quality images.                                                   |
      |                            | DLAA: Provides higher image quality with an AI-based anti-aliasing       |
      |                            | technique. DLAA uses the same Super Resolution technology developed for  |
      |                            | DLSS, reconstructing a native resolution image to maximize image quality.|
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_dlssg                   | Bool. Enables the use of DLSS-G. DLSS Frame Generation boosts           |
      |                                    | performance by using AI to generate more frames. This feature           |
      |                                    | requires an Ada Lovelace architecture GPU and can hurt performance      |
      |                                    | due to additional thread-related activities.                            |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_dl_denoiser             | Bool. Enables the use of a DL denoiser, which improves render quality   |
      |                                    | at the cost of performance.                                             |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_dlss_mode                      | Literal[0, 1, 2, 3]. For DLSS anti-aliasing, selects the                |
      |                                    | performance/quality tradeoff mode: 0 (Performance), 1 (Balanced),       |
      |                                    | 2 (Quality), 3 (Auto).                                                  |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_direct_lighting         | Bool. Enable direct light contributions from lights.                    |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_samples_per_pixel              | Int. Defines direct lighting samples per pixel. Higher values increase  |
      |                                    | quality at the cost of performance.                                     |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_shadows                 | Bool. Enables shadows at the cost of performance. When disabled, lights |
      |                                    | will not cast shadows.                                                  |
      +------------------------------------+-------------------------------------------------------------------------+
      | kit_enable_ambient_occlusion       | Bool. Enables ambient occlusion at the cost of some performance.        |
      +------------------------------------+-------------------------------------------------------------------------+


2. Customize or add named profiles in :attr:`~sim.SimulationCfg.rendering_quality_cfgs`,
   then select them from per-visualizer ``rendering_quality`` fields.


Current Limitations
-------------------

For performance reasons, we default to using DLSS for denoising, which generally provides better performance.
This may result in renders of lower quality, which may be especially evident at lower resolutions.
Due to this, we recommend using per-tile or per-camera resolution of at least 100 x 100.
For renders at lower resolutions, we advise setting ``kit_antialiasing_mode="DLAA"``
in :class:`~sim.RenderingQualityCfg`, and also potentially enabling
``kit_enable_dl_denoiser=True``. Both of these settings can help improve render quality,
but come at a cost of performance.
