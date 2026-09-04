Visualization
=============

.. currentmodule:: isaaclab

Isaac Lab provides 5 visualizers for real-time simulation inspection and debugging. Where
renderers produce sensor data for training, visualizers give fast, lightweight feedback for
human monitoring and recording. Visualizers can also stream that same sensor data through the
streaming camera panel.

This page covers:

- `Quick Start`_: launch a visualizer with ``--viz``
- `Visualizer Overview`_: each visualizer in its own section
- `Shared Features`_: what's common across visualizers
- `Usage`_: common CLI and config recipes
- `Limitations`_: per-visualizer differences and troubleshooting

.. raw:: html

   <style>
   .viz-grid { display:flex; gap:7px; margin: 0.5em 0; }
   .viz-hero-stack { display:flex; flex-direction:column; gap:7px; margin: 0.5em 0 1em; }
   .viz-hero-stack .viz-grid { margin:0; }
   .viz-grid > div { flex:1 1 0; min-width:0; }
   .viz-grid video, .viz-grid img { display:block; width:100%; border-radius:0 !important; padding:0 !important; background:none !important; }
   .viz-grid-stretch video, .viz-grid-stretch img { height:260px; object-fit:cover; }
   .viz-grid-stretch.viz-grid-hero-tiles video { height:286px; }
   .viz-grid-fit { justify-content:center; }
   .viz-grid-fit > div { flex:0 0 auto; }
   .viz-grid-fit video, .viz-grid-fit img { width:auto; height:488px; }
   .viz-grid-stretch video.viz-no-crop { object-fit:contain; background:#000; }
   .viz-grid-stretch video.viz-crop-bottom { object-position:center bottom; }
   .viz-grid-stretch video.viz-crop-kit-bottom { object-position:center 76%; }
   .viz-grid-stretch video.viz-crop-x8 { width:calc(100% + 16px); margin-left:-8px; }
   .viz-grid-stretch img.viz-crop-top { object-position:center 25%; }
   .viz-hero-wrap.viz-hero-newton-gl { aspect-ratio:960/397; }
   .viz-hero-wrap.viz-hero-newton-gl video.viz-crop-newton-hero { width:calc(100% + 2px); height:calc(100% + 70px);
                margin-left:-1px; margin-top:-55px; object-fit:cover; object-position:center 55.3%; }
   .viz-label.viz-label-raise { bottom:10px; }
   /* Trims pixels off each hero tile on top of whatever object-position crop is already
      applied, so the tile itself is shorter rather than just repositioning the existing crop.
      Both classes initially crop to the same 221px height, keeping tiles within each row even.
      Kit/Rerun/Newton RTX crop 45px off the top and 20px off the bottom; Viser crops 35px off
      the top and 30px off the bottom (its object-position framing already leaves more headroom
      at the bottom, so it can take a heavier bottom crop). */
   .viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap { height:276px; }
   .viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap video { margin-top:-10px; }
   .viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap.viz-crop-top25 { height:221px; }
   .viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap.viz-crop-top25 video { margin-top:-45px; }
   .viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap.viz-crop-mixed { height:221px; }
   .viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap.viz-crop-mixed video { margin-top:-35px; }
   /* The top row (Viser, Newton RTX) overrides both tiles above to a shorter, still-even 206px:
      15px more off the top than the bottom row, with bottom crop unchanged (height shrinks by
      15px and margin-top grows by 15px in lockstep). The bottom row (Rerun, Kit) keeps the 221px
      height set above. */
   .viz-hero-row-top.viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap.viz-crop-top25 { height:206px; }
   .viz-hero-row-top.viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap.viz-crop-top25 video { margin-top:-60px; }
   .viz-hero-row-top.viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap.viz-crop-mixed { height:206px; }
   .viz-hero-row-top.viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap.viz-crop-mixed video { margin-top:-50px; }
   .viz-grid-record { justify-content:center; align-items:flex-start; }
   .viz-grid-record > div { flex:0 0 auto; }
   .viz-grid-record video { display:block; width:auto; height:300px; }
   .viz-stack-centered { display:flex; flex-direction:column; align-items:center; gap:1em; margin: 0.5em 0; }
   .viz-stack-centered > div { width:65%; }
   .viz-stack-centered video { display:block; width:100%; height:auto; }
   .viz-cap { text-align:center; font-style:italic; margin-top:0.4em; font-size:0.9em; }
   .viz-hero-wrap { position:relative; overflow:hidden; }
   .viz-label { position:absolute; bottom:8px; right:8px; max-width:35%; background:rgba(32,32,32,0.85);
                color:#fff; padding:3px 10px; border-radius:3px; font-size:14px; font-weight:600;
                line-height:1.2; white-space:nowrap; }
   </style>

   <p class="viz-cap">5 visualizers running Isaac-Velocity-Flat-AnymalD with a circular velocity command<br>
   Green arrow: commanded velocity. Blue arrow: current velocity.</p>

   <div class="viz-hero-stack">
   <div class="viz-hero-wrap viz-hero-newton-gl">
     <video autoplay loop muted playsinline controls preload="auto" class="viz-crop-newton-hero">
       <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/hero_newton_gl.mp4" type="video/mp4">
     </video>
     <div class="viz-label viz-label-raise">Newton GL</div>
   </div>

   <div class="viz-grid viz-grid-stretch viz-grid-hero-tiles viz-hero-row-top">
     <div class="viz-hero-wrap viz-crop-mixed">
       <video autoplay loop muted playsinline controls preload="auto" class="viz-crop-bottom">
         <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/hero_viser.mp4" type="video/mp4">
       </video>
       <div class="viz-label">Viser</div>
     </div>
     <div class="viz-hero-wrap viz-crop-top25">
       <video autoplay loop muted playsinline controls preload="auto" class="viz-crop-bottom">
         <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/hero_newton_rtx.mp4" type="video/mp4">
       </video>
       <div class="viz-label">Newton RTX</div>
     </div>
   </div>
   <div class="viz-grid viz-grid-stretch viz-grid-hero-tiles">
     <div class="viz-hero-wrap viz-crop-top25">
       <video autoplay loop muted playsinline controls preload="auto" class="viz-crop-bottom">
         <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/hero_rerun.mp4" type="video/mp4">
       </video>
       <div class="viz-label">Rerun</div>
     </div>
     <div class="viz-hero-wrap viz-crop-top25">
       <video autoplay loop muted playsinline controls preload="auto" class="viz-crop-kit-bottom viz-crop-x8">
         <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/hero_kit.mp4" type="video/mp4">
       </video>
       <div class="viz-label">Kit</div>
     </div>
   </div>
   </div>

   <p class="viz-cap">Note: Newton RTX has no velocity arrows, since it doesn't yet support visualization markers.</p>


Quick Start
-----------

Pass ``--viz`` to any train command to launch a visualizer. ``--visualizer`` is an equivalent
alias.

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          # Newton GL: lightweight OpenGL viewport
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --viz newton_gl

          # Viser: browser-based viewer
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --viz viser

          # Kit viewport
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --viz kit

          # Multiple visualizers simultaneously (comma-separated, no spaces)
          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --viz rerun,newton_rtx

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole --viz newton_gl
          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole --viz viser
          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole --viz kit
          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole --viz rerun,newton_rtx

.. note::

   Most tasks default to a PhysX backend, which requires Isaac Sim. If it isn't installed yet,
   add ``--extra isaacsim`` to the ``uv run`` commands above; see
   :ref:`installation-optional-extras` for details.

For combining visualizers, running headless, and other common use cases, see `Usage`_ below.


Visualizer Overview
-------------------

.. list-table::
   :widths: 20 80
   :width: 100%
   :header-rows: 1

   * - Visualizer
     - Description
   * - **Newton GL**
     - Lightweight, strong feature support; Recommended
   * - **Viser**
     - Web-based, supports recording and replay; Recommended
   * - **Newton RTX**
     - High-quality RTX rendering; experimental, missing some features
   * - **Rerun**
     - Web-based, supports recording and replay; limited UI toggles
   * - **Kit**
     - High-quality RTX rendering, rich Isaac Sim tooling; longer start-up time

.. tab-set::

   .. tab-item:: Newton GL

      The Newton GL visualizer is a lightweight OpenGL window with minimal startup overhead.

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="auto" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/showcase_newton_gl_allegro.mp4" type="video/mp4">
         </video>
         <p class="viz-cap">Isaac-Reorient-Cube-Allegro in Newton GL Visualizer<br>Each Allegro hand reorients
         its cube to match the target shown by the marker above it</p>

      .. raw:: html

         <div class="viz-grid">
           <div>

      **Visualizer-specific features:**

      - **Pause Rendering**, **Pause Simulation**, and **Reset Episode** ImGui controls
      - Rigid-body dragging via right-click, see right (Newton solvers only)
      - Adjustable render update frequency

      .. raw:: html

           </div>
           <div>
             <video autoplay loop muted playsinline controls preload="auto" style="width:100%;">
               <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/showcase_newton_gl_dominoes.mp4" type="video/mp4">
             </video>
             <p class="viz-cap">newton_viewer_dominoes demo<br>Right-click dragging the first domino
             triggers the cascade across an NVIDIA-logo domino layout</p>
           </div>
         </div>

      **Core configuration:**

      .. code-block:: python

          from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

          visualizer_cfg = NewtonGLVisualizerCfg(
              eye=(8.0, 8.0, 3.0),
              lookat=(0.0, 0.0, 0.0),
              window_width=1920,
              window_height=1080,
              show_joints=False,
              show_contacts=False,
              enable_live_plots=True,
          )

      For the full config reference, see the config classes below.

      .. dropdown:: NewtonGLVisualizerCfg source
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualizer_cfg.py
            :language: python
            :pyobject: NewtonGLVisualizerCfg

      .. dropdown:: NewtonVisualizerCfg source (shared Newton base class)
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualizer_cfg.py
            :language: python
            :pyobject: NewtonVisualizerCfg

      .. dropdown:: VisualizerCfg source (shared base class)
         :icon: code

         .. literalinclude:: ../../../source/isaaclab/isaaclab/visualizers/visualizer_cfg.py
            :language: python
            :pyobject: VisualizerCfg

   .. tab-item:: Viser

      `Viser <https://viser.studio/>`_ streams the Newton Warp renderer to a local web server,
      at ``http://localhost:8080`` by default, with optional public share URLs via ``share=True`` and
      ``.viser`` recording files for replay.

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="auto" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/showcase_viser_lift_kuka_allegro.mp4" type="video/mp4">
         </video>
         <p class="viz-cap">Isaac-Lift-KukaAllegro in Viser Visualizer<br>Each Kuka arm and Allegro hand lifts
         its cube off the table<br>Table color shows success: red until the cube reaches its target pose, green
         once it does.</p>

      **Visualizer-specific features:**

      - Public share URL for remote access, set ``share=True``
      - **Pause Rendering**, **Pause Simulation**, and **Reset Episode** sidebar controls

      .. important::

         A URL is printed before training begins. Set ``open_browser=True`` to open it
         automatically. For remote access, set ``display_address`` to the machine hostname or IP
         and ensure the configured ``port`` is reachable from the browser.

         .. code-block:: text

            ╭────── viser (listening *:8080) ───────╮
            │   URL  │  http://localhost:8080        │
            ╰───────────────────────────────────────╯

      **Core configuration:**

      .. code-block:: python

          from isaaclab_visualizers.viser import ViserVisualizerCfg

          visualizer_cfg = ViserVisualizerCfg(
              port=8080,
              bind_address="0.0.0.0",       # use 0.0.0.0 for remote access
              display_address="localhost",   # hostname shown in the printed URL
              open_browser=False,
              share=False,                   # request a public share URL
              record_to_viser=None,          # set a path to save a .viser recording
          )

      For the full config reference, see the config classes below.

      .. dropdown:: ViserVisualizerCfg source
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/viser/viser_visualizer_cfg.py
            :language: python
            :pyobject: ViserVisualizerCfg

      .. dropdown:: VisualizerCfg source (shared base class)
         :icon: code

         .. literalinclude:: ../../../source/isaaclab/isaaclab/visualizers/visualizer_cfg.py
            :language: python
            :pyobject: VisualizerCfg

   .. tab-item:: Newton RTX

      The **experimental** Newton RTX visualizer, launched with ``--viz newton_rtx``, uses the
      OVRTX path-tracer for photorealistic rendering without a Kit process. It is slightly
      slower than the Newton GL visualizer due to the cost of high-quality RTX rendering.

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="auto" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/showcase_newton_rtx_h1_stairs.mp4" type="video/mp4">
         </video>
         <p class="viz-cap">Isaac-Velocity-Rough-H1 in Newton RTX Visualizer<br>Each H1 humanoid climbs a
         staircase sub-terrain</p>

      **Visualizer-specific features:**

      - OVRTX path-traced rendering for photorealistic, physically-based lighting

      .. note::

         The following features are not yet supported and will be added in a future release:
         visualization markers, live plots, and pause rendering. All are available in the other
         visualizers. The streaming camera panel's live on-screen preview is also unavailable, but
         headless streaming capture (e.g. for :class:`~isaaclab.envs.VideoRecorderCfg`) works.

      **Core configuration:**

      .. code-block:: python

          from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

          visualizer_cfg = NewtonRTXVisualizerCfg(
              eye=(8.0, 8.0, 3.0),
              lookat=(0.0, 0.0, 0.0),
          )

      For the full config reference, see the config classes below.

      .. dropdown:: NewtonRTXVisualizerCfg source
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualizer_cfg.py
            :language: python
            :pyobject: NewtonRTXVisualizerCfg

      .. dropdown:: NewtonVisualizerCfg source (shared Newton base class)
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualizer_cfg.py
            :language: python
            :pyobject: NewtonVisualizerCfg

      .. dropdown:: VisualizerCfg source (shared base class)
         :icon: code

         .. literalinclude:: ../../../source/isaaclab/isaaclab/visualizers/visualizer_cfg.py
            :language: python
            :pyobject: VisualizerCfg

      .. warning::

         Newton RTX (OVRTX) is a kitless renderer and cannot be used in the same process as the Kit
         visualizer or PhysX. This rules out ``presets=ovphysx`` and ``presets=isaacsim_physx``; use
         ``presets=newton_mjwarp,ovrtx`` with ``--viz newton_rtx``, or switch to ``--viz newton_gl``,
         ``--viz viser``, ``--viz rerun``, or ``--viz kit`` with a Kit-compatible physics backend.

   .. tab-item:: Rerun

      Like Viser, `Rerun <https://rerun.io/>`_ streams simulation state to a local web server, for
      remote monitoring, timeline playback, and recording to ``.rrd`` files.

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="auto" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/showcase_rerun_shadow_reorient.mp4" type="video/mp4">
         </video>
         <p class="viz-cap">Isaac-Reorient-Cube-Shadow-Direct in Rerun Visualizer<br>Each Shadow Hand reorients
         its cube to match the target pose</p>

      **Visualizer-specific features:**

      - Timeline scrubbing and playback of ``.rrd`` recordings
      - **Pause Rendering** and **Reset Episode** controls via the ImGui sidebar

      .. note::

         The native Play/Pause timeline controls in the Rerun visualizer UI do not work while
         visualizing a live simulation or training run. They are hidden by default, but Rerun's
         dock panel UI can still be used to reveal them; when revealed, clicking them has no
         effect. Use Isaac Lab's own **Pause Rendering** / **Reset Episode** controls instead.
         The timeline controls are only meaningful when replaying a saved ``.rrd`` recording.

      .. important::

         A highlighted URL is printed to the terminal before training begins. Ctrl-click it to
         open the viewer, or set ``open_browser=True`` to open it automatically.

         .. code-block:: text

            ╭─────────────────────────── rerun (listening *:9090) ───────────────────────────╮
            │   URL  │  http://127.0.0.1:9090/?url=rerun%2Bhttp://127.0.0.1:9876/proxy       │
            ╰────────────────────────────────────────────────────────────────────────────────╯

      **Core configuration:**

      .. code-block:: python

          from isaaclab_visualizers.rerun import RerunVisualizerCfg

          visualizer_cfg = RerunVisualizerCfg(
              eye=(8.0, 8.0, 3.0),
              lookat=(0.0, 0.0, 0.0),
              keep_historical_data=False,   # enable for time scrubbing
              keep_scalar_history=False,    # enable for scalar time-series
              record_to_rrd=None,           # set a path to save a .rrd recording
              open_browser=False,
          )

      For the full config reference, see the config classes below.

      .. dropdown:: RerunVisualizerCfg source
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/rerun/rerun_visualizer_cfg.py
            :language: python
            :pyobject: RerunVisualizerCfg

      .. dropdown:: VisualizerCfg source (shared base class)
         :icon: code

         .. literalinclude:: ../../../source/isaaclab/isaaclab/visualizers/visualizer_cfg.py
            :language: python
            :pyobject: VisualizerCfg

   .. tab-item:: Kit

      The Kit visualizer embeds Isaac Lab inside a Kit process, providing access to the
      full Isaac Sim USD stage, RTX renderer, and GUI tooling. It has the highest startup and
      runtime overhead of the five visualizers.

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="auto" style="width:100%;">
           <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/showcase_kit_franka_reach.mp4" type="video/mp4">
         </video>
         <p class="viz-cap">Isaac-Reach-Franka in Kit Visualizer<br>Each Franka arm reaches its end-effector
         toward a randomized target pose</p>

      **Visualizer-specific features:**

      - Direct access to the Isaac Sim USD stage for inspecting and editing prims at runtime
      - Full Isaac Sim GUI tooling (Property, Layers, and Stage panels)

      **Core configuration:**

      .. code-block:: python

          from isaaclab_visualizers.kit import KitVisualizerCfg

          visualizer_cfg = KitVisualizerCfg(
              eye=(8.0, 8.0, 3.0),
              lookat=(0.0, 0.0, 0.0),
              window_width=1280,
              window_height=720,
              enable_markers=True,
              enable_live_plots=True,
          )

      For the full config reference, see the config classes below.

      .. dropdown:: KitVisualizerCfg source
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer_cfg.py
            :language: python
            :pyobject: KitVisualizerCfg

      .. dropdown:: VisualizerCfg source (shared base class)
         :icon: code

         .. literalinclude:: ../../../source/isaaclab/isaaclab/visualizers/visualizer_cfg.py
            :language: python
            :pyobject: VisualizerCfg

Shared Features
---------------

Streaming Camera View
~~~~~~~~~~~~~~~~~~~~~

The streaming camera view composites per-environment sensor data into a tiled panel that
updates every step.

.. raw:: html

   <div class="viz-stack-centered">
     <div>
       <video autoplay loop muted playsinline controls preload="auto">
         <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/streaming_newton_galbot_interactive.mp4" type="video/mp4">
       </video>
       <p class="viz-cap">Interactive view of the Galbot cube stacking environment</p>
     </div>
     <div>
       <video autoplay loop muted playsinline controls preload="auto">
         <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/streaming_newton_galbot_tiled.mp4" type="video/mp4">
       </video>
       <p class="viz-cap">Streaming view showing tiled Galbot wrist-camera feeds</p>
     </div>
   </div>

The streaming panel supports RGB, depth, segmentation, and surface normals, with a configurable
number of environments shown.

Streams can come from auto-created cameras that track and follow robot bodies, or from existing
scene camera sensors, letting you toggle between different views, such as the Galbot task's
wrist-mounted and ego cameras. Supported on Kit, Newton GL, Rerun, and Viser; not yet
supported on Newton RTX (experimental).

See :doc:`/source/features/visualizer_tiled_camera` for the full guide and tutorial.


Visualization Markers
~~~~~~~~~~~~~~~~~~~~~

Visualization markers draw debug geometry over the scene via
:class:`~isaaclab.markers.VisualizationMarkers`.

.. raw:: html

   <div class="viz-grid viz-grid-stretch">
     <div>
       <img src="../../_static/markers_anymal_d.jpg" alt="Velocity arrow marker on an AnymalD robot">
       <p class="viz-cap">Large green/blue arrow markers showing target and base velocity for an AnymalD robot</p>
     </div>
     <div>
       <img src="../../_static/markers_franka.jpg" alt="Joint arrow markers on a Franka arm and contact sensor markers on a cube" class="viz-crop-top">
       <p class="viz-cap">Arrow markers on the Franka arm's joints, with contact sensor markers on the cube</p>
     </div>
   </div>

See :doc:`/source/features/draw_markers` for creating and configuring custom markers.

There are 2 types of visualization markers:

**Built-in marker types:**

- Sensor markers: contact hit/no-contact spheres, ray-caster hit points, deformable-object
  kinematic targets, visuo-tactile contact points
- Frame markers: coordinate frame with a connecting line
- Colored arrows (red, blue, green) for direction and velocity indicators
- Goal shapes: cuboid, sphere, and multi-state position-goal markers
- Center of mass markers

In the Newton GL visualizer's HUD, built-in marker types that aren't currently active are
shown greyed out.

**Custom Markers:**

- Defined by individual tasks and MDP terms that instantiate their own
  :class:`~isaaclab.markers.VisualizationMarkersCfg`
- Example: the AnymalD velocity command draws its green/blue velocity arrows this way
- Example: the Isaac-Reorient-Cube-Allegro task defines its own goal marker (a DexCube asset)
  to show the target cube orientation

.. note::

   Visualization markers are supported on Kit, Newton GL, Rerun, and Viser; not yet
   supported on Newton RTX (experimental).


Live Plots
~~~~~~~~~~

Live plots stream per-step scalar data into the visualizer.

.. raw:: html

   <div class="viz-grid viz-grid-fit">
     <div>
       <video autoplay loop muted playsinline controls preload="auto">
         <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/liveplot_newton_gl.mp4" type="video/mp4">
       </video>
       <p class="viz-cap">Newton GL visualizer, with 3 plots opened</p>
     </div>
     <div>
       <video autoplay loop muted playsinline controls preload="auto">
         <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/liveplot_viser.mp4" type="video/mp4">
       </video>
       <p class="viz-cap">Viser visualizer, with 1 plot opened</p>
     </div>
   </div>

Live Plot windows start hidden or collapsed, except for Rerun, where the Live Plots can be
collapsed by selecting the "eye" icon. Supported on Kit, Newton GL, Rerun, and Viser; not
yet supported on Newton RTX (experimental).

**What's plotted:** per-episode total reward and episode length, plus any active manager
term for manager-based environments (e.g. ``joint_pos``).

Video Recording
~~~~~~~~~~~~~~~~

Video Recording supports capturing feed from any visualizer view, both interactive and
streaming views, and renderer-based sensor data capture.

.. raw:: html

   <div class="viz-grid viz-grid-record">
     <div>
       <video autoplay loop muted playsinline controls preload="auto">
         <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_tiled_kit_viewport.mp4" type="video/mp4">
       </video>
       <p class="viz-cap">Kit visualizer tiled streaming</p>
     </div>
     <div>
       <video autoplay loop muted playsinline controls preload="auto">
         <source src="https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/record_video_shadow_hand_tiled_gt.mp4" type="video/mp4">
       </video>
       <p class="viz-cap">Recorded clip with ground-truth sensor views</p>
     </div>
   </div>

Pass ``--video`` on the command line for a quick recording from the default visualizer, or
define multiple ``VideoRecorderCfg`` entries to record multiple sources at once. Kit, Newton
GL, and Newton RTX visualizers can be recorded while in headless mode, to reduce overhead or
in case no display is available.

Not currently supported by the web-based visualizers Viser and Rerun; add a headless
visualizer as a capture source alongside them to record video.

See :doc:`/source/features/record_video` for the full guide and tutorial.


.. _visualization-configuration:

Usage
-----

Common Recipes
~~~~~~~~~~~~~~

**Headless training with video recording**

Run without a window and record clips from a Newton GL or Kit visualizer kept alive as the
capture source:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole \
              --viz newton_gl --headless --video

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole \
              --viz newton_gl --headless --video

See :doc:`/source/features/record_video` for clip length, interval, and multi-source options.

**Combining an interactive view with a headless recording source**

Watch training live in Kit while recording from a separate headless Newton GL angle:

.. code-block:: python

    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    sim_cfg.visualizer_cfgs = [
        KitVisualizerCfg(eye=(4.0, 4.0, 2.0)),
        NewtonGLVisualizerCfg(eye=(12.0, 0.0, 6.0), headless=True),
    ]

See the "Recording from an independent camera angle" section of
:doc:`/source/features/record_video` for the full example.

**Following a moving robot (Kit)**

Lock the Kit camera to a moving asset instead of updating ``eye``/``lookat`` yourself every
step:

.. code-block:: python

    from isaaclab_visualizers.kit import KitVisualizerCfg

    sim_cfg.visualizer_cfgs = [
        KitVisualizerCfg(
            origin_type="asset",
            origin_track_path="robot",  # or "robot/panda_hand" to track a specific body
            eye=(4.0, 4.0, 2.0),  # offset from the tracked asset
        )
    ]

**Sharing a live view with a remote teammate**

Viser can request a public share URL for the running session, useful for remote pairing
without screen-sharing:

.. code-block:: python

    from isaaclab_visualizers.viser import ViserVisualizerCfg

    sim_cfg.visualizer_cfgs = [ViserVisualizerCfg(share=True, open_browser=True)]

The share URL is logged on startup. Rerun has no equivalent config field, but exposes its own
share button in the native UI.

Resolution Rules
~~~~~~~~~~~~~~~~

Visualizers are resolved from ``--viz`` (comma-separated, e.g. ``--viz kit,newton_gl``) or
``SimulationCfg.visualizer_cfgs`` in code. If ``--viz`` is omitted, the config value is used;
``--viz none`` always disables all visualizers, regardless of config.

Add ``--headless`` alongside ``--viz kit`` or ``--viz newton_gl`` to keep that visualizer
running without an on-screen window, e.g. as a ``--video`` recording source on a machine
without a display.

To configure visualizer settings in code, pass ``VisualizerCfg`` instances to
``SimulationCfg``:

.. code-block:: python

    from isaaclab.sim import SimulationCfg
    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    sim_cfg = SimulationCfg(
        visualizer_cfgs=[
            KitVisualizerCfg(eye=(0.0, 0.0, 20.0)),
            NewtonGLVisualizerCfg(eye=(5.0, 5.0, 5.0), show_joints=True),
        ]
    )

.. _visualization-common-modes:

.. list-table:: Common modes
   :header-rows: 1
   :widths: 30 35 35

   * - CLI args
     - ``visualizer_cfgs``
     - Effective behavior
   * - no ``--viz``
     - ``[]``
     - No visualizer launches; no window, no capture source.
   * - ``--viz kit,newton_gl``
     - ``[]``
     - Launch default Kit and Newton GL visualizers.
   * - ``--viz newton_gl --headless``
     - ``[]``
     - Launch Newton GL without a window, e.g. as a ``--video`` recording source.
   * - ``--viz kit,newton_gl``
     - ``[NewtonGLVisualizerCfg(...), RerunVisualizerCfg(...)]``
     - Launch default Kit and custom Newton GL; Rerun is not launched.
   * - no ``--viz``
     - ``[NewtonGLVisualizerCfg(...), RerunVisualizerCfg(...)]``
     - Launch custom Newton GL and Rerun from config.
   * - ``--viz none``
     - ``[NewtonGLVisualizerCfg(...), RerunVisualizerCfg(...)]``
     - All visualizers disabled; no window, no capture source.

For migration context, see :doc:`/source/migration/migrating_to_isaaclab_3-0`.


Performance
~~~~~~~~~~~

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - Visualizer
     - Tips
   * - Newton GL
     - Lowest overhead; increase ``update_frequency`` to skip render calls.
   * - Viser
     - Newton Warp renderer; use ``max_visible_envs`` to limit the number of rendered
       environments.
   * - Newton RTX
     - Path-traced; highest per-frame cost, use ``--max_visible_envs`` to reduce load.
   * - Rerun
     - Web viewer may slow down with many environments; use ``--num_envs`` to reduce load.
   * - Kit
     - Highest overhead of the five visualizers; reduce ``window_width`` / ``window_height``
       or use ``--max_visible_envs``.


Limitations
-----------

**Backend feature support:**

.. list-table::
   :widths: 34 11 11 12 11 11
   :header-rows: 1

   * - Feature
     - Newton GL
     - Newton RTX
     - Viser
     - Rerun
     - Kit
   * - Visualization markers
     - ✓
     - ✗
     - ✓
     - ✓
     - ✓
   * - Live plots
     - ✓
     - ✗
     - ✓
     - ✓
     - ✓
   * - Streaming camera panel
     - ✓
     - ✗
     - ✓
     - ✓
     - ✓
   * - Video recording, ``--video``
     - ✓
     - ✓
     - ✗
     - ✗
     - ✓
   * - Pause rendering
     - ✓
     - ✗
     - ✓
     - ✓
     - ✓
   * - Headless mode
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓

**Kit: incompatible with ovphysx / ovrtx presets**

``--viz kit`` cannot be used with ``presets=ovphysx`` or ``presets=ovrtx`` in the same process.
Use ``--viz newton_gl``, ``--viz rerun``, or ``--viz viser`` with those presets, or omit
``--viz`` for headless execution.

**Rerun: large environment performance**

The Rerun web viewer may slow down or crash with many environments. Reduce load with
``--num_envs``:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --viz rerun --num_envs 512

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

          ./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole --viz rerun --num_envs 512

**Newton GL: CUDA/OpenGL interoperability warnings**

In some configurations, the Newton GL visualizer emits warnings about CUDA/OpenGL
interoperability:

.. code-block:: text

    Warning: Could not get MSAA config, falling back to non-AA.
    Warp CUDA error 999: unknown error (in function wp_cuda_graphics_register_gl_buffer)
    Warp UserWarning: Could not register GL buffer …

The visualizer still functions correctly but falls back to CPU copy operations, which reduces
performance.

**Newton GL: OpenGL context failures**

If pyglet reports that ``glCreateShader`` is not exported or that OpenGL 2.0 is required, the
process is running without a GPU-backed display context (for example, in a service session or
a remote desktop without GPU acceleration). Run from a GPU-backed interactive display session,
or omit ``--viz newton_gl`` for headless execution.

**Newton GL: Spark + conda**

Conda-installed X11 libraries may conflict with pyglet on Spark, producing:

.. code-block:: text

    pyglet.window.xlib.XlibException: Could not create UTF8 text property

Remove the conflicting conda packages to use the system libraries instead:

.. code-block:: bash

    conda remove --force xorg-libx11 libxcb


See Also
--------

- :doc:`/source/features/visualizer_tiled_camera`: full streaming camera panel guide and tutorial
- :doc:`/source/features/record_video`: recording MP4 clips from a visualizer or sensor
- :doc:`/source/features/draw_markers`: creating and configuring custom visualization markers
- :doc:`/source/how-to/capture_sensor_frames`: saving per-frame sensor outputs during training
- :doc:`/source/overview/core-concepts/renderers`: renderer backends (RTX, Newton Warp, OVRTX)
- :doc:`/source/concepts/scene_data_providers`: how scene data flows to visualizers
- :doc:`/source/overview/core-concepts/physical-backends/newton/index`: Newton backend guide
- :doc:`/source/migration/migrating_to_isaaclab_3-0`: visualizer migration reference
