Visualization
=============

.. currentmodule:: isaaclab

Visualizers let you inspect, debug, and record a running simulation. Isaac Lab
offers five options, from lightweight viewers to the Isaac Sim viewport. You can
also view camera-sensor output in the streaming camera panel.

Try :ref:`Franka cube stacking <interactive-demo-franka-stack>` for a compact simulation
running directly in the browser. Orbit its camera, watch the scripted grasp, or control
the end effector in Manual mode.

.. raw:: html

   <style>
   .viz-grid { display:flex; gap:7px; margin: 0.5em 0; }
   .viz-hero-stack { display:flex; flex-direction:column; gap:7px; margin: 0.5em 0 1em; }
   .viz-hero-stack .viz-grid { margin:0; }
   .viz-grid > div { flex:1 1 0; min-width:0; }
   .viz-grid video, .viz-grid img { display:block; width:100%; border-radius:0 !important; padding:0 !important; background:none !important; }
   .viz-grid-stretch video, .viz-grid-stretch img { height:260px; object-fit:cover; }
   .viz-grid-stretch.viz-grid-hero-tiles video { height:315px; }
   .viz-grid-fit { justify-content:center; }
   .viz-grid-fit > div { flex:0 0 auto; }
   .viz-grid-fit video, .viz-grid-fit img { width:auto; height:488px; }
   .viz-grid-stretch video.viz-no-crop { object-fit:contain; background:#000; }
   .viz-grid-stretch video.viz-crop-x8 { width:calc(100% + 16px); margin-left:-8px; }
   .viz-grid-stretch img.viz-crop-top { object-position:center 25%; }
   /* Per-tile crop windows: each video's (taller) height sets the zoom and its negative
      margin-top picks which slice of the 265px wrap is shown, matching robot size within each
      row. Tuned against the currently hosted clips, not capture_visualizer.py's own (much
      smaller) fixed trim -- re-check if the clips are ever regenerated. Rerun/Viser cap how far
      they can zoom out before their own baked-in UI chrome reappears; Viser is already at that
      limit. */
   .viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap { height:265px; }
   .viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap video { object-position:center center; height:265px; margin-top:0; }
   .viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap.viz-crop-newton-gl video { height:480px; margin-top:-105px; }
   .viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap.viz-crop-viser video { height:314px; margin-top:-49px; }
   .viz-grid-stretch.viz-grid-hero-tiles .viz-hero-wrap.viz-crop-rerun video { height:293px; margin-top:-19px; }
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
   <div class="viz-grid viz-grid-stretch viz-grid-hero-tiles">
     <div class="viz-hero-wrap viz-crop-newton-gl">
       <video autoplay loop muted playsinline preload="auto">
         <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/hero_newton_gl.mp4" type="video/mp4">
       </video>
       <div class="viz-label">Newton GL</div>
     </div>
     <div class="viz-hero-wrap viz-crop-viser">
       <video autoplay loop muted playsinline preload="auto" class="viz-hero-start-1s">
         <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/hero_viser.mp4" type="video/mp4">
       </video>
       <div class="viz-label">Viser</div>
     </div>
   </div>
   <div class="viz-grid viz-grid-stretch viz-grid-hero-tiles">
     <div class="viz-hero-wrap viz-crop-newton-rtx">
       <video autoplay loop muted playsinline preload="auto" class="viz-hero-speedup">
         <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/hero_newton_rtx.mp4" type="video/mp4">
       </video>
       <div class="viz-label">Newton RTX</div>
     </div>
     <div class="viz-hero-wrap viz-crop-kit">
       <video autoplay loop muted playsinline preload="auto" class="viz-crop-x8 viz-hero-speedup">
         <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/hero_kit.mp4" type="video/mp4">
       </video>
       <div class="viz-label">Kit</div>
     </div>
     <div class="viz-hero-wrap viz-crop-rerun">
       <video autoplay loop muted playsinline preload="auto" class="viz-hero-start-1s">
         <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/hero_rerun.mp4" type="video/mp4">
       </video>
       <div class="viz-label">Rerun</div>
     </div>
   </div>
   </div>

   <script>
   // Kit/Newton RTX are encoded 10% slower than the other 3 (output_speed_factor in
   // capture_visualizer.py); undo that so all 5 independently-looping clips stay in phase.
   document.querySelectorAll(".viz-hero-speedup").forEach(function (v) {
     v.playbackRate = 1 / 0.9;
   });
   // Start Viser/Rerun 1.2s into their clip instead of at 0; native loop still wraps to 0 as usual.
   document.querySelectorAll(".viz-hero-start-1s").forEach(function (v) {
     var seek = function () { v.currentTime = 1.2; };
     if (v.readyState >= 1) seek();
     else v.addEventListener("loadedmetadata", seek, { once: true });
   });
   </script>

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

.. note::

   Most tasks default to a PhysX backend, which requires Isaac Sim. If it isn't installed yet,
   add ``--extra isaacsim`` to the ``uv run`` commands above; see
   :ref:`installation-optional-extras` for details.

.. note::

   ``train`` and ``play`` run without a visualizer unless ``--viz`` is passed, while packaged demos and
   examples open their own default visualizer. ``--viz none`` turns it off.

For combining visualizers, running headless, and other common use cases, see `Usage`_ below.

Visualizer presets
~~~~~~~~~~~~~~~~~~

Task commands accept ``visualizer=newton_gl``, ``visualizer=kit``, or
``visualizer=rerun,newton_gl``. ``--viz`` and ``--visualizer`` are aliases for the same
selection. ``visualizer=none`` disables viewers. Selection happens during configuration
composition, before choosing process runtimes or creating the scene.

Registered tasks already supply the built-in backend choices. Declare only common settings for
ordinary use; ``--viz`` selects the backend during task composition:

.. code-block:: python

   sim_cfg = SimulationCfg(
       visualizer_cfgs=VisualizerCfg(eye=(8.0, 8.0, 3.0), max_visible_envs=16),
   )

Task scripts can customize alternatives with ``isaaclab_tasks.utils.presets.MultiBackendVisualizerCfg``.
Keep view, window, and scene settings on ``VisualizerCfg``. Its ``backend`` selects the implementation;
``MultiBackendVisualizerCfg()`` supplies the built-in choices without repeating those common settings.
Compose standalone declarations during initial configuration, then call the core launcher directly:

.. code-block:: python

   from isaaclab.app import launch_simulation
   from isaaclab.sim import SimulationCfg
   from isaaclab.visualizers import VisualizerCfg, WindowCfg

   from isaaclab_tasks.utils import resolve_presets
   from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

   sim_cfg = SimulationCfg(
       visualizer_cfgs=VisualizerCfg(
           eye=(8.0, 8.0, 3.0),
           max_visible_envs=16,
           window=WindowCfg(size=(1280, 720)),
           backend=MultiBackendVisualizerCfg(),
       ),
   )

   sim_cfg = resolve_presets(sim_cfg, overrides={"visualizer_cfgs.backend": args_cli.visualizer or "default"})
   with launch_simulation(sim_cfg, args_cli):
       # Construct the simulation and scene here.
       ...

One ``VisualizerCfg`` supplies the common settings for every selected backend. Selecting
``rerun,newton_gl`` creates both viewers; no outer list or repeated common settings is needed.
Backend-specific options are available when needed, but ordinary configuration uses the unchanged
``MultiBackendVisualizerCfg()`` declaration.

Use a list of ``VisualizerCfg`` declarations for different views or window sizes. Backend selection
applies to each declaration independently. For example, a task with these two views opens four
windows with ``--viz=newton_gl,newton_rtx``: one GL and one RTX window per view.

.. code-block:: python

   visualizer_cfgs = [
       VisualizerCfg(
           eye=(4.0, -4.0, 3.0), window=WindowCfg(size=(640, 480)),
       ),
       VisualizerCfg(
           eye=(8.0, 0.0, 2.0), window=WindowCfg(size=(800, 600)),
       ),
   ]

Keep ``MultiBackendVisualizerCfg`` when declaring available alternatives. A list of concrete backend
configs declares active viewers, so replacing the preset with a list changes the default behavior.

Task entry points perform this selection through ``resolve_task_config``:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole visualizer=rerun,newton_gl

``--viz rerun,newton_gl`` is equivalent. Task composition selects the backends inside
``VisualizerCfg.backend`` before launch; launch creates each viewer with the shared settings.
Viewer implementations are imported only when constructed.

For ``--video`` without a selected capture-capable viewer, launch selects a headless Newton GL
producer. ``visualizer=none`` disables windows; recording still runs its producer on demand.
Use ``--video sensor:<name>`` to record a scene camera without a visualizer producer.


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
           <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/showcase_newton_gl_allegro.mp4" type="video/mp4">
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

      Simulation particles are visible by default. Set ``NewtonGLBackendCfg(show_particles=False)``
      to hide them, for example when displaying only a reconstructed fluid surface or a cloth mesh.

      .. raw:: html

           </div>
           <div>
             <video autoplay loop muted playsinline controls preload="auto" style="width:100%;">
               <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/showcase_newton_gl_dominoes.mp4" type="video/mp4">
             </video>
             <p class="viz-cap">newton-dominoes example<br>Right-click dragging the first domino
             triggers the cascade across an NVIDIA-logo domino layout</p>
           </div>
         </div>

      Try :ref:`Domino cascade <interactive-demo-dominoes>` for a compact 24-slab version running
      locally in the browser, with friction and a repeatable toppling pulse.

      .. dropdown:: NewtonGLBackendCfg source
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualizer_cfg.py
            :language: python
            :pyobject: NewtonGLBackendCfg

      .. dropdown:: VisualizerCfg source (shared settings)
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
           <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/showcase_viser_lift_kuka_allegro.mp4" type="video/mp4">
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

      Browser and recording options belong to the Viser backend. Select this alternative with
      ``--viz viser``:

      .. code-block:: python

         from isaaclab_visualizers.viser import ViserBackendCfg
         from isaaclab.visualizers import VisualizerCfg
         from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

         visualizer_cfgs = VisualizerCfg(
             max_visible_envs=16,
             backend=MultiBackendVisualizerCfg(
                 viser=ViserBackendCfg(
                     port=8080, bind_address="0.0.0.0", display_address="localhost",
                     open_browser=True, share=False, record_to_viser="rollout.viser",
                 ),
             ),
         )

      .. dropdown:: ViserBackendCfg source
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/viser/viser_visualizer_cfg.py
            :language: python
            :pyobject: ViserBackendCfg

      .. dropdown:: VisualizerCfg source (shared settings)
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
           <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/showcase_newton_rtx_h1_stairs.mp4" type="video/mp4">
         </video>
         <p class="viz-cap">Isaac-Velocity-Rough-H1 in Newton RTX Visualizer<br>Each H1 humanoid climbs a
         staircase sub-terrain</p>

      **Visualizer-specific features:**

      - OVRTX path-traced rendering for photorealistic, physically-based lighting

      .. note::

         The following features are not yet supported and will be added in a future release:
         live plots and rigid-body dragging. Visualization markers, camera selection, scene-camera
         display, pause controls, and headless recording are supported.

      .. dropdown:: NewtonRTXBackendCfg source
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/newton/newton_visualizer_cfg.py
            :language: python
            :pyobject: NewtonRTXBackendCfg

      .. dropdown:: Shared window options
         :icon: code

         .. literalinclude:: ../../../source/isaaclab/isaaclab/visualizers/visualizer_cfg.py
            :language: python
            :pyobject: WindowCfg

      .. dropdown:: VisualizerCfg source (shared settings)
         :icon: code

         .. literalinclude:: ../../../source/isaaclab/isaaclab/visualizers/visualizer_cfg.py
            :language: python
            :pyobject: VisualizerCfg

      .. warning::

         Newton RTX (OVRTX) is a kitless renderer and cannot be used in the same process as Kit
         (``presets=isaacsim_physx``). ``presets=ovphysx`` is itself kitless and works fine with
         Newton RTX. Use ``presets=newton_mjwarp,ovrtx`` or ``presets=ovphysx,ovrtx`` with
         ``--viz newton_rtx``, or switch to ``--viz newton_gl``, ``--viz viser``, ``--viz rerun``,
         or ``--viz kit`` with a Kit-compatible physics backend.

   .. tab-item:: Rerun

      Like Viser, `Rerun <https://rerun.io/>`_ streams simulation state to a local web server, for
      remote monitoring, timeline playback, and recording to ``.rrd`` files.

      .. raw:: html

         <video autoplay loop muted playsinline controls preload="auto" style="width:100%;">
           <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/showcase_rerun_shadow_reorient.mp4" type="video/mp4">
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

      Keep history or record a replay through the Rerun backend; select it with ``--viz rerun``:

      .. code-block:: python

         from isaaclab_visualizers.rerun import RerunBackendCfg
         from isaaclab.visualizers import VisualizerCfg
         from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

         visualizer_cfgs = VisualizerCfg(
             backend=MultiBackendVisualizerCfg(
                 rerun=RerunBackendCfg(
                     open_browser=True, keep_historical_data=True, record_to_rrd="rollout.rrd",
                 ),
             ),
         )

      .. dropdown:: RerunBackendCfg source
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/rerun/rerun_visualizer_cfg.py
            :language: python
            :pyobject: RerunBackendCfg

      .. dropdown:: VisualizerCfg source (shared settings)
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
           <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/showcase_kit_franka_reach.mp4" type="video/mp4">
         </video>
         <p class="viz-cap">Isaac-Reach-Franka in Kit Visualizer<br>Each Franka arm reaches its end-effector
         toward a randomized target pose</p>

      **Visualizer-specific features:**

      - Direct access to the Isaac Sim USD stage for inspecting and editing prims at runtime
      - Full Isaac Sim GUI tooling (Property, Layers, and Stage panels)

      .. note::

         Kit's ImGui-based UI writes an ``imgui.ini`` file, recording window layout (panel
         positions, docking, collapsed state), to the process working directory. If that is the
         repository root, it shows up as an untracked file. ``imgui.ini`` is already listed in
         ``.gitignore``; delete it or launch from outside the repository if you would rather it
         not appear at all.

      .. dropdown:: KitBackendCfg source
         :icon: code

         .. literalinclude:: ../../../source/isaaclab_visualizers/isaaclab_visualizers/kit/kit_visualizer_cfg.py
            :language: python
            :pyobject: KitBackendCfg

      .. dropdown:: VisualizerCfg source (shared settings)
         :icon: code

         .. literalinclude:: ../../../source/isaaclab/isaaclab/visualizers/visualizer_cfg.py
            :language: python
            :pyobject: VisualizerCfg

Shared Features
---------------

.. _visualization-streaming-camera-view:

Streaming Camera View
~~~~~~~~~~~~~~~~~~~~~

The streaming camera view is a live monitoring and debugging tool. It combines ground-truth camera
frames from multiple environments (RGB, depth, segmentation, or surface normals) into a single
panel. Cameras are declared in the scene before cloning; visualizers only read their output.
Multiple visualizers can display the same sensor with different tile selections.

.. raw:: html

   <div class="viz-stack-centered">
     <div>
       <video autoplay loop muted playsinline controls preload="auto">
         <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/streaming_newton_galbot_interactive.mp4" type="video/mp4">
       </video>
       <p class="viz-cap">Interactive view of the Galbot cube stacking environment</p>
     </div>
     <div>
       <video autoplay loop muted playsinline controls preload="auto">
         <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/streaming_newton_galbot_tiled.mp4" type="video/mp4">
       </video>
       <p class="viz-cap">Streaming view showing tiled Galbot wrist-camera feeds</p>
     </div>
   </div>

Attach a camera to a robot body to follow its motion, or select an existing view such as the Galbot
task's wrist-mounted and ego cameras. The view is supported in the Kit, Newton GL, Newton RTX,
Rerun, and Viser visualizers. **Kit** opens it as a separate **Streaming View** viewport, which can
sit side by side with the default viewport for dual monitoring. **Newton GL** and **Newton RTX**
provide a **Camera View** selector in the sidebar for switching between perspective and scene-camera
views.

See :doc:`/source/how-to/visualizer_streaming_camera_view` for the ``VisualizerCfg`` options, display
sources, troubleshooting, migration from generated cameras, and step-by-step examples.


Visualization Markers
~~~~~~~~~~~~~~~~~~~~~

Visualization markers draw debug geometry over the scene via
:class:`~isaaclab.markers.VisualizationMarkers`.

.. raw:: html

   <div class="viz-grid viz-grid-stretch">
     <div>
       <img src="../../_static/visualizers/markers_anymal_d.jpg" alt="Velocity arrow marker on an AnymalD robot">
       <p class="viz-cap">Large green/blue arrow markers showing target and base velocity for an AnymalD robot</p>
     </div>
     <div>
       <img src="../../_static/visualizers/markers_franka.jpg" alt="Joint arrow markers on a Franka arm and contact sensor markers on a cube" class="viz-crop-top">
       <p class="viz-cap">Arrow markers on the Franka arm's joints, with contact sensor markers on the cube</p>
     </div>
   </div>

See :doc:`/source/how-to/create_visualization_markers` for creating and configuring custom markers.

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
         <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/liveplot_newton_gl.mp4" type="video/mp4">
       </video>
       <p class="viz-cap">Newton GL visualizer, with 3 plots opened</p>
     </div>
     <div>
       <video autoplay loop muted playsinline controls preload="auto">
         <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/visualizers/liveplot_viser.mp4" type="video/mp4">
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
         <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/how_to/record_video_tiled_kit_viewport.mp4" type="video/mp4">
       </video>
       <p class="viz-cap">Kit visualizer tiled streaming</p>
     </div>
     <div>
       <video autoplay loop muted playsinline controls preload="auto">
         <source src="https://huggingface.co/datasets/nvidia/isaaclab-docs-assets/resolve/main/how_to/record_video_shadow_hand_tiled_gt.mp4" type="video/mp4">
       </video>
       <p class="viz-cap">Recorded clip with ground-truth sensor views</p>
     </div>
   </div>

.. _record_video_cli:

Pass ``--video [SOURCE]`` on the command line for a quick recording, or define multiple
``VideoRecorderCfg`` entries to record multiple sources at once. ``--viz`` still decides which
visualizers open a window; a visualizer that only records runs headless:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Command line
     - Records from
   * - ``--video`` (same as ``--video viz``)
     - the first capture-capable visualizer ``--viz`` selected, in its window; if none is selected, or only
       streaming ones such as ``viser`` or ``rerun``, a headless ``newton_gl``
   * - ``--video viz:newton_rtx``
     - a ``newton_rtx``: the ``--viz``-selected one if selected, else an extra headless one
   * - ``--video --viz newton_rtx``
     - the selected ``newton_rtx`` window
   * - ``--video viz:newton_gl --viz viser``
     - ``viser`` runs; an extra headless ``newton_gl`` is added only for recording
   * - ``--video sensor:wrist_camera[:rgb|depth|...]``
     - that scene sensor; no visualizer is added

A Hydra override after ``--video`` is never taken as its source: ``--video presets=newton_mjwarp``
records from ``viz`` and applies the preset.

Kit, Newton GL, and Newton RTX visualizers can be recorded from; the web-based visualizers Viser and
Rerun cannot.

See :doc:`/source/how-to/record_video` for the full guide and tutorial.

Keyboard Input
~~~~~~~~~~~~~~

A visualizer with a local window reports the keys typed into it through
:attr:`~isaaclab.visualizers.BaseVisualizer.key_event_source`, for example to drive a robot from
the keyboard. Each key press and release arrives as a W3C ``KeyboardEvent.code`` string such as
``"KeyW"``:

- :class:`~isaaclab_visualizers.kit.KitVisualizer` reports physical key positions, independent of
  the keyboard layout. Keys typed into Kit's own UI, such as a text field, are reported too.
- :class:`~isaaclab_visualizers.newton.NewtonGLVisualizer` and
  :class:`~isaaclab_visualizers.newton.NewtonRTXVisualizer` report layout-mapped keys: a code names
  the key that types the same character on a US layout. Presses typed into the window's UI are
  withheld. Presses and releases pair up reliably on a US layout; on other layouts, changing Shift
  while holding a digit or punctuation key can leave it held until the window loses focus.

The Viser and Rerun visualizers, and any visualizer running headless, have no keyboard source and
return ``None``. A headless Viser server still takes input from connected browsers, but not through
``key_event_source``.

Each Newton window has its own source. Kit visualizers share the Kit application window, so each of
them reports the same keys; a consumer that listens to every visualizer must accept a key from more
than one source.

The visualizer connects native keyboard and focus events when its window initializes and
disconnects them before the window closes. A subscription only adds or removes a listener;
it does not initialize window resources. The source owns no held-key state or camera policy.

Close the subscription when you are done with it. A consumer that uses the camera's motion keys
also turns off keyboard camera movement with
:meth:`~isaaclab.visualizers.BaseVisualizer.set_camera_keys_enabled`; a Newton window then stops
moving its camera with WASD/QE and the arrow keys, and Kit's viewport, whose camera keys only act
while the right mouse button is held, is unaffected:

.. code-block:: python

   visualizer = sim.visualizers[0]
   source = visualizer.key_event_source
   if source is not None:
       held = set()

       def on_key(code: str, pressed: bool) -> None:
           (held.add if pressed else held.discard)(code)

       visualizer.set_camera_keys_enabled(False)
       try:
           with source.add_key_listener(on_key, held.clear):
               ...  # use held while stepping the simulation
       finally:
           visualizer.set_camera_keys_enabled(True)

Subscribe, unsubscribe and close on the thread that steps the visualizer; callbacks run there too,
in event order. Events arrive as the window reports them: key repeat can arrive as further presses,
and a release can arrive without its press, so a listener tracks the keys it holds.
``on_focus_lost`` runs when the window stops delivering keys, because it lost focus, its UI took
the keyboard, or the visualizer closed; treat every held key as released.


.. _visualization-configuration:

Usage
-----

Common Recipes
~~~~~~~~~~~~~~

**Headless training with video recording**

Run without a window and record clips from a headless Newton GL visualizer, or name another
capture source, e.g. ``--video viz:kit``:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --video

See :doc:`/source/how-to/record_video` for clip length, interval, and multi-source options.

.. _visualization-recording-angle:

**Combining an interactive view with a headless recording source**

Watch training live in Kit while recording from a separate headless Newton GL angle by running with
``--viz kit --video viz:newton_gl`` and giving each visualizer its own camera. See
:ref:`Recording from an independent camera angle <how_to_record_video_angle>` for the configuration.

**Following a moving robot (Kit)**

Lock the Kit camera to a moving asset instead of updating ``eye``/``lookat`` yourself every
step, running with ``--viz kit``:

.. code-block:: python

    from isaaclab_visualizers.kit import KitBackendCfg

    from isaaclab.visualizers import VisualizerCfg

    from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

    sim_cfg.visualizer_cfgs = VisualizerCfg(
        eye=(4.0, 4.0, 2.0),
        backend=MultiBackendVisualizerCfg(kit=KitBackendCfg(origin_type="asset", origin_track_path="robot")),
    )

**Sharing a live view with a remote teammate**

Viser can request a public share URL for the running session, useful for remote pairing
without screen-sharing, running with ``--viz viser``:

.. code-block:: python

    from isaaclab_visualizers.viser import ViserBackendCfg

    from isaaclab.visualizers import VisualizerCfg

    from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

    sim_cfg.visualizer_cfgs = VisualizerCfg(
        backend=MultiBackendVisualizerCfg(viser=ViserBackendCfg(share=True, open_browser=True)),
    )

The share URL is logged on startup. Rerun has no equivalent config field, but exposes its own
share button in the native UI.

Resolution Rules
~~~~~~~~~~~~~~~~

``--viz`` (comma-separated, e.g. ``--viz kit,newton_gl``) selects which visualizers run, and
``SimulationCfg.visualizer_cfgs`` configures them. Each selected type uses its configured settings,
or else that type's default config. Concrete visualizer configs run as given when no CLI selection
is supplied. ``SimulationCfg.visualizer_cfgs`` defaults to ``VisualizerCfg(backend=[])``;
settings alone do not open a viewer.

Isaac Sim opens a window when CLI or config selects ``kit``, and never with ``HEADLESS=1`` or
livestreaming. A visualizer a ``--video`` source adds for recording always runs headless.

Use one ``VisualizerCfg(backend=MultiBackendVisualizerCfg())`` for shared settings, selecting
one or several backends through ``visualizer=...``. A list of ``VisualizerCfg`` declarations is
only needed for independently configured views, such as a different camera angle for recording
(see :ref:`how_to_record_video_angle`).

.. _visualization-common-modes:

.. list-table:: Common modes
   :header-rows: 1
   :widths: 30 35 35

   * - CLI args
     - ``visualizer_cfgs``
     - Effective behavior
   * - no ``--viz``
     - ``VisualizerCfg()`` (the default)
     - No visualizer launches; no window, no capture source.
   * - ``--viz kit,newton_gl``
     - ``VisualizerCfg(max_visible_envs=16)``
     - Launch Kit and Newton GL, both showing at most 16 environments.
   * - ``--video`` without ``--viz``
     - ``VisualizerCfg()``
     - Launch Newton GL without a window, only as the ``--video`` recording source.
   * - no ``--viz``
     - ``VisualizerCfg(backend=MultiBackendVisualizerCfg())``
     - No visualizer launches; backend alternatives remain inactive until selected.
   * - ``visualizer=rerun,newton_gl``
     - ``VisualizerCfg(max_visible_envs=16, backend=MultiBackendVisualizerCfg())``
     - Launch Rerun and Newton GL, both showing at most 16 environments.

For migration context, see :doc:`/source/isaaclab/migrating`.


.. _visualization-partial-visualization:

Partial Visualization
~~~~~~~~~~~~~~~~~~~~~

Visualizers can be configured to visualize just a subset of environments.
This is called partial visualization.

There are 3 fields exposed in the ``VisualizerCfg`` for selecting environments for partial visualization:

- ``max_visible_envs`` caps how many envs are shown.
- ``visible_env_indices`` explicitly selects the envs to visualize, in the order given. It takes precedence over
  ``randomly_sample_visible_envs`` and is also capped by ``max_visible_envs``.
- ``randomly_sample_visible_envs`` (default ``True``): when ``visible_env_indices`` is unset and ``max_visible_envs``
  is set, enables randomly sampling the selected envs. If disabled, the first ``max_visible_envs`` envs are selected.

Also, there is a CLI arg ``--max_visible_envs`` that overrides ``VisualizerCfg.max_visible_envs`` for the run.

.. note::

   By default, selected envs keep their simulated positions, so a random sample or the first envs of a large grid can lie
   outside the camera view. Choose indices near the camera target, or move ``eye`` and ``lookat``.

Scene Background
~~~~~~~~~~~~~~~~

Kit and Newton RTX use the scene's authored lights and HDR background by default. Configure the
environment once on the scene, without repeating its HDR path or intensity in the visualizer:

.. code-block:: python

    env_cfg.scene.sky_light.spawn.texture_file = "/path/to/evening.hdr"
    env_cfg.scene.sky_light.spawn.intensity = 1000.0

``VisualizerCfg.background_color`` defaults to ``None``. An explicit normalized RGB color changes
only the visible background; the scene lights continue to illuminate objects and contribute
reflections. Newton GL uses its procedural sky when no color override is supplied:

.. code-block:: python

    from isaaclab.visualizers import VisualizerCfg

    from isaaclab_tasks.utils.presets import MultiBackendVisualizerCfg

    env_cfg.sim.visualizer_cfgs = VisualizerCfg(
        background_color=(0.1, 0.2, 0.3), backend=MultiBackendVisualizerCfg(),
    )


Performance
~~~~~~~~~~~

.. list-table::
   :widths: 18 82
   :header-rows: 1

   * - Visualizer
     - Tips
   * - Newton GL
     - Lowest overhead; lower ``window.fps`` to limit wall-clock render frequency.
   * - Viser
     - Newton Warp renderer; use ``max_visible_envs`` to limit the number of rendered
       environments (see :ref:`visualization-partial-visualization`).
   * - Newton RTX
     - Native Newton RTX rendering; reduce configured perspective resolution or lower ``window.fps``.
       Environment selection limits sensor display tiles; the perspective view shows the full scene.
   * - Rerun
     - Web viewer may slow down with many environments; use ``--num_envs`` to reduce load.
   * - Kit
     - Highest overhead of the five visualizers; reduce ``window.size``
       or use ``--max_visible_envs`` (see :ref:`visualization-partial-visualization`).


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
   * - Keyboard input, ``key_event_source``
     - ✓
     - ✓
     - ✗
     - ✗
     - ✓

**Lighting differences across visualizers**

Each backend lights the scene differently, so the same environment can look noticeably
different across visualizers. Kit renders the scene's actual authored USD lights. Newton GL
uses a fixed sky-gradient and single directional light color
(:attr:`~isaaclab_visualizers.newton.NewtonGLBackendCfg.sky_upper_color`,
``sky_lower_color``, ``light_color``), independent of scene USD lights. Newton RTX renders the
authored scene through Isaac Lab's OVRTX renderer, shared with cameras using the same
``OVRTXRendererCfg``. Newton GL displays the rendered GPU image and handles window controls;
Newton's ``ViewerRTX`` is not constructed. Scene transforms, deformable geometry, and material
updates follow the camera renderer's normal synchronization. Perspective resolution stays
fixed when the window is resized; the window scales the image for presentation.
Configure lights and environment placement in the scene. Viewer-only ``rtx_environment`` rigs
and ``world_spacing`` are no longer supported: declare scene lights for studio lighting, remove
scene lights for an unlit scene, and set ``InteractiveSceneCfg.env_spacing`` for world placement.
Particle color overrides and rigid-body picking remain GL-only. Viser uses a single ambient light
with no directional key light, so scenes tend to look darker and flatter than the other backends.
Rerun uses fixed built-in viewer shading with no scene-driven lighting.

**Kit: incompatible with ovphysx / ovrtx presets**

``--viz kit`` cannot be used with ``presets=ovphysx`` or ``presets=ovrtx`` in the same process.
Use ``--viz newton_gl``, ``--viz rerun``, or ``--viz viser`` with those presets, or omit
``--viz`` for headless execution.

**Newton RTX: incompatible with Kit**

``--viz newton_rtx`` raises a ``RuntimeError`` at startup if the active physics backend is
``isaacsim_physx`` (i.e. ``presets=isaacsim_physx``), since OVRTX is a kitless renderer and cannot share
a process with Kit. ``presets=ovphysx`` is itself kitless and remains supported. Use
``presets=newton_mjwarp,ovrtx`` or ``presets=ovphysx,ovrtx`` with ``--viz newton_rtx``, or switch
to ``--viz newton_gl``, ``--viz viser``, ``--viz rerun``, or ``--viz kit`` with a Kit-compatible
physics backend.

**Rerun: large environment performance**

The Rerun web viewer may slow down or crash with many environments. Reduce load with
``--num_envs``:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

          uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole --viz rerun --num_envs 512

**Rerun: blank page until the first payload loads**

Rerun may show a blank browser tab for up to about 10 seconds while it connects
and receives scene data. Wait for the first frame; no action is needed.

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

.. raw:: html

   <span id="see-also"></span>

Where to go next
----------------

- :doc:`/source/how-to/visualizer_streaming_camera_view`: stream a camera view step by step
- :doc:`/source/how-to/record_video`: recording MP4 clips from a visualizer or sensor
- :doc:`/source/how-to/create_visualization_markers`: creating and configuring custom visualization markers
- :doc:`/source/how-to/capture_sensor_frames`: saving per-frame sensor outputs during training
- :doc:`/source/concepts/renderers`: renderer backends (RTX, Newton Warp, OVRTX)
- :doc:`/source/concepts/scene_data_providers`: how scene data flows to visualizers
- :ref:`physics-backends-newton`: Newton backend guide
- :doc:`/source/isaaclab/migrating`: visualizer migration reference
