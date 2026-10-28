.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _demos:

Demos
=====

Select a demo, choose its backend and visualizer, then copy the command to run it locally.
No source checkout is required.

.. raw:: html

   <div class="environment-browser demo-browser" data-demo-browser>
     <section class="environment-command-panel" aria-label="Isaac Lab demo command builder">
       <div class="environment-command-row environment-command-row-primary demo-command-row">
         <span class="environment-command-prefix">uvx isaaclab demo</span>
         <strong class="demo-command-selection" data-demo-name>zoo</strong>
       </div>
       <div class="environment-command-row demo-command-options">
         <label class="environment-selector environment-selector-physics">
           <span>--physics</span>
           <select data-demo-field="physics" aria-label="Physics backend"></select>
         </label>
         <label class="environment-selector environment-selector-renderer">
           <span>--viz</span>
           <select data-demo-field="visualizer" aria-label="Visualizer"></select>
         </label>
       </div>
       <p class="demo-command-description" data-demo-description></p>
       <div class="environment-command-output">
         <code data-command-output></code>
         <div class="environment-command-actions">
           <span class="environment-copy-status" data-copy-status aria-live="polite"></span>
           <button type="button" class="environment-copy-button" data-copy-command
                   aria-label="Copy command" title="Copy command">
             <i class="fa-regular fa-copy" aria-hidden="true"></i>
           </button>
         </div>
       </div>
     </section>

     <div class="demo-card-grid" data-demo-list>
       <button type="button" class="demo-card is-selected" aria-pressed="true"
               data-demo-name="Zoo" data-demo-id="zoo"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-visualizers-isaacsim-physx="kit,newton_gl,rerun,viser"
               data-demo-visualizers-newton-mjwarp="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Animate an arm, biped, quadruped, dexterous hand, quadcopter, and rigid props in one deterministic scene.">
         <img src="../../_static/demos/zoo.jpg" alt="Robots in the Isaac Lab Zoo demo" loading="lazy">
         <span>Zoo</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="MicroDuck Warehouse" data-demo-id="microduck-warehouse"
               data-demo-physics="isaacsim_physx"
               data-demo-visualizers="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="View the static MicroDuck Warehouse through selectable renderers and saved cameras.">
         <img src="../../_static/browser_demos/previews/microduck.jpg"
              alt="MicroDuck warehouse scene" loading="lazy">
         <span>MicroDuck Warehouse</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="H1 Locomotion" data-demo-id="h1-locomotion"
               data-demo-physics="isaacsim_physx,newton_mjwarp"
               data-demo-visualizers="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-visualizers-isaacsim-physx="kit,newton_gl,rerun,viser"
               data-demo-visualizers-newton-mjwarp="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-description="Select H1 robots and control a trained rough-terrain policy with the keyboard and follow camera.">
         <img src="../../_static/demos/h1_locomotion.jpg" alt="H1 locomotion in Isaac Lab" loading="lazy">
         <span>H1 Locomotion</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="SONIC" data-demo-id="sonic"
               data-demo-physics="newton_mjwarp,isaacsim_physx"
               data-demo-visualizers="newton_gl,kit,newton_rtx,rerun,viser"
               data-demo-visualizers-isaacsim-physx="kit,newton_gl,rerun,viser"
               data-demo-visualizers-newton-mjwarp="newton_gl,kit,newton_rtx,rerun,viser"
               data-demo-description="Record G1 traversing deformable gravel and sand with released SONIC policies.">
         <img src="../../_static/browser_demos/previews/g1.jpg" alt="G1 humanoid robot in Isaac Lab" loading="lazy">
         <span>SONIC</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="NIST" data-demo-id="nist"
               data-demo-physics="newton_mjwarp" data-demo-fixed-physics="true"
               data-demo-visualizers="newton_gl,newton_rtx"
               data-demo-description="Assemble ten randomized NIST parts using an automatically downloaded TorchScript policy.">
         <img src="../../_static/demos/nist.jpg" alt="Franka robot assembling parts on a NIST task board" loading="lazy">
         <span>NIST</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Pick and Place" data-demo-id="pick-and-place"
               data-demo-physics="isaacsim_physx" data-demo-visualizers="kit"
               data-demo-description="Interactively pick up a cube with a parallel robot and place it on a target.">
         <img src="../../_static/demos/pick_and_place.jpg" alt="Interactive pick and place demo" loading="lazy">
         <span>Pick and Place</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Newton Block and Tackle" data-demo-id="newton-block-and-tackle"
               data-demo-physics="newton_vbd" data-demo-fixed-physics="true"
               data-demo-visualizers="newton_gl"
               data-demo-description="Drag a cable handle to lift a load through a 4:1 pulley system.">
         <img src="../../_static/demos/newton_block_and_tackle.jpg" alt="Block and tackle pulleys and a red load" loading="lazy">
         <span>Newton Block and Tackle</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Snowball Smash" data-demo-id="snowball-smash"
               data-demo-physics="newton_mpm" data-demo-fixed-physics="true"
               data-demo-visualizers="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-args="--device cuda:0"
               data-demo-description="Smash rigid crates with coupled MPM snowballs.">
         <img src="../../_static/demos/snowball_smash.jpg" alt="Snowballs striking a stack of colored crates" loading="lazy">
         <span>Snowball Smash</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Teapot Fill" data-demo-id="teapot-fill"
               data-demo-physics="newton_mpm" data-demo-fixed-physics="true"
               data-demo-visualizers="newton_gl"
               data-demo-args="--device cuda:0"
               data-demo-description="Fill a Utah teapot with MPM water particles and pour them into a bowl.">
         <img src="../../_static/demos/teapot_fill.jpg" alt="Utah teapot pouring simulated water" loading="lazy">
         <span>Teapot Fill</span>
       </button>
       <button type="button" class="demo-card" aria-pressed="false"
               data-demo-name="Rizon–Sharpa Coffee" data-demo-id="rizon-sharpa-coffee"
               data-demo-physics="newton_coupler" data-demo-fixed-physics="true"
               data-demo-visualizers="newton_gl,newton_rtx,kit"
               data-demo-args="--device cuda:0 --water"
               data-demo-description="Scoop coffee with scripted IK, dissolve it into water and stir.">
         <img src="../../_static/environment-previews/rizon_sharpa_coffee.jpg"
              alt="Rizon and Sharpa hand scooping coffee beside a filled cup" loading="lazy">
         <span>Rizon–Sharpa Coffee</span>
       </button>
     </div>
       <aside class="admonition hint demo-command-note" data-demo-note="h1-locomotion" hidden>
         <p class="admonition-title">H1 locomotion</p>
         <p>H1 locomotion uses a published policy. For autonomous H1 task playback, use <code>isaaclab play</code>.</p>
         <p data-demo-note-visualizers="newton_gl,newton_rtx" hidden>In the Newton viewer, press <kbd>N</kbd> to select a robot,
           <kbd>I</kbd>/<kbd>J</kbd>/<kbd>L</kbd> to walk forward or turn, <kbd>K</kbd> to stop, and <kbd>C</kbd> to toggle the follow camera.</p>
       </aside>
       <aside class="admonition hint demo-command-note" data-demo-note="pick-and-place" hidden>
         <p class="admonition-title">Pick and place</p>
         <p>Pick and place requires Kit input.</p>
       </aside>
     <aside class="admonition hint demo-command-note" data-demo-note="rizon-sharpa-coffee" hidden>
       <p class="admonition-title">Coffee assets</p>
       <p>Coffee props and the licensed Rizon/Sharpa robot bundle load from the Isaac Lab asset library.
         See the demo README for asset-root overrides.</p>
     </aside>
   </div>

NIST peg insertion in the browser
---------------------------------

Try a compact version of the NIST policy with three loose, draggable pegs.
The browser uses small convex peg colliders that preserve the tip chamfers
and simplified socket walls. The native ``nist`` demo above runs the full
ten-part assembly.

Full CAD visuals show the other 16 assemblies already seated as display objects.
The camera orbits and zooms around the board center; hold Shift while dragging to pan.

.. isaaclab-demos:: alphanist
   :title: Try AlphaNIST peg insertion

More examples
-------------

List all demos or focused API examples:

.. code-block:: bash

   uvx --torch-backend cu130 isaaclab demo list
   uvx --torch-backend cu130 isaaclab example list

Run a focused example with ``isaaclab example <name>``. In the Newton GL viewer, the
**Isaac Lab Programs** panel lets you switch between available demos and examples.

MicroDuck Warehouse rendering
-----------------------------

View the static MicroDuck Warehouse through selectable renderers and saved cameras, without advancing physics.

.. code-block:: bash

   uv run --extra ovrtx isaaclab demo microduck-warehouse
   uv run --extra isaacsim isaaclab demo microduck-warehouse renderer=isaacsim_rtx presets=path_traced
   uv run isaaclab demo microduck-warehouse renderer=newton_renderer presets=aisle,depth

Use ``presets=aisle`` for the alternate camera and output presets such as ``depth`` or ``normals``.
For full MDL minimal shading, select ``renderer=isaacsim_rtx presets=minimal``.


.. _demo-sonic:

SONIC on gravel and sand
------------------------

The SONIC demo runs a G1 robot over gravel and sand with Newton MJWarp and MPM,
or Isaac Sim PhysX and PBD. It requires an NVIDIA GPU and uses Kit for RTX rendering.

From the repository root, install the simulator, recording dependencies, and the
additional packages used to download and run the released ONNX models:

.. code-block:: bash

   uv sync --extra isaacsim --extra video
   uv pip install "huggingface-hub>=0.34" "onnxruntime-gpu>=1.22"

The demo downloads the planner and tracker from
`nvidia/GEAR-SONIC <https://huggingface.co/nvidia/GEAR-SONIC>`_ at a pinned revision.
The robot USD is downloaded from the public Isaac asset server.

Run either backend:

.. code-block:: bash

   uv run --extra isaacsim --extra video isaaclab demo sonic \
     --physics newton_mjwarp --output logs/sonic_newton

   uv run --extra isaacsim --extra video isaaclab demo sonic \
     --physics isaacsim_physx --video --output logs/sonic_physx

Without ``--viz``, Newton records offscreen with Kit/RTX and opens no simulation window;
PhysX opens Kit and records only with ``--video``. Both commands above save
``sonic_0000.mp4`` in the selected output directory (``logs/sonic`` when ``--output`` is omitted).
The terminal reports simulation progress. Recording advances
in simulation time, so the full MPM scene can take substantially longer than the video
duration. Pass ``--seconds`` to change the simulation duration.

Pass ``--viz newton_gl`` or ``--viz kit`` to open a Newton simulation window instead.
With an explicit ``--viz`` selection, recording is opt-in via ``--video``.

Append ``env.sim.device=cuda:1`` to select a different CUDA device for physics and SONIC
inference. PhysX keeps state buffers on the CPU for USD readback while particle
simulation uses the selected GPU.

``uv run`` preserves the manually installed packages. If you run ``uv sync`` again,
repeat the ``uv pip install`` command before launching SONIC.

.. _tablecloth-example:

Focused examples
----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Tablecloth trick (Newton VBD, experimental)

      .. figure:: ../_static/examples/tablecloth_h2.jpg
         :width: 100%
         :alt: Three H2 Plus robots comparing cloth pulls in the Cloth Pull Challenge cantina.

         Native Kit viewport with the three pull stations aligned and interior lighting enabled.

      Three H2 Plus robots with SHARPA hands pull tablecloths at different speeds
      in the Cloth Pull Challenge cantina. Newton VBD simulates the cloth,
      tableware, and contact-driven grasps; scripted IK controls the fixed-base robots.
      The contact settings prioritize accuracy over real-time throughput.

      .. code-block:: bash

         uvx --torch-backend cu130 isaaclab example tablecloth --device cuda:0

      Newton GL is the default visualizer. For the Isaac Sim RTX viewport,
      install the ``isaacsim`` extra and select ``--visualizer kit``.

      .. code-block:: bash

         uvx --torch-backend cu130 --from 'isaaclab[isaacsim]' isaaclab example tablecloth --device cuda:0 --visualizer kit

      Kit videos: `five-speed overview
      <https://github.com/user-attachments/assets/6a22e791-ed3c-4488-9853-6e8afdbcbcc2>`__
      and `wider-side grasp and pull
      <https://github.com/user-attachments/assets/27ac0365-e862-4d60-bd85-51f585a580d5>`__.

   .. grid-item-card:: Sharpa RJ45

      .. _sharpa-rj45-example:

      .. figure:: ../_static/examples/sharpa_rj45.jpg
         :width: 100%
         :alt: A SHARPA hand holding a green RJ45 connector at a GB300 switch port.

         Kit preview with optional GB300 rack visuals.

      A scripted Rizon 4s and SHARPA hand insert, release, regrasp and extract an
      RJ45 connector in one continuous simulation. The non-RL manager-based
      example uses native hydroelastic hand contact, a spring clip and a VBD cable.

      .. code-block:: bash

         uvx isaaclab example sharpa-rj45

      From a checkout, use ``uv run isaaclab example sharpa-rj45``.
      NewtonGL displays one GB300 rack; ``--viz kit`` and ``--viz newton_rtx``
      load the full render-only data center without adding physics geometry.
      Requires the external Rizon/SHARPA robot asset bundle; see the
      :download:`asset and recording setup <../../../source/isaaclab_tasks/isaaclab_tasks/contrib/gb300_sharpa_insertion/README.md>`.
