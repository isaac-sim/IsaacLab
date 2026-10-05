.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _demos:

Demos
=====

Demos are polished showcases of Isaac Lab capabilities. They ship in the ``isaaclab`` wheel, so you can run them
without a source checkout. Start with Zoo to see several robot families and simulation features in one scene, or run
``uvx isaaclab demo list`` to inspect the complete catalog.

All programs live under the repository-level ``examples/`` directory: curated showcases in ``examples/demos/``,
focused programs for learning an API or tuning a feature in directories such as ``examples/mpm/`` and
``examples/sensors/``, and shared data in ``examples/assets/``. List focused programs with
``uvx isaaclab example list`` and run one with ``uvx isaaclab example <name>``.

Demo and example commands show the same Isaac Lab startup screen as task playback while
their simulation initializes. Pass ``--info`` to keep startup messages visible.

In any packaged demo or example running with ``--viz newton_gl``, the **Isaac Lab Programs**
panel shows the core showcases first, with focused examples in a separate collapsed section.
Selecting one closes the current simulation and relaunches it with ``--viz newton_gl``.
For example, run ``uvx isaaclab demo zoo --viz newton_gl`` to explore the catalog.
Programs whose required modules are unavailable, hardware-dependent teleoperation, and
Kit-only demos are not shown.

For particle-material comparisons and solver guidance, see :ref:`newton-tuning-mpm`.

Command Builder
---------------

.. raw:: html

   <div class="environment-browser demo-browser" data-demo-browser>
     <section class="environment-command-panel" aria-label="Isaac Lab demo command builder">
       <div class="environment-command-row environment-command-row-primary demo-command-row">
         <span class="environment-command-prefix" aria-hidden="true">uvx</span>
         <strong class="demo-command-selection" data-demo-name>Zoo</strong>
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
         <img src="../../_static/demos/arms.jpg" alt="Robots in the Isaac Lab Zoo demo" loading="lazy">
         <span>Zoo</span>
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
               data-demo-visualizers="kit,newton_gl,newton_rtx,rerun,viser"
               data-demo-args="--device cuda:0"
               data-demo-description="Fill a Utah teapot with MPM water particles and pour them into a bowl.">
         <img src="../../_static/demos/teapot_fill.jpg" alt="Utah teapot pouring simulated water" loading="lazy">
         <span>Teapot Fill</span>
       </button>
     </div>
   </div>

H1 locomotion uses a published policy. In the Newton viewer, press N to select a robot, I/J/L to walk
forward or turn, K to stop, and C to toggle the follow camera. Pick and place requires Kit input. For autonomous H1 task playback, use ``isaaclab play``.

QA with uvx
-----------

Give QA ``tools/qa_uvx.py`` from the source checkout. It is a standalone Python script and can be copied
to a machine without Isaac Lab installed. The same commands work on Linux and Windows with uv on PATH:

.. code-block:: bash

   # Test the default published package, demo catalogs, installed resources, and demo help.
   uv run --no-project python tools/qa_uvx.py

   # Test the candidate wheel and also check example help.
   uv run --no-project python tools/qa_uvx.py --package /path/to/isaaclab.whl --examples

   # Run Zoo for 60 steps with its default interactive viewer.
   uv run --no-project python tools/qa_uvx.py --package /path/to/isaaclab.whl --run --demo zoo

   # Run a short simulation without a viewer on a GPU worker.
   uv run --no-project python tools/qa_uvx.py --package /path/to/isaaclab.whl --run --demo zoo -- --viz none --device cuda:0

``--package`` also accepts a pinned requirement such as ``isaaclab==<candidate-version>`` or
``isaaclab[isaacsim]==<candidate-version>`` for Pick and Place. Add ``--index https://pypi.nvidia.com``
when testing a release hosted on NVIDIA's index. The package must contain the redesigned ``demo`` command;
older releases, including ``3.0.0rc1``, do not. Validate both the candidate wheel and the default published
package before declaring the advertised ``uvx isaaclab demo <name>`` path ready.

The script launches the real ``uvx`` executable with Python 3.12, ignores uv configuration files and
already-installed tools, clears Python import overrides, and runs from a temporary directory outside the
checkout. It checks that every catalog entry resolves to a file inside the installed package. Default checks
do not simulate; ``--run`` launches the selected demos with ``--max_steps``. Repeat ``--demo`` to select
several demos, use ``--steps`` to change their duration, and pass backend or viewer options after ``--``.
Without a selection, all demos with their required modules available are checked. Missing optional modules
are recorded as skips; explicitly selected demos fail when their dependencies are missing.

Each run writes command logs and ``report.json`` in a new ``uvx-qa-<timestamp>`` directory; ``--output``
can choose a different new directory. The report records the OS, architecture, Python and package versions,
GPU/driver information when available, commands, exit codes, durations, and skips. Failures and timeouts
return a nonzero exit status. ``--timeout`` sets the deadline per command, including the initial package
installation (default: 1800 seconds). uv's download cache is reused; to test with a fresh cache, set
``UV_CACHE_DIR`` to a new directory before running the script.

Run the candidate on each supported QA platform, including Linux x86_64, Linux aarch64, and Windows x86_64.
CLI checks need network access and disk space for the package dependencies. Simulation checks also need
the selected backend's supported hardware; viewer checks need a desktop session. Exercise Zoo and H1 for
rigid-body simulation, Block and Tackle for VBD, and Snowball Smash and Teapot Fill for CUDA MPM. Test
Pick and Place separately with the Isaac Sim extra. A successful simulation check establishes startup,
bounded execution, and shutdown; QA should also inspect rendering and keyboard/drag interactions in a
longer interactive run. Attach the result directory and any visual observations to the QA report.

Release publication
^^^^^^^^^^^^^^^^^^^

The bare ``uvx isaaclab`` command resolves the package from PyPI. Wheels uploaded only as GitHub artifacts
or to NVIDIA's index do not make this command available. A Python 3.12 ABI error can mean the index still
contains only the older Python 3.10/3.11 packages; changing demo arguments or GPU drivers cannot fix that.

``.github/workflows/wheel.yml`` validates candidate wheels through this QA script. On a published GitHub
release, it builds without the CI-only local version suffix and publishes the validated wheel to PyPI.
The release tag must be ``v<VERSION>`` and match the repository's ``VERSION`` file. After publishing, it
checks both the exact release requirement and the default ``uvx isaaclab`` path against the public index.

Before enabling release publication, the PyPI ``isaaclab`` project owner must register the trusted publisher
with owner ``isaac-sim``, repository ``IsaacLab``, workflow ``wheel.yml``, and environment ``pypi``. See
`PyPI's trusted publisher setup <https://docs.pypi.org/trusted-publishers/adding-a-publisher/>`__.
The workflow requires this registration and the GitHub ``pypi`` environment; it does not use a stored API
token. Backport these changes to the release branch before tagging the release that supplies the corrected
wheel. Until that wheel is published, QA should use the candidate wheel and report the public-index failure
as unresolved.
