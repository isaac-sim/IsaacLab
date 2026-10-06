.. Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
.. All rights reserved.
..
.. SPDX-License-Identifier: BSD-3-Clause

.. _interactive-documentation-examples:

Adding Interactive Documentation Examples
=========================================

Browser examples let readers change a small scene's inputs beside the guidance
they are learning. The existing examples run compiled Newton physics on the
browser CPU, render through Three.js, and load only when visible. Their sources
and rebuild commands live in ``docs/browser_demos/``.

Choose the execution mode
-------------------------

Use a **local WebAssembly bundle** for a compact Newton scene with fixed buffer
sizes and an exportable CPU step. Readers need no Python process or GPU service.
The current bundles illustrate selected behaviors; they do not export complete
Isaac Lab applications, training workflows, or every component of an environment.

For a complete environment that needs native physics, Python controllers,
learned actuators, large scenes, or GPU rendering, use a **running Isaac Lab
session with a browser viewer**. The existing :doc:`Viser visualizer
</source/concepts/visualization>` keeps simulation and controllers in Python
and provides a browser interface with pause and reset controls. For example,
on a machine with Isaac Lab and a compatible checkpoint:

.. code-block:: bash

   uv run --extra viser isaaclab play --rl_library rsl_rl --task Isaac-Cartpole \
       --checkpoint latest --num_envs 1 --viz viser

This runs a native session and prints its viewer URL. Hosting that session for
documentation readers requires compute, session management, and any
example-specific command controls. Viser's view does not reproduce RTX sensor
images; image streaming requires an appropriate native rendering/streaming path.

Embed a checked bundle
----------------------

Put a compiled bundle under
``docs/source/_static/browser_demos/<name>/``. Its ``manifest.json`` names the
JavaScript module, WebAssembly binary, bindings, parameters, and visualization.
Policy weights, robot meshes, and licenses stay beside that manifest.

Place the widget in the concept or tutorial that explains it:

.. code-block:: rst

   .. isaaclab-browser-demo:: stiffness
      :title: VBD material tuning

The optional title changes the heading. The directive checks the manifest's
bundle/runtime ABI versions and required module, binary, policy, and mesh files, calculates paths
for the HTML builder, and loads the shared stylesheet and module once per page.
Missing bundles fail a documentation build with warnings treated as errors.
Non-HTML builders omit the widget; keep the lesson and native commands in the
surrounding text. The standalone ``preview.html?demo=<name>`` page is for reviewing
a bundle. Keep the teaching examples beside their relevant guides.

Prepare the export boundary
---------------------------

1. **Reuse the native definition.** Read the source scene/configuration and
   resolve one supported Newton preset. Reuse geometry, masses, joint order,
   time step, substeps, material values, and actuator semantics. Build assets
   before capture. Document how a small browser scene relates to the native example.
2. **Capture a complete repeatable CPU step.** Record controls, collision,
   solver updates, and state copies with ``wp.ScopedCapture(device="cpu", apic=True)``.
   Include every operation needed on replay. Array capacities and topology are
   fixed. Reset must restore physical and controller state, including plastic
   history and recurrent inference state when present. Warp capture records
   `supported operations <https://nvidia.github.io/warp/v1.17/user_guide/runtime.html#graphs>`_,
   so arbitrary Python or Torch execution needs a separate port.
3. **Expose inputs and outputs.** Use Newton Web's named bindings and parameters.
   Add controls to the shared widget and reuse an existing viewer where possible.
   The manifest's ``isaacLabDemo.kind`` selects the current viewer/control path.
   A new kind needs a reviewed runtime implementation; adding a manifest alone
   does not implement its controller or rendering.
4. **Package assets separately.** Native USD/URDF import runs at build time.
   Package browser geometry, policy weights, and their licenses. Match checkpoint
   observations, action order/scaling, control frequency, and joint mappings.
   Source hashes in the existing exporters guard those contracts.
5. **Validate behavior.** Replay the same commands on native CPU and WASM and
   compare trajectories with stated tolerances. Check contacts, control limits,
   pause, visibility, and complete reset. Record device, step time, download size,
   and differences from the original example. A finite trajectory alone does not
   establish policy or physical fidelity.

The ``export_stiffness`` and ``export_rigid_friction`` functions in
``docs/browser_demos/export.py`` are small capture examples. The policy examples
show asset/checkpoint mapping. MPM currently includes adaptations to private
Newton and Warp interfaces; review them when either dependency changes.

Current bundles and manual adaptations
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Bundle
     - Source or reference
     - Work outside the shared capture/compiler
   * - ``stiffness``
     - VBD volume tuning guidance
     - Construct the tetrahedral comparison and bind material/gravity controls.
   * - ``cloth_bending``
     - ``examples/deformables.py`` and the documented bending comparison
     - Construct supported sheets; package roller visuals and bending/gravity controls.
   * - ``rigid_friction``
     - MJWarp contact tuning guidance
     - Construct the incline comparison and bind the middle box's friction.
   * - ``joint_pd``
     - Actuator guide and the PD tutorial proposal
     - Construct the teaching arm; provide step/sine commands and tracking plots.
   * - ``cartpole``
     - ``Isaac-Cartpole-Direct`` Newton configuration and checkpoint
     - Map asset joints, observations, actor weights, force input, and episode reset.
   * - ``g1``
     - WBC-AGILE ``Velocity-G1-v0`` and Unitree URDF
     - Map the external policy and package visuals; retain the reference's contact selection.
   * - ``anymal``
     - ``Isaac-Velocity-Flat-AnymalD`` configuration and checkpoint
     - Package visuals and actor inference. Replacing the LSTM actuator with PD changes dynamics.
   * - ``franka_reach``, ``franka_lift``, ``franka_drawer``
     - Native core task configurations and published Newton RSL-RL checkpoints
     - Port action/observation packing, lift history/normalization, and fixed episode reset.
       Reuse native scene and actuators; reduce lift's batched contact capacity for one world.
       Share Franka visuals and the policy viewer. Target controls use the robot's root frame.
   * - ``mpm``
     - Newton implicit MPM and the granular material lesson
     - Analytic boundaries, a scalar strain-basis path, and allocation-alias capture handling.
       The coarse-grid materials are qualitative references.

Coverage of the remaining examples
----------------------------------

These groups cover the packaged ``examples/`` programs, the
``scripts/tutorials/`` families, and task families presented in the docs.
They are source-review assessments; their WASM exports have not been validated.

.. list-table::
   :header-rows: 1
   :widths: 29 28 43

   * - Documentation/example family
     - Likely execution mode
     - Remaining manual work or blocker
   * - Rigid objects, articulations, cube/Cartpole tutorials; dominoes and bin packing
     - Compact Newton WASM candidates
     - Share native scene setup, package assets, and port command/reset logic.
       Bound scene and contact sizes for browser CPU performance.
   * - Deformables, cables, block and tackle
     - Compact VBD WASM candidates
     - Share volume/cloth/cable setup; expose handles/stiffness and package pulley/cable visuals.
       Preserve required self-contact and validate contact behavior.
   * - Granular MPM, two-way coupling, snowball smash, teapot fill
     - Small WASM lessons; native session for full showcases
     - Portable mesh/BVH state, proxy coupling, complete history, and particle/grid budgets.
       Surface reconstruction has separate native/GPU dependencies.
   * - Zoo, multi-asset, heterogeneous scenes, ARL robot, procedural terrain
     - Native session initially; selected small scenes in WASM
     - Asset selection/packaging, scene size, terrain/BVH serialization, and control loops.
       Resolve each selected physics backend explicitly.
   * - Other locomotion/manipulation policies; Ant, Humanoid, Allegro, and reaching robots
     - Native session initially; WASM subsets after actuator/controller work
     - Match actor contracts, recurrent state, actuator limits/delays, resets/events,
       and sensors. Warp frontend coverage alone is insufficient.
   * - IK/OSC controllers and pick-and-place/surface-gripper tutorials
     - Native session initially
     - Port controller math and actuator state; review attachment/gripper backend APIs.
   * - Contact, frame-transform, IMU, and PVA sensors
     - Compact WASM candidates after sensor adapters
     - Export matching frames, timing, filtering/history, and reset state.
       Reuse native math and measured quantities.
   * - Newton and multi-mesh ray casting; height scans
     - Native session until portable queries are available
     - Serialize/rebuild mesh/BVH resources; captured native handles cannot be reused in WASM.
   * - Cameras, PPISP, TacSL, tiled-camera/recording examples; vision policies
     - Native rendering/session
     - Preserve renderer/sensor semantics and image inference. Three.js display geometry does
       not reproduce RTX, Gaussian rendering, or tactile simulation.
   * - Markers and visual-color randomization
     - Browser visualization lessons are possible
     - Implement display inputs and explain which native rendering properties are represented.
   * - Haply teleoperation, deployment, launch, import and logging tutorials
     - Native/hardware workflow
     - Hardware, process lifecycle, file conversion, and logging are the lesson itself.

Version maintenance
-------------------

**Published bundles are self-contained.** Updating Warp in an Isaac Lab Python
environment does not change an existing WASM binary. Sphinx copies the bundles
without importing Newton Web or running the compiler.

**Rebuilds use a reviewed toolchain.** ``export.py`` checks Newton, Warp, and
MJWarp pins and the Newton Web checkout revision before capture. Newton Web
checks its Emscripten version. New exports record the checked dependencies in
``manifest.build.isaacLabExporter``; older bundles retain their existing metadata.
Use the pinned installation steps in ``docs/browser_demos/README.md``, then check:

.. code-block:: bash

   uv run --no-sync python docs/browser_demos/export.py --check-toolchain

Warp graph layouts, generated C++, CPU runtime operations, and private solver
interfaces can change on upgrades. Review the compiler and affected adapters,
then rebuild and compare representative VBD, rigid/policy, and MPM scenes.
Regenerate affected bundles after those checks pass. Keep runtime ABI changes
separate from changes to the Python toolchain.

Keep compatibility work with its owner: Warp provides capture/code generation;
Newton Web provides the WASM compiler/runtime bridge; Isaac Lab provides scene
recipes, policy/actuator contracts, and documentation. Move reusable operation
or resource support into the compiler/runtime to reduce per-example patches.

The documentation CI job runs the focused embedding checks without the
simulator test orchestrator. Run them locally with:

.. code-block:: bash

   uv run --isolated --extra dev python -m pytest --confcutdir=tools/test \
       tools/test/test_browser_docs.py

Moving more code to Warp
------------------------

The :doc:`Warp frontend </source/concepts/warp_environments>` supplies Warp
observations, actions, rewards, and resets for several task families. Scene
writes and actuator operations still cross into Torch, and CUDA capture does
not establish a complete CPU-APIC export boundary.

To reduce per-example work, share pure scene/configuration factories between
native demos and exporters. Provide reusable CPU-capable Warp actuator updates,
observations, control, and reset. Port the ANYdrive LSTM with hidden/cell state
and effort clipping to preserve native dynamics. Provide portable geometry
resource construction for queries and coupling; keep rendering interfaces explicit.

Prioritize reusable ports and native/WASM behavior comparisons over parallel
browser-only copies of every task. A complete Warp step still needs supported
CPU operations, portable resources, and a scene small enough for its target
browser. Native browser sessions remain useful beyond that boundary.
