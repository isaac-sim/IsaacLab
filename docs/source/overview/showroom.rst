Showroom Demos
==============

The main core interface extension in Isaac Lab ``isaaclab`` provides
the main modules for actuators, objects, robots and sensors. We provide
a list of demo scripts and tutorials. These showcase how to use the provided
interfaces within a code in a minimal way.

A few quick showroom scripts to run and checkout:

.. rst-class:: showroom-demo-list

-  Spawn different arms and apply random joint position commands:

   **Physics:** ``isaacsim_physx``, ``newton_mjwarp``

   **Visualizer:** ``none``, ``kit``, ``newton``, ``rerun``, ``viser``

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/arms.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/arms.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\arms.py

   .. image:: ../_static/demos/arms.jpg
      :width: 100%
      :alt: Arms in Isaac Lab


-  Spawn different biped robots:

   **Physics:** ``isaacsim_physx``, ``newton_mjwarp``

   **Visualizer:** ``none``, ``kit``, ``newton``, ``rerun``, ``viser``

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/bipeds.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/bipeds.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\bipeds.py

   .. image:: ../_static/demos/bipeds.jpg
      :width: 100%
      :alt: Biped robots in Isaac Lab


-  Spawn different deformable objects and let them fall from a height:

   **Physics:** ``isaacsim_physx``, ``newton_vbd``

   **Visualizer:** ``none``, ``kit`` for either physics backend; ``newton``,
   ``rerun``, and ``viser`` with Newton VBD only

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/deformables.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/deformables.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\deformables.py

   .. image:: ../_static/demos/deformables.jpg
      :width: 100%
      :alt: Deformable primitive-shaped objects in Isaac Lab


-  Interactive inference of trained H1 rough terrain locomotion policy:

   **Physics:** ``isaacsim_physx`` only

   **Visualizer:** ``kit`` only

   This demo downloads a policy and requires interactive input.

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/h1_locomotion.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/h1_locomotion.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\h1_locomotion.py

   .. image:: ../_static/demos/h1_locomotion.jpg
      :width: 100%
      :alt: H1 locomotion in Isaac Lab

   This is an interactive demo that can be run using the mouse and keyboard.
   To enter third-person perspective, click on a humanoid character in the scene.
   Once entered into third-person view, the humanoid can be controlled by keyboard using:

   * ``UP``: go forward
   * ``LEFT``: turn left
   * ``RIGHT``: turn right
   * ``DOWN``: stop
   * ``C``: switch between third-person and perspective views
   * ``ESC``: exit current third-person view

   If a misclick happens outside of the humanoid bodies when selecting a humanoid,
   a message is printed to console indicating the error, such as
   ``The selected prim was not a H1 robot`` or
   ``Multiple prims are selected. Please only select one!``.


-  Spawn different hands and command them to open and close:

   **Physics:** ``isaacsim_physx``, ``newton_mjwarp``

   **Visualizer:** ``none``, ``kit``, ``newton``, ``rerun``, ``viser``

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/hands.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/hands.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\hands.py

   .. image:: ../_static/demos/hands.jpg
      :width: 100%
      :alt: Dexterous hands in Isaac Lab


-  Define multiple markers that are useful for visualizations:

   **Physics:** ``isaacsim_physx`` only

   **Visualizer:** ``none``, ``kit``, ``newton``, ``rerun``, ``viser``

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/markers.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/markers.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\markers.py

   .. image:: ../_static/demos/markers.jpg
      :width: 100%
      :alt: Markers in Isaac Lab


-  Use the interactive scene and spawn varying assets in individual environments:

   **Physics:** ``isaacsim_physx``, ``newton_mjwarp``

   **Visualizer:** ``none``, ``kit``, ``newton``, ``rerun``, ``viser``

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/multi_asset.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/multi_asset.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\multi_asset.py

   .. image:: ../_static/demos/multi_asset.jpg
      :width: 100%
      :alt: Multiple assets managed through the same simulation handles


-  Compose task scenes into one heterogeneous simulation using clone combinations:

   **Physics:** ``isaacsim_physx`` only

   **Visualizer:** ``none``, ``kit``, ``newton``, ``rerun``, ``viser``

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            ./isaaclab.sh -p scripts/demos/heterogeneous_scene.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\heterogeneous_scene.py

   .. image:: ../_static/demos/heterogeneous_scene.jpg
      :width: 100%
      :alt: Task scenes composed into one heterogeneous cloned simulation

   The demo resolves a curated selection of PhysX task scenes that share a flat
   floor at height zero, folds them into a single scene with
   :func:`~isaaclab.scene.add`, and clones the combined scene so each
   environment hosts one task's assets. Use ``--num_task`` and ``--num_envs``
   to run a smaller composition.


-  Use the RigidObjectCollection spawn and view manipulation to demonstrate bin-packing example:

   **Physics:** ``isaacsim_physx``, ``newton_mjwarp``

   **Visualizer:** ``none``, ``kit``, ``newton``, ``rerun``, ``viser``

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/bin_packing.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/bin_packing.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\bin_packing.py

   .. image:: ../_static/demos/bin_packing.jpg
      :width: 100%
      :alt: Spawning random number of random asset per env_id using combination of MultiAssetSpawner and RigidObjectCollection



-  Use the interactive scene and spawn a simple parallel robot for pick and place:

   **Physics:** ``isaacsim_physx`` only

   **Visualizer:** ``kit`` only

   This demo requires interactive input and uses the CPU-only PhysX surface
   gripper. Newton physics is not supported.

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/pick_and_place.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/pick_and_place.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\pick_and_place.py

   .. image:: ../_static/demos/pick_and_place.jpg
      :width: 100%
      :alt: User controlled pick and place with a parallel robot

   This is an interactive demo that can be run using the mouse and keyboard.
   Your goal is pick up the purple cube and to drop it on the red sphere!
   Use the following controls to interact with the simulation:

   * Hold the ``A`` key to have the gripper track the cube position.
   * Hold the ``D`` key to have the gripper track the target position
   * Press the ``W`` or ``S`` keys to move the gantry UP or DOWN respectively
   * Press ``Q`` or ``E`` to OPEN or CLOSE the gripper respectively



-  Teleoperate a Franka Panda robot using Haply haptic device with force feedback:

   **Physics:** ``isaacsim_physx``, ``newton_mjwarp``

   **Visualizer:** ``none``, ``kit``, ``newton``, ``rerun``, ``viser``

   Haply hardware is required.

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/haply_teleoperation.py --websocket_uri ws://localhost:10001 --pos_sensitivity 1.65

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/haply_teleoperation.py --websocket_uri ws://localhost:10001 --pos_sensitivity 1.65

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\haply_teleoperation.py --websocket_uri ws://localhost:10001 --pos_sensitivity 1.65

   .. image:: ../_static/demos/haply_teleop_franka.jpg
      :width: 100%
      :alt: Haply teleoperation with force feedback

   This demo requires Haply Inverse3 and VerseGrip devices.
   The goal of this demo is to pick up the cube or touch it with the end-effector.
   The Haply devices provide:

   * 3 dimensional position tracking for end-effector control
   * Directional force feedback for contact sensing
   * Button inputs for gripper and end-effector rotation control

   See :ref:`haply-teleoperation` for detailed setup instructions.



-  Create and spawn procedurally generated terrains with different configurations:

   **Physics:** ``isaacsim_physx`` only

   **Visualizer:** ``none``, ``kit``, ``newton``, ``rerun``, ``viser``

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/procedural_terrain.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/procedural_terrain.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\procedural_terrain.py

   .. image:: ../_static/demos/procedural_terrain.jpg
      :width: 100%
      :alt: Procedural Terrains in Isaac Lab



-  Spawn a quadcopter in the default environment:

   **Physics:** ``isaacsim_physx``, ``newton_mjwarp``

   **Visualizer:** ``none``, ``kit``, ``newton``, ``rerun``, ``viser``

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/quadcopter.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/quadcopter.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\quadcopter.py

   .. image:: ../_static/demos/quadcopter.jpg
      :width: 100%
      :alt: Quadcopter in Isaac Lab


-  Spawn different quadrupeds and make robots stand using position commands:

   **Physics:** ``isaacsim_physx``, ``newton_mjwarp``

   **Visualizer:** ``none``, ``kit``, ``newton``, ``rerun``, ``viser``

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/quadrupeds.py

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/quadrupeds.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\quadrupeds.py

   .. image:: ../_static/demos/quadrupeds.jpg
      :width: 100%
      :alt: Quadrupeds in Isaac Lab


-  Spawn a multi-mesh ray caster that uses Warp kernels for raycasting

   **Physics:** ``isaacsim_physx``, ``newton_mjwarp``

   **Visualizer:** ``none``, ``newton``, ``rerun``, ``viser`` with either physics
   backend; ``kit`` with PhysX only

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. tab-set::

            .. tab-item:: uv (Recommended)

               .. code:: bash

                  uv run python scripts/demos/sensors/multi_mesh_raycaster.py --num_envs 16 --asset_type objects

            .. tab-item:: isaaclab.sh / isaaclab.bat

               .. code:: bash

                  ./isaaclab.sh -p scripts/demos/sensors/multi_mesh_raycaster.py --num_envs 16 --asset_type objects

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            isaaclab.bat -p scripts\demos\sensors\multi_mesh_raycaster.py --num_envs 16 --asset_type objects

   .. image:: ../_static/demos/multi-mesh-raycast.jpg
      :width: 100%
      :alt: Multi-mesh raycaster in Isaac Lab
