.. _experimental-demos:

Experimental Demos
==================

The `Isaac Lab demos repository <https://github.com/isaac-sim/isaaclab_demos>`_ collects runnable demos built on Isaac
Lab, outside its core packages. Each demo pins the Isaac Lab commit it was tested with, and the repository's launcher
installs that commit the first time the demo runs.

.. attention::

   These demos are experimental, live outside Isaac Lab, and are Linux only. They need an NVIDIA GPU with a CUDA
   13-compatible driver, and about 8 GB of disk for each Isaac Lab version they pin.

Install
-------

Install Git and `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, then clone the repository and list
its demos:

.. code-block:: bash

   git clone https://github.com/isaac-sim/isaaclab_demos.git
   cd isaaclab_demos
   uv run --locked isaaclab-demos list

Run a demo by its ID, for example ``uv run --locked isaaclab-demos run coffee``. Its first run downloads and installs
the Isaac Lab commit it pins, which takes a few minutes. Arguments after the ID go to the demo; ``--help`` lists them.

Demos
-----

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Demo
     - What it shows
     - Guide
   * - ``coffee``
     - A Flexiv arm scoops coffee into a cup with a spoon; in the wet scene, the Sharpa hand stirs the mixture.
     - `Coffee preparation <https://github.com/isaac-sim/isaaclab_demos/blob/main/demos/coffee/README.md>`_
   * - ``rj45``
     - A Rizon arm with a Sharpa hand inserts an RJ45 plug into a server rack, releases it, then regrasps and
       extracts it.
     - `RJ45 plug insertion <https://github.com/isaac-sim/isaaclab_demos/blob/main/demos/rj45/README.md>`_
   * - ``water-pouring``
     - A Sharpa hand grasps a teapot by its handle, lifts it and tips simulated water into a bowl.
     - `Water pouring <https://github.com/isaac-sim/isaaclab_demos/blob/main/demos/water_pouring/README.md>`_
   * - ``conveyor-racetrack``
     - A Franka transfers four cubes between counter-rotating conveyor belts.
     - `Racetrack and warehouse
       <https://github.com/isaac-sim/isaaclab_demos/blob/main/packages/conveyor_franka/README.md>`_
   * - ``conveyor-warehouse``
     - A Franka sorts colored parcels onto two circulating warehouse conveyors.
     - `Racetrack and warehouse
       <https://github.com/isaac-sim/isaaclab_demos/blob/main/packages/conveyor_franka/README.md>`_
   * - ``conveyor-suction``
     - A Franka suction tool sorts parcels of different sizes onto receiving belts and into bins.
     - `Suction sorting <https://github.com/isaac-sim/isaaclab_demos/blob/main/demos/conveyor_suction/README.md>`_
   * - ``cube-reorient``
     - Trains the Shadow Hand camera task to turn a cube to a goal pose on images Cosmos3-Nano-Sim-Transfer
       generates from the camera's depth. It needs the Cosmos server.
     - `Cube reorientation <https://github.com/isaac-sim/isaaclab_demos/blob/main/demos/cube_reorient/README.md>`_
   * - ``apple-pick-place``
     - A GR00T N1.7 policy moves an apple onto a plate with a Unitree G1, seeing the head camera image
       Cosmos3-Nano-Sim-Transfer generates from depth. It needs a GR00T policy server and the Cosmos server.
     - `G1 apple pick-and-place
       <https://github.com/isaac-sim/isaaclab_demos/blob/main/demos/apple_pick_place/README.md>`_

Each guide explains how to install and run its demo, including what else it needs to download, such as a policy or the
Cosmos server. The repository's `README <https://github.com/isaac-sim/isaaclab_demos#install-and-run-a-demo>`_ has
every demo's run command.
