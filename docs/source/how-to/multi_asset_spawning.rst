Spawning Multiple Assets
========================

.. currentmodule:: omni.isaac.lab

This guide describes how to create different assets within each environment using spawning functions that allow for random selection of assets.
The approach is useful to create diversity in your simulations. The sample script ``multi_asset.py`` is used as a reference, located in the ``IsaacLab/source/standalone/demos`` directory.

.. dropdown:: Code for multi_asset.py
   :icon: code

   .. literalinclude:: ../../../source/standalone/demos/multi_asset.py
      :language: python
      :linenos:

This script creates multiple environments, where each environment has a rigid object that is either a cone, a cube, or a sphere,
and an articulation that is either the ANYmal-C or ANYmal-D robot.

.. image:: ../_static/demos/multi_asset.jpg
  :width: 100%
  :alt: result of multi_asset.py

Using Multi-Asset Spawning Functions
------------------------------------

It is possible to spawn different assets and USDs in each environment using the spawners :class:`~sim.spawners.wrappers.MultiAssetSpawnerCfg` and :class:`~sim_utils.MultiUsdFileCfg`:

* We set the spawn configuration in :class:`~RigidObjectCfg` to be :class:`~sim.spawners.wrappers.MultiAssetSpawnerCfg`:

  .. literalinclude:: ../../../source/standalone/demos/multi_asset.py
     :language: python
     :lines: 99-125
     :dedent:

  This function allows you to define a list of different assets that can be spawned as rigid objects.
  When ``random_choice=True`` is set, one asset from the list is randomly selected each time an object is spawned.

* We set the spawn configuration in :class:`~ArticulationCfg` to be :class:`~sim_utils.MultiUsdFileCfg`:

  .. literalinclude:: ../../../source/standalone/demos/multi_asset.py
     :language: python
     :lines: 128-161
     :dedent:

  This configuration allows the selection of different USD files representing articulated assets.
  With ``random_choice=True``, the spawning process picks one of the specified assets at random. In the following scenario,
  it is important to note that the articulation must have the same structure (same links, joints, names, etc.) across all USD files.
  The purpose is to enable the user to create randomized versions of the same asset, for example with different link lengths.



Executing the Simulation
------------------------

To execute the script with multiple environments and randomized assets, use the following command:

.. code-block:: bash

  ./isaaclab.sh -p source/standalone/demos/multi_asset.py --num_envs 2048

This command runs the simulation with 2048 environments, each with randomly selected assets.

Stopping the Simulation
-----------------------

To stop the simulation, use the following command in the terminal:

.. code-block:: bash

  Ctrl+C

This safely terminates the simulation.
