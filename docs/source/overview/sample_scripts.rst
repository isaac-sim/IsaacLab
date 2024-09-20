Showroom Demos
==============

The main core interface extension in Isaac Lab ``omni.isaac.lab`` provides
the main modules for actuators, objects, robots and sensors. We provide
a list of demo scripts and tutorials. These showcase how to use the provided
interfaces within a code in a minimal way.

A few quick showroom scripts to run and checkout:

-  Spawn different quadrupeds and make robots stand using position commands:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p source/standalone/demos/quadrupeds.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p source\standalone\demos\quadrupeds.py

-  Spawn different arms and apply random joint position commands:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p source/standalone/demos/arms.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p source\standalone\demos\arms.py

-  Spawn different hands and command them to open and close:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p source/standalone/demos/hands.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p source\standalone\demos\hands.py

-  Spawn procedurally generated terrains with different configurations:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p source/standalone/demos/procedural_terrain.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p source\standalone\demos\procedural_terrain.py

-  Spawn different deformable (soft) bodies and let them fall from a height:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p source/standalone/demos/deformables.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p source\standalone\demos\deformables.py

-  Spawn multiple markers that are useful for visualizations:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p source/standalone/demos/markers.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p source\standalone\demos\markers.py
