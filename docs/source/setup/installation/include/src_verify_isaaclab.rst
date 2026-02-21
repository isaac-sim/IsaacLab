Verifying the Isaac Lab installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that the installation was successful, run the following command from the
top of the repository:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # Option 1: Using the isaaclab.sh executable
         # note: this works for both the bundled python and the virtual environment
         ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

         # Option 2: Using python in your virtual environment
         python scripts/tutorials/00_sim/create_empty.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: Option 1: Using the isaaclab.bat executable
         :: note: this works for both the bundled python and the virtual environment
         isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py

         :: Option 2: Using python in your virtual environment
         python scripts\tutorials\00_sim\create_empty.py


The above command should launch the simulator and display a window with a black
viewport. You can exit the script by pressing ``Ctrl+C`` on your terminal.
On Windows machines, please terminate the process from Command Prompt using
``Ctrl+Break`` or ``Ctrl+fn+B``.

.. figure:: /source/_static/setup/verify_install.jpg
    :align: center
    :figwidth: 100%
    :alt: Simulator with a black window.


If you see this, then the installation was successful! |:tada:|

.. note::

   If you see an error ``ModuleNotFoundError: No module named 'isaacsim'``, please ensure that the virtual
   environment is activated and ``source _isaac_sim/setup_conda_env.sh`` has been executed (for uv as well).


Train a robot!
~~~~~~~~~~~~~~

You can now use Isaac Lab to train a robot through Reinforcement Learning! The quickest way to use Isaac Lab is through the predefined workflows using one of our **Batteries-included** robot tasks. Execute the following command to quickly train an ant to walk!
We recommend adding ``--headless`` for faster training.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless

... Or a robot dog!

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless

Isaac Lab provides the tools you'll need to create your own **Tasks** and **Workflows** for whatever your project needs may be.
Take a look at our :ref:`how-to` guides like :ref:`Adding your own learning Library <how-to-add-library>` or :ref:`Wrapping Environments <how-to-env-wrappers>` for details.

.. figure:: /source/_static/setup/isaac_ants_example.jpg
    :align: center
    :figwidth: 100%
    :alt: Idle hands...
