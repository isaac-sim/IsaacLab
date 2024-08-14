Verifying the Installation
==========================


Verifying the Isaac Sim installation
------------------------------------

Isaac Sim installed from pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Make sure that your virtual environment is activated (if applicable)


-  Check that the simulator runs as expected:

   .. code:: bash

      # note: you can pass the argument "--help" to see all arguments possible.
      isaacsim

   By default, this will launch an empty mini Kit window.

-  To run with a specific experience file, run:

   .. code:: bash

      # experience files can be absolute path, or relative path searched in isaacsim/apps or omni/apps
      isaacsim omni.isaac.sim.python.kit


.. attention::

   When running Isaac Sim for the first time, all dependent extensions will be pulled from the registry.
   This process can take upwards of 10 minutes and is required on the first run of each experience file.
   Once the extensions are pulled, consecutive runs using the same experience file will use the cached extensions.

   In addition, the first run will prompt users to accept the Nvidia Omniverse License Agreement.
   To accept the EULA, reply ``Yes`` when prompted with the below message:

   .. code:: bash

      By installing or using Isaac Sim, I agree to the terms of NVIDIA OMNIVERSE LICENSE AGREEMENT (EULA)
      in https://docs.omniverse.nvidia.com/isaacsim/latest/common/NVIDIA_Omniverse_License_Agreement.html

      Do you accept the EULA? (Yes/No): Yes


If the simulator does not run or crashes while following the above
instructions, it means that something is incorrectly configured. To
debug and troubleshoot, please check Isaac Sim
`documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`__
and the
`forums <https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_sim_forums.html>`__.


Isaac Sim installed from binaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To avoid the overhead of finding and locating the Isaac Sim installation
directory every time, we recommend exporting the following environment
variables to your terminal for the remaining of the installation instructions:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # Isaac Sim root directory
         export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.1.0"
         # Isaac Sim python executable
         export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: Isaac Sim root directory
         set ISAACSIM_PATH="C:\Users\user\AppData\Local\ov\pkg\isaac-sim-4.1.0"
         :: Isaac Sim python executable
         set ISAACSIM_PYTHON_EXE="%ISAACSIM_PATH%\python.bat"


For more information on common paths, please check the Isaac Sim
`documentation <https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_faq.html#common-path-locations>`__.


-  Check that the simulator runs as expected:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # note: you can pass the argument "--help" to see all arguments possible.
            ${ISAACSIM_PATH}/isaac-sim.sh

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: note: you can pass the argument "--help" to see all arguments possible.
            %ISAACSIM_PATH%\isaac-sim.bat


-  Check that the simulator runs from a standalone python script:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # checks that python path is set correctly
            ${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
            # checks that Isaac Sim can be launched from python
            ${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/omni.isaac.core/add_cubes.py

      .. tab-item:: :icon:`fa-brands fa-windows` Windows
         :sync: windows

         .. code:: batch

            :: checks that python path is set correctly
            %ISAACSIM_PYTHON_EXE% -c "print('Isaac Sim configuration is now complete.')"
            :: checks that Isaac Sim can be launched from python
            %ISAACSIM_PYTHON_EXE% %ISAACSIM_PATH%\standalone_examples\api\omni.isaac.core\add_cubes.py


.. caution::

   If you have been using a previous version of Isaac Sim, you need to run the following command for the *first*
   time after installation to remove all the old user data and cached variables:

   .. tab-set::

      .. tab-item:: :icon:`fa-brands fa-linux` Linux

      	.. code:: bash

      		${ISAACSIM_PATH}/isaac-sim.sh --reset-user

      .. tab-item:: :icon:`fa-brands fa-windows` Windows

         .. code:: batch

            %ISAACSIM_PATH%\isaac-sim.bat --reset-user


If the simulator does not run or crashes while following the above
instructions, it means that something is incorrectly configured. To
debug and troubleshoot, please check Isaac Sim
`documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`__
and the
`forums <https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_sim_forums.html>`__.


Verifying the Isaac Lab installation
------------------------------------

To verify that the installation was successful, run the following command from the
top of the repository:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # Option 1: Using the isaaclab.sh executable
         # note: this works for both the bundled python and the virtual environment
         ./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py

         # Option 2: Using python in your virtual environment
         python source/standalone/tutorials/00_sim/create_empty.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         :: Option 1: Using the isaaclab.bat executable
         :: note: this works for both the bundled python and the virtual environment
         isaaclab.bat -p source\standalone\tutorials\00_sim\create_empty.py

         :: Option 2: Using python in your virtual environment
         python source\standalone\tutorials\00_sim\create_empty.py


The above command should launch the simulator and display a window with a black
ground plane. You can exit the script by pressing ``Ctrl+C`` on your terminal.
On Windows machines, please terminate the process from Command Prompt using
``Ctrl+Break`` or ``Ctrl+fn+B``.

If you see this, then the installation was successful! |:tada:|
