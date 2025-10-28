Check that the simulator runs as expected:

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


Check that the simulator runs from a standalone python script:

.. tab-set::
    :sync-group: os

    .. tab-item:: :icon:`fa-brands fa-linux` Linux
        :sync: linux

        .. code:: bash

            # checks that python path is set correctly
            ${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
            # checks that Isaac Sim can be launched from python
            ${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.api/add_cubes.py

    .. tab-item:: :icon:`fa-brands fa-windows` Windows
        :sync: windows

        .. code:: batch

            :: checks that python path is set correctly
            %ISAACSIM_PYTHON_EXE% -c "print('Isaac Sim configuration is now complete.')"
            :: checks that Isaac Sim can be launched from python
            %ISAACSIM_PYTHON_EXE% %ISAACSIM_PATH%\standalone_examples\api\isaacsim.core.api\add_cubes.py

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
`Isaac Sim Forums <https://docs.isaacsim.omniverse.nvidia.com/latest/common/feedback.html>`_.
