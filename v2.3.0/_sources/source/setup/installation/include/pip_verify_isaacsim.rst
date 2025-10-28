
Verifying the Isaac Sim installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Make sure that your virtual environment is activated (if applicable)

-  Check that the simulator runs as expected:

   .. code:: bash

      # note: you can pass the argument "--help" to see all arguments possible.
      isaacsim

-  It's also possible to run with a specific experience file, run:

   .. code:: bash

      # experience files can be absolute path, or relative path searched in isaacsim/apps or omni/apps
      isaacsim isaacsim.exp.full.kit


.. note::

   When running Isaac Sim for the first time, all dependent extensions will be pulled from the registry.
   This process can take upwards of 10 minutes and is required on the first run of each experience file.
   Once the extensions are pulled, consecutive runs using the same experience file will use the cached extensions.

.. attention::

   The first run will prompt users to accept the Nvidia Omniverse License Agreement.
   To accept the EULA, reply ``Yes`` when prompted with the below message:

   .. code:: bash

      By installing or using Isaac Sim, I agree to the terms of NVIDIA OMNIVERSE LICENSE AGREEMENT (EULA)
      in https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html

      Do you accept the EULA? (Yes/No): Yes


If the simulator does not run or crashes while following the above
instructions, it means that something is incorrectly configured. To
debug and troubleshoot, please check Isaac Sim
`documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`__
and the
`Isaac Sim Forums <https://docs.isaacsim.omniverse.nvidia.com/latest/common/feedback.html>`_.
