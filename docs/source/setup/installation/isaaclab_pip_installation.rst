Installation using Isaac Lab Pip Packages
=========================================

From Isaac Lab 2.0, pip packages are provided to install both Isaac Sim and Isaac Lab extensions from pip.
Note that this installation process is only recommended for advanced users working on additional extension projects
that are built on top of Isaac Lab. Isaac Lab pip packages **does not** include any standalone python scripts for
training, inferencing, or running standalone workflows such as demos and examples. Therefore, users are required
to define their own runner scripts when installing Isaac Lab from pip.

To learn about how to set up your own project on top of Isaac Lab, please see :ref:`template-generator`.

.. note::

   Currently, we only provide pip packages for every major release of Isaac Lab.
   For example, we provide the pip package for release 2.1.0 and 2.2.0, but not 2.1.1.
   In the future, we will provide pip packages for every minor release of Isaac Lab.

.. include:: include/pip_python_virtual_env.rst

Installing dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   In case you used UV to create your virtual environment, please replace ``pip`` with ``uv pip``
   in the following commands.

-  Install a CUDA-enabled PyTorch 2.7.0 build for CUDA 12.8:

   .. code-block:: none

      pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

-  If you want to use ``rl_games`` for training and inferencing, install the
   its Python 3.11 enabled fork:

   .. code-block:: none

      pip install git+https://github.com/isaac-sim/rl_games.git@python3.11

-  Install the Isaac Lab packages along with Isaac Sim:

   .. code-block:: none

      pip install isaaclab[isaacsim,all]==2.2.0 --extra-index-url https://pypi.nvidia.com


.. include:: include/pip_verify_isaacsim.rst

Running Isaac Lab Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~

By following the above scripts, your Python environment should now have access to all of the Isaac Lab extensions.
To run a user-defined script for Isaac Lab, simply run

.. code:: bash

    python my_awesome_script.py

Generating VS Code Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Due to the structure resulting from the installation, VS Code IntelliSense (code completion, parameter info
and member lists, etc.) will not work by default. To set it up (define the search paths for import resolution,
the path to the default Python interpreter, and other settings), for a given workspace folder,
run the following command:

.. code-block:: bash

   python -m isaaclab --generate-vscode-settings


.. warning::

   The command will generate a ``.vscode/settings.json`` file in the workspace folder.
   If the file already exists, it will be overwritten (a confirmation prompt will be shown first).
