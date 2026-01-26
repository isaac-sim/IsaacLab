.. _setup-devcontainer:

Using VS Code Dev Containers
=============================

.. attention::

   This is optional. You do not need to use Dev Containers to develop Isaac Lab.

`Visual Studio Code Dev Containers <https://code.visualstudio.com/docs/devcontainers/containers>`_ provide a consistent,
reproducible development environment by running your workspace inside a Docker container. This is particularly useful for
Isaac Lab development as it encapsulates all dependencies, including Isaac Sim, NVIDIA drivers, and CUDA libraries.

.. note::

   Dev Containers leverage the Docker infrastructure already provided by Isaac Lab. For more information on
   Isaac Lab's Docker setup, please refer to the :ref:`Docker Guide <deployment-docker>`.


Prerequisites
-------------

Before using Dev Containers with Isaac Lab, ensure you have the following installed:

1. **Docker Engine** (version 26.0.0 or newer) and **Docker Compose** (version 2.25.0 or newer)

   * Follow the `Docker installation guide <https://docs.docker.com/engine/install/>`_
   * Complete the `post-installation steps <https://docs.docker.com/engine/install/linux-postinstall/>`_ to run Docker without ``sudo``

2. **NVIDIA Container Toolkit** for GPU acceleration

   * Follow the `Container Toolkit installation guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_

3. **Visual Studio Code** with the **Dev Containers extension**

   * Download `VS Code <https://code.visualstudio.com/>`_
   * Install the `Dev Containers extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_

.. caution::

   By using Isaac Lab's Docker containers, you are implicitly agreeing to the
   `NVIDIA Software License Agreement <https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement>`_.
   If you do not agree to the EULA, do not use this container.


Understanding Dev Containers
-----------------------------

Dev Containers allow you to:

* **Consistent Environment**: Every developer works with the same dependencies, libraries, and tools
* **Isolated Development**: Keep your host machine clean while having all necessary tools in the container
* **GPU Access**: Use NVIDIA GPUs inside the container for Isaac Lab and training workloads
* **VS Code Integration**: Full IDE features including IntelliSense, debugging, and extensions work seamlessly inside the container

When you open a workspace in a Dev Container, VS Code:

1. Builds or pulls the specified Docker image
2. Starts a container from that image
3. Mounts your workspace into the container
4. Connects VS Code to the container
5. Installs VS Code extensions inside the container


Using the Dev Container Configuration
---------------------------------------

Isaac Lab includes a pre-configured Dev Container setup in the ``.devcontainer/`` directory at the root of the repository.
This configuration leverages Isaac Lab's existing Docker infrastructure and is ready to use out of the box.

The provided Dev Container configuration:

* Uses the ``isaac-lab-base`` service from Isaac Lab's ``docker-compose.yaml``
* Mounts the workspace at ``/workspace/isaaclab``
* Configures the Python interpreter to use Isaac Sim's Python at ``/isaac-sim/python.sh``
* Installs useful VS Code extensions for Python development (Pylance, formatting tools)
* Includes port forwarding for Jupyter (8888) and TensorBoard (6006)
* Configures Python formatting with Black (120 character line length)
* Sets up appropriate file exclusions for Python development


.. caution::

   The Dev Container runs as the ``root`` user due to Isaac Sim's Docker requirements. This means:

   * Files created inside the container will be owned by ``root``
   * Files created in mounted directories (your workspace) will also have ``root`` ownership on the host
   * You may need to use ``sudo chown`` on your host to reclaim ownership of files after working in the container

   While you can modify ``remoteUser`` in ``.devcontainer/devcontainer.json`` to use a non-root user,
   this is not recommended as it may cause Isaac Sim to fail. This is a limitation of the Isaac Sim
   Docker image, not Isaac Lab.


Customizing the Dev Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration file is located at ``.devcontainer/devcontainer.json``. You can customize it to suit your
development needs, such as adding additional VS Code extensions or modifying editor settings.

**Adding VS Code Extensions:**

.. code-block:: json

   "extensions": [
     "ms-python.python",
     "njpwerner.autodocstring",        // Add automatic docstring generation
     "streetsidesoftware.code-spell-checker"  // Add spell checking
   ]

**Adding Port Forwarding:**

.. code-block:: json

   "forwardPorts": [8888, 6006, 5000],  // Add additional ports as needed

**For ROS2 Development:**

Change the service to use the ROS2-enabled container:

.. code-block:: json

   "service": "isaac-lab-ros2"


Opening the Dev Container
--------------------------

To start developing in the Dev Container:

1. Open the Isaac Lab folder in VS Code
2. Press ``Ctrl+Shift+P`` (or ``Cmd+Shift+P`` on macOS) to open the command palette
3. Select ``Dev Containers: Reopen in Container``

   .. image:: ../../_static/vscode_tasks.png
      :width: 600px
      :align: center
      :alt: VS Code Command Palette

4. VS Code will build the container (if needed) and reopen the workspace inside it
5. Wait for the container to start and extensions to install

.. note::

   The first time you open the Dev Container, it will build the Docker image, which may take 10-30 minutes
   depending on your internet connection and system resources. Subsequent launches will be much faster.


Working Inside the Dev Container
---------------------------------

Once inside the Dev Container, you can:

* **Run scripts**: Use the integrated terminal to run Isaac Lab scripts

  .. code-block:: bash

     ./isaaclab.sh -p scripts/tutorials/00_sim/log_time.py

* **Debug code**: Set breakpoints and use VS Code's debugger with the Python interpreter
* **Edit code**: All changes are reflected in real-time due to mounted volumes
* **Run GUI applications**: Isaac Sim GUI works with X11 forwarding enabled

.. tip::

   The terminal inside VS Code is running inside the container. Any commands you run will use the
   container's environment, Python interpreter, and installed packages.


Python Interpreter Configuration
---------------------------------

The Dev Container is configured to use Isaac Sim's Python interpreter located at ``/isaac-sim/python.sh``.
This ensures compatibility with Isaac Sim's dependencies and extensions.

To verify the correct interpreter is selected:

1. Open the command palette (``Ctrl+Shift+P``)
2. Select ``Python: Select Interpreter``
3. Choose ``/isaac-sim/python.sh``

The Python interpreter path is pre-configured in the ``devcontainer.json`` file, so this should be automatic.


Using ROS2 with Dev Containers
-------------------------------

If you need ROS2 support, modify the ``.devcontainer/devcontainer.json`` file to use the ``isaac-lab-ros2`` service
instead of ``isaac-lab-base``:

.. code-block:: json

   "service": "isaac-lab-ros2"

Then reopen the container using the command palette (``Ctrl+Shift+P`` > ``Dev Containers: Rebuild Container``).
ROS2 will be sourced automatically in the terminal.


Troubleshooting
---------------

Container won't start
~~~~~~~~~~~~~~~~~~~~~

* Ensure Docker daemon is running: ``systemctl status docker``
* Check you have NVIDIA Container Toolkit installed: ``nvidia-docker --version``
* Verify GPU access: ``docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi``

X11 forwarding not working
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* On your host machine, run: ``xhost +local:docker``
* Ensure the ``DISPLAY`` environment variable is set correctly
* Check that ``.Xauthority`` is mounted correctly in the Dev Container configuration

Python interpreter issues
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Verify the interpreter path is correct: ``which python`` should return ``/isaac-sim/python.sh``
* Reload the VS Code window: ``Ctrl+Shift+P`` > ``Developer: Reload Window``
* Check Python extension logs: ``Ctrl+Shift+P`` > ``Python: Show Output``

Build errors
~~~~~~~~~~~~

* Ensure all Docker prerequisites are met (see :ref:`Docker Guide <deployment-docker>`)
* Try rebuilding the container: ``Ctrl+Shift+P`` > ``Dev Containers: Rebuild Container``
* Check Docker logs: ``docker logs isaac-lab-base``


Performance Considerations
--------------------------

* **First build**: The initial container build can take significant time. Subsequent starts are much faster.
* **Named volumes**: Isaac Lab uses named volumes for caches (Kit, pip, GL, compute). These persist between
  container rebuilds to speed up subsequent launches.
* **File synchronization**: Since source files are mounted from the host, changes are immediately visible
  both inside and outside the container.


Further Reading
---------------

For more information on Dev Containers and Isaac Lab's Docker infrastructure:

* `VS Code Dev Containers documentation <https://code.visualstudio.com/docs/devcontainers/containers>`_
* `Dev Container specification <https://containers.dev/>`_
* :ref:`Isaac Lab Docker Guide <deployment-docker>`
* :ref:`Setting up Visual Studio Code <setup-vs-code>`
* `Isaac Sim Container Installation <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html>`_
