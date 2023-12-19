Running an example with Docker
==============================

From the root of the ``orbit`` repository,  the ``docker`` directory contains all the Docker relevant files. These include the three files
(**Dockerfile**, **docker-compose.yaml**, **.env**) which are used by Docker, and an additional script that we use to interface with them,
**container.sh**.

In this tutorial, we will learn how to use the Orbit Docker container for development. For a detailed description of the Docker setup,
including installation and obtaining access to an Isaac Sim image, please reference the :ref:`deployment-docker`. For a description
of Docker in general, please refer to `their official documentation <https://docs.docker.com/get-started/overview/>`_.


Building the Container
~~~~~~~~~~~~~~~~~~~~~~

To build the Orbit container from the root of the Orbit repository, we will run the following:


.. code-block:: console

   ./docker/container.sh start


The terminal will first pull the base IsaacSim image, build the Orbit image's additional layers on top of it, and run the Orbit container.
This should take several minutes upon the first build but will be shorter in subsequent runs as Docker's caching prevents repeated work.
If we run the command ``docker container ls`` on the terminal, the output will list the containers that are running on the system. If
everything has been set up correctly, a container with the ``NAME`` **orbit** should appear, similar to below:


.. code-block:: console

   CONTAINER ID   IMAGE         COMMAND   CREATED           STATUS         PORTS     NAMES
   483d1d5e2def   orbit         "bash"    30 seconds ago   Up 30 seconds             orbit


Once the container is up and running, we can enter it from our terminal.

.. code-block:: console

   ./docker/container.sh enter


On entering the Orbit container, we are in the terminal as the superuser, ``root``. This environment contains a copy of the
Orbit repository, but also has access to the directories and libraries of Isaac Sim. We can run experiments from this environment
using a few convenient aliases that have been put into the ``root`` **.bashrc**. For instance, we have made the **orbit.sh** script
usable from anywhere by typing its alias ``orbit``.

Additionally in the container, we have `bind mounted`_ the ``orbit/source`` directory from the
host machine. This means that if we modify files under this directory from an editor on the host machine, the changes are
reflected immediately within the container without requiring us to rebuild the Docker image.

We will now run a sample script from within the container to demonstrate how to extract artifacts
from the Orbit Docker container.

The Code
~~~~~~~~

The tutorial corresponds to the ``log_time.py`` script in the ``orbit/source/standalone/tutorials/00_sim`` directory.

.. dropdown:: Code for log_time.py
   :icon: code

   .. literalinclude:: ../../../source/standalone/tutorials/00_sim/log_time.py
      :language: python
      :emphasize-lines: 46-55, 72-79
      :linenos:


The Code Explained
~~~~~~~~~~~~~~~~~~

The Orbit Docker container has several `volumes`_ to facilitate persistent storage between the host computer and the
container. One such volume is the ``/workspace/orbit/logs`` directory.
The ``log_time.py`` script designates this directory as the location to which a ``log.txt`` should be written:

.. literalinclude:: ../../../source/standalone/tutorials/00_sim/log_time.py
   :language: python
   :start-at: # Specify that the logs must be in logs/docker_tutorial
   :end-at: print(f"[INFO] Logging experiment to directory: {log_dir_path}")


As the comments note, :func:`os.path.abspath()` will prepend ``/workspace/orbit`` because in
the Docker container all python execution is done through ``/workspace/orbit/orbit.sh``.
The output will be a file, ``log.txt``, with the ``sim_time`` written on a newline at every simulation step:

.. literalinclude:: ../../../source/standalone/tutorials/00_sim/log_time.py
   :language: python
   :start-at: # Prepare to count sim_time
   :end-at: sim_time += sim_dt


Executing the Script
~~~~~~~~~~~~~~~~~~~~

We will execute the script to produce a log, adding a ``--headless`` flag to our execution to prevent a GUI:

.. code-block:: bash

  orbit -p source/standalone/tutorials/00_sim/log_time.py --headless


Now ``log.txt`` will have been produced at ``/workspace/orbit/logs/docker_tutorial``. If we exit the container
by typing ``exit``, we will return to ``orbit/docker`` in our host terminal environment. We can then enter
the following command to retrieve our logs from the Docker container and put them on our host machine:

.. code-block:: console

  ./container.sh copy


We will see a terminal readout reporting the artifacts we have retrieved from the container. If we navigate to
``/orbit/docker/artifacts/logs/docker_tutorial``, we will see a copy of the ``log.txt`` file which was produced
by the script above.

Each of the directories under ``artifacts`` corresponds to Docker `volumes`_ mapped to directories
within the container and the ``container.sh copy`` command copies them from those `volumes`_ to these directories.

We could return to the Orbit Docker terminal environment by running ``container.sh enter`` again,
but we have retrieved our logs and wish to go inspect them. We can stop the Orbit Docker container with the following command:

.. code-block:: console

  ./container.sh stop


This will bring down the Docker Orbit container. The image will persist and remain available for further use, as will
the contents of any `volumes`_. If we wish to free up the disk space taken by the image, (~20.1GB), and do not mind repeating
the build process when we next run ``./container.sh start``, we may enter the following command to delete the **orbit** image:

.. code-block:: console

  docker image rm orbit

A subsequent run of ``docker image ls``` will show that the image tagged **orbit** is now gone. We can repeat the process for the
underlying NVIDIA container if we wish to free up more space. If a more powerful method of freeing resources from Docker is desired,
please consult the documentation for the `docker prune`_ commands.


.. _volumes: https://docs.docker.com/storage/volumes/
.. _bind mounted: https://docs.docker.com/storage/bind-mounts/
.. _docker prune: https://docs.docker.com/config/pruning/
