.. _deployment-cluster:


Cluster Guide
=============

Clusters are a great way to speed up training and evaluation of learning algorithms.
While the Orbit Docker image can be used to run jobs on a cluster, many clusters only
support singularity images. This is because `singularity`_ is designed for
ease-of-use on shared multi-user systems and high performance computing (HPC) environments.
It does not require root privileges to run containers and can be used to run user-defined
containers.

Singularity is compatible with all Docker images. In this section, we describe how to
convert the Orbit Docker image into a singularity image and use it to submit jobs to a cluster.

.. attention::

    Cluster setup varies across different institutions. The following instructions have been
    tested on the `ETH Zurich Euler`_ cluster, which uses the SLURM workload manager.

    The instructions may need to be adapted for other clusters. If you have successfully
    adapted the instructions for another cluster, please consider contributing to the
    documentation.


Setup Instructions
------------------

In order to export the Docker Image to a singularity image, `apptainer`_ is required.
A detailed overview of the installation procedure for ``apptainer`` can be found in its
`documentation`_. For convenience, we summarize the steps here for a local installation:

.. code:: bash

    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:apptainer/ppa
    sudo apt update
    sudo apt install -y apptainer

For simplicity, we recommend that an SSH connection is set up between the local
development machine and the cluster. Such a connection will simplify the file transfer and prevent
the user cluster password from being requested multiple times.

.. attention::
  The workflow has been tested with ``apptainer version 1.2.5-1.el7``. There have been reported binding issues with
  previous versions (such as ``apptainer version 1.1.3-1.el7``). Please ensure that you are using the latest version.

Configuring the cluster parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, you need to configure the cluster-specific parameters in ``docker/.env`` file.
The following describes the parameters that need to be configured:

- ``CLUSTER_ISAAC_SIM_CACHE_DIR``:
  The directory on the cluster where the Isaac Sim cache is stored. This directory
  has to end on ``docker-isaac-sim``. This directory will be copied to the compute node
  and mounted into the singularity container. It should increase the speed of starting
  the simulation.
- ``CLUSTER_ORBIT_DIR``:
  The directory on the cluster where the orbit code is stored. This directory has to
  end on ``orbit``. This directory will be copied to the compute node and mounted into
  the singularity container. When a job is submitted, the latest local changes will
  be copied to the cluster.
- ``CLUSTER_LOGIN``:
  The login to the cluster. Typically, this is the user and cluster names,
  e.g., ``your_user@euler.ethz.ch``.
- ``CLUSTER_SIF_PATH``:
  The path on the cluster where the singularity image will be stored. The image will be
  copied to the compute node but not uploaded again to the cluster when a job is submitted.
- ``CLUSTER_PYTHON_EXECUTABLE``:
  The path within orbit to the Python executable that should be executed in the submitted job.

Exporting to singularity image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we need to export the Docker image to a singularity image and upload
it to the cluster. This step is only required once when the first job is submitted
or when the Docker image is updated. For instance, due to an upgrade of the Isaac Sim
version, or additional requirements for your project.

To export to a singularity image, execute the following command:

.. code:: bash

    ./docker/container.sh push

This command will create a singularity image under ``docker/exports`` directory and
upload it to the defined location on the cluster. Be aware that creating the singularity
image can take a while.


Job Submission and Execution
----------------------------

Defining the job parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The job parameters are defined inside the ``docker/cluster/submit_job.sh``.
A typical SLURM operation requires specifying the number of CPUs and GPUs, the memory, and
the time limit. For more information, please check the `SLURM documentation`_.

The default configuration is as follows:

.. literalinclude:: ../../../docker/cluster/submit_job.sh
  :language: bash
  :lines: 12-19
  :linenos:
  :lineno-start: 12

An essential requirement for the cluster is that the compute node has access to the internet at all times.
This is required to load assets from the Nucleus server. For some cluster architectures, extra modules
must be loaded to allow internet access.

For instance, on ETH Zurich Euler cluster, the ``eth_proxy`` module needs to be loaded. This can be done
by adding the following line to the ``submit_job.sh`` script:

.. literalinclude:: ../../../docker/cluster/submit_job.sh
  :language: bash
  :lines: 3-5
  :linenos:
  :lineno-start: 3

Submitting a job
~~~~~~~~~~~~~~~~

To submit a job on the cluster, the following command can be used:

.. code:: bash

    ./docker/container.sh job "argument1" "argument2" ...

This command will copy the latest changes in your code to the cluster and submit a job. Please ensure that
your Python executable's output is stored under ``orbit/logs`` as this directory will be copied again
from the compute node to ``CLUSTER_ORBIT_DIR``.

The training arguments anove are passed to the Python executable. As an example, the standard
ANYmal rough terrain locomotion training can be executed with the following command:

.. code:: bash

    ./docker/container.sh job --task Isaac-Velocity-Rough-Anymal-C-v0 --headless --video --offscreen_render

The above will, in addition, also render videos of the training progress and store them under ``orbit/logs`` directory.

.. note::

    The ``./docker/container.sh job`` command will copy the latest changes in your code to the cluster. However,
    it will not delete any files that have been deleted locally. These files will still exist on the cluster
    which can lead to issues. In this case, we recommend removing the ``CLUSTER_ORBIT_DIR`` directory on
    the cluster and re-run the command.


.. _Singularity: https://docs.sylabs.io/guides/2.6/user-guide/index.html
.. _ETH Zurich Euler: https://scicomp.ethz.ch/wiki/Euler
.. _apptainer: https://apptainer.org/
.. _documentation: www.apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages
.. _SLURM documentation: www.slurm.schedmd.com/sbatch.html
