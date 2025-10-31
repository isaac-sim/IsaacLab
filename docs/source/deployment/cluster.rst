.. _deployment-cluster:


Cluster Guide
=============

Clusters are a great way to speed up training and evaluation of learning algorithms.
While the Isaac Lab Docker image can be used to run jobs on a cluster, many clusters only
support singularity images. This is because `singularity`_ is designed for
ease-of-use on shared multi-user systems and high performance computing (HPC) environments.
It does not require root privileges to run containers and can be used to run user-defined
containers.

Singularity is compatible with all Docker images. In this section, we describe how to
convert the Isaac Lab Docker image into a singularity image and use it to submit jobs to a cluster.

.. attention::

    Cluster setup varies across different institutions. The following instructions have been
    tested on the `ETH Zurich Euler`_ cluster (which uses the SLURM workload manager), and the
    IIT Genoa Franklin cluster (which uses PBS workload manager).

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
  The workflow has been tested with:

  - ``apptainer version 1.2.5-1.el7`` and ``docker version 24.0.7``
  - ``apptainer version 1.3.4`` and ``docker version 27.3.1``

  In the case of issues, please try to switch to those versions.


Configuring the cluster parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, you need to configure the cluster-specific parameters in ``docker/cluster/.env.cluster`` file.
The following describes the parameters that need to be configured:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - CLUSTER_JOB_SCHEDULER
     - The job scheduler/workload manager used by your cluster. Currently, we support 'SLURM' and
       'PBS' workload managers.
   * - CLUSTER_ISAAC_SIM_CACHE_DIR
     - The directory on the cluster where the Isaac Sim cache is stored. This directory
       has to end on ``docker-isaac-sim``. It will be copied to the compute node
       and mounted into the singularity container. This should increase the speed of starting
       the simulation.
   * - CLUSTER_ISAACLAB_DIR
     - The directory on the cluster where the Isaac Lab logs are stored. This directory has to
       end on ``isaaclab``. It will be copied to the compute node and mounted into
       the singularity container. When a job is submitted, the latest local changes will
       be copied to the cluster to a new directory in the format ``${CLUSTER_ISAACLAB_DIR}_${datetime}``
       with the date and time of the job submission. This allows to run multiple jobs with different code versions at
       the same time.
   * - CLUSTER_LOGIN
     - The login to the cluster. Typically, this is the user and cluster names,
       e.g., ``your_user@euler.ethz.ch``.
   * - CLUSTER_SIF_PATH
     - The path on the cluster where the singularity image will be stored. The image will be
       copied to the compute node but not uploaded again to the cluster when a job is submitted.
   * - REMOVE_CODE_COPY_AFTER_JOB
     - Whether the copied code should be removed after the job is finished or not. The logs from the job will not be deleted
       as these are saved under the permanent ``CLUSTER_ISAACLAB_DIR``. This feature is useful
       to save disk space on the cluster. If set to ``true``, the code copy will be removed.
   * - CLUSTER_PYTHON_EXECUTABLE
     - The path within Isaac Lab to the Python executable that should be executed in the submitted job.

When a ``job`` is submitted, it will also use variables defined in ``docker/.env.base``, though these
should be correct by default.

Exporting to singularity image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we need to export the Docker image to a singularity image and upload
it to the cluster. This step is only required once when the first job is submitted
or when the Docker image is updated. For instance, due to an upgrade of the Isaac Sim
version, or additional requirements for your project.

To export to a singularity image, execute the following command:

.. code:: bash

    ./docker/cluster/cluster_interface.sh push [profile]

This command will create a singularity image under ``docker/exports`` directory and
upload it to the defined location on the cluster. It requires that you have previously
built the image with the ``container.py`` interface. Be aware that creating the singularity
image can take a while.
``[profile]`` is an optional argument that specifies the container profile to be used. If no profile is
specified, the default profile ``base`` will be used.

.. note::
  By default, the singularity image is created without root access by providing the ``--fakeroot`` flag to
  the ``apptainer build`` command. In case the image creation fails, you can try to create it with root
  access by removing the flag in ``docker/cluster/cluster_interface.sh``.


Defining the job parameters
---------------------------

The job parameters need to be defined based on the job scheduler used by your cluster.
You only need to update the appropriate script for the scheduler available to you.

- For SLURM, update the parameters in ``docker/cluster/submit_job_slurm.sh``.
- For PBS, update the parameters in ``docker/cluster/submit_job_pbs.sh``.

For SLURM
~~~~~~~~~

The job parameters are defined inside the ``docker/cluster/submit_job_slurm.sh``.
A typical SLURM operation requires specifying the number of CPUs and GPUs, the memory, and
the time limit. For more information, please check the `SLURM documentation`_.

The default configuration is as follows:

.. literalinclude:: ../../../docker/cluster/submit_job_slurm.sh
  :language: bash
  :lines: 12-19
  :linenos:
  :lineno-start: 12

An essential requirement for the cluster is that the compute node has access to the internet at all times.
This is required to load assets from the Nucleus server. For some cluster architectures, extra modules
must be loaded to allow internet access.

For instance, on ETH Zurich Euler cluster, the ``eth_proxy`` module needs to be loaded. This can be done
by adding the following line to the ``submit_job_slurm.sh`` script:

.. literalinclude:: ../../../docker/cluster/submit_job_slurm.sh
  :language: bash
  :lines: 3-5
  :linenos:
  :lineno-start: 3

For PBS
~~~~~~~

The job parameters are defined inside the ``docker/cluster/submit_job_pbs.sh``.
A typical PBS operation requires specifying the number of CPUs and GPUs, and the time limit. For more
information, please check the `PBS Official Site`_.

The default configuration is as follows:

.. literalinclude:: ../../../docker/cluster/submit_job_pbs.sh
  :language: bash
  :lines: 11-17
  :linenos:
  :lineno-start: 11


Submitting a job
----------------

To submit a job on the cluster, the following command can be used:

.. code:: bash

    ./docker/cluster/cluster_interface.sh job [profile] "argument1" "argument2" ...

This command will copy the latest changes in your code to the cluster and submit a job. Please ensure that
your Python executable's output is stored under ``isaaclab/logs`` as this directory is synced between the compute
node and ``CLUSTER_ISAACLAB_DIR``.

``[profile]`` is an optional argument that specifies which singularity image corresponding to the  container profile
will be used. If no profile is specified, the default profile ``base`` will be used. The profile has be defined
directlty after the ``job`` command. All other arguments are passed to the Python executable. If no profile is
defined, all arguments are passed to the Python executable.

The training arguments are passed to the Python executable. As an example, the standard
ANYmal rough terrain locomotion training can be executed with the following command:

.. code:: bash

    ./docker/cluster/cluster_interface.sh job --task Isaac-Velocity-Rough-Anymal-C-v0 --headless --video --enable_cameras

The above will, in addition, also render videos of the training progress and store them under ``isaaclab/logs`` directory.

.. _Singularity: https://docs.sylabs.io/guides/2.6/user-guide/index.html
.. _ETH Zurich Euler: https://www.gdc-docs.ethz.ch/EulerManual/site/overview/
.. _PBS Official Site: https://openpbs.org/
.. _apptainer: https://apptainer.org/
.. _documentation: https://www.apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages
.. _SLURM documentation: https://www.slurm.schedmd.com/sbatch.html
