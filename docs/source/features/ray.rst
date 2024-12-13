===========================
Ray Job Dispatch and Tuning
===========================

.. currentmodule:: omni.isaac.lab

Isaac Lab supports `Ray <https://docs.ray.io/en/latest/index.html>`_ for streamlining dispatching multiple training jobs (in parallel and in series),
and hyperparameter tuning, both on local and remote configurations.

This `independent community contributed walkthrough video <https://youtu.be/z7MDgSga2Ho?feature=shared>`_
demonstrates some of the core functionality of the Ray integration covered in this overview. Although there may be some
differences in the codebase (such as file names being shortened) since the creation of the video,
the general workflow is the same.

.. attention::

  This functionality is experimental, and has been tested only on Linux.

.. contents:: Table of Contents
  :depth: 3
  :local:

Overview
--------

The Ray integration is useful for the following:

- Dispatching several training jobs in parallel or sequentially with minimal interaction
- Tuning hyperparameters; in parallel or sequentially with support for multiple GPUs and/or multiple GPU Nodes
- Using the same training setup everywhere (on cloud and local) with minimal overhead
- Resource Isolation for training jobs

The core functionality of the Ray workflow consists of two main scripts that enable the orchestration
of resource-wrapped and tuning aggregate jobs. These scripts facilitate the decomposition of
aggregate jobs (overarching experiments) into individual jobs, which are discrete commands
executed on the cluster. An aggregate job can include multiple individual jobs.
For clarity, this guide refers to the jobs one layer below the topmost aggregate level as sub-jobs.

Both resource-wrapped and tuning aggregate jobs dispatch individual jobs to a designated Ray
cluster, which leverages the cluster's resources (e.g., a single workstation node or multiple nodes)
to execute these jobs with workers in parallel and/or sequentially. By default, aggregate jobs use all \
available resources on each available GPU-enabled node for each sub-job worker. This can be changed through
specifying the ``--num_workers`` argument, especially critical for parallel aggregate
job processing on local or virtual multi-GPU machines

In resource-wrapped aggregate jobs, each sub-job and its
resource requirements are defined manually, enabling resource isolation.
For tuning aggregate jobs, individual jobs are generated automatically based on a hyperparameter
sweep configuration. This assumes homogeneous node resource composition for nodes with GPUs.

.. dropdown:: source/standalone/workflows/ray/wrap_resources.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/wrap_resources.py
    :language: python
    :emphasize-lines: 14-66

.. dropdown:: source/standalone/workflows/ray/tuner.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/tuner.py
    :language: python
    :emphasize-lines: 18-53


The following script can be used to submit aggregate
jobs to one or more Ray cluster(s), which can be used for
running jobs on a remote cluster or simultaneous jobs with heterogeneous
resource requirements:

.. dropdown:: source/standalone/workflows/ray/submit_job.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/submit_job.py
    :language: python
    :emphasize-lines: 12-53

The following script can be used to extract KubeRay Cluster information for aggregate job submission.

.. dropdown:: source/standalone/workflows/ray/grok_cluster_with_kubectl.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/grok_cluster_with_kubectl.py
    :language: python
    :emphasize-lines: 14-26

The following script can be used to easily create clusters on Google GKE.

.. dropdown:: source/standalone/workflows/ray/launch.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/launch.py
    :language: python
    :emphasize-lines: 16-37

**Installation**
----------------

The Ray functionality requires additional dependencies be installed.

To use Ray without Kubernetes, like on a local computer or VM,
``kubectl`` is not required. For use on Kubernetes clusters with KubeRay,
such as Google Kubernetes Engine or Amazon Elastic Kubernetes Service, ``kubectl`` is required, and can
be installed via the `Kubernetes website <https://kubernetes.io/docs/tasks/tools/>`_

The pythonic dependencies can be installed with:

.. code-block:: bash

  # For multi-run support and resource isolation
  ./isaaclab.sh -p -m pip install ray[default]==2.31.0
  # For hyperparameter tuning
  ./isaaclab.sh -p -m pip install ray[tune]==2.31.0
  ./isaaclab.sh -p -m pip install optuna bayesian-optimization
  # MLFlow is needed only for fetching logs on clusters, not needed for local
  ./isaaclab.sh -p -m pip install mlflow

If using KubeRay clusters on Google GKE with the batteries-included cluster launch file,
the following dependencies are also needed.

.. code-block:: bash

  ./isaaclab.sh -p -m pip install kubernetes Jinja2

**Setup Overview: Cluster Configuration**
-----------------------------------------

Select one of the following methods to create a Ray Cluster to accept and execute dispatched jobs.

Single-Node Ray Cluster (Recommended for Beginners)
'''''''''''''''''''''''''''''''''''''''''''''''''''
For use on a single machine (node) such as a local computer or VM, the
following command can be used start a ray server. This is compatible with
multiple-GPU machines. This Ray server will run indefinitely until it is stopped with ``CTRL + C``

.. code-block:: bash

  echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | ./isaaclab.sh -p

KubeRay Clusters
''''''''''''''''
.. attention::
  The ``ray`` command should be modified to use Isaac python, which could be achieved in a fashion similar to
  ``sed -i "1i $(echo "#!/workspace/isaaclab/_isaac_sim/python.sh")" \
  /isaac-sim/kit/python/bin/ray && ln -s /isaac-sim/kit/python/bin/ray /usr/local/bin/ray``.

Google Cloud is currently the only platform tested, although
any cloud provider should work if one configures the following:

- An container registry (NGC, GCS artifact registry, AWS ECR, etc) with
  an Isaac Lab image configured to support Ray. See ``cluster_configs/Dockerfile`` to see how to modify the ``isaac-lab-base``
  container for Ray compatibility. Ray should use the isaac sim python shebang, and ``nvidia-smi``
  should work within the container. Be careful with the setup here as
  paths need to be configured correctly for everything to work. It's likely that
  the example dockerfile will work out of the box and can be pushed to the registry, as
  long as the base image has already been built as in the container guide
- A Kubernetes setup with available NVIDIA RTX (likely ``l4`` or ``l40`` or ``tesla-t4`` or ``a10``) GPU-passthrough node-pool resources,
  that has access to your container registry/storage bucket and has the Ray operator enabled with correct IAM
  permissions. This can be easily achieved with services such as Google GKE or AWS EKS,
  provided that your account or organization has been granted a GPU-budget. It is recommended
  to use manual kubernetes services as opposed to "autopilot" services for cost-effective
  experimentation as this way clusters can be completely shut down when not in use, although
  this may require installing the `Nvidia GPU Operator <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/google-gke.html>`_
- An MLFlow server that your cluster has access to.
- A ``kuberay.yaml.ninja`` file that describes how to allocate resources (already included for
  Google Cloud, which can be referenced for the format and MLFlow integration)

Ray Clusters (Without Kubernetes)
'''''''''''''''''''''''''''''''''
.. attention::
  Modify the Ray command to use Isaac Python like in KubeRay Clusters, and follow the same
  steps for creating an image/cluster permissions/bucket access.

See the `Ray Clusters Overview <https://docs.ray.io/en/latest/cluster/getting-started.html>`_ or
`Anyscale <https://www.anyscale.com/product>`_ for more information


**Dispatching Jobs and Tuning**
-------------------------------

Select one of the following guides that matches your desired Cluster configuration.

Simple Ray Cluster (Local/VM)
'''''''''''''''''''''''''''''

This guide assumes that there is a Ray cluster already running, and that this script is run locally on the cluster, or
that the cluster job submission address is known.

1.) Testing that the cluster works can be done as follows.

.. code-block:: bash

  ./isaaclab.sh -p source/standalone/workflows/ray/wrap_resources.py --test

2.) Submitting resource-wrapped sub-jobs can be done as described in the following file:

.. dropdown:: source/standalone/workflows/ray/wrap_resources.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/wrap_resources.py
    :language: python
    :emphasize-lines: 14-66

3.) For tuning jobs, specify the hyperparameter sweep similar to the following two files.

.. dropdown:: source/standalone/workflows/ray/hyperparameter_tuning/vision_cfg.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/hyperparameter_tuning/vision_cfg.py
    :language: python

.. dropdown:: source/standalone/workflows/ray/hyperparameter_tuning/vision_cartpole_cfg.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/hyperparameter_tuning/vision_cartpole_cfg.py
    :language: python

Then, see the local examples in the following file to see how to start a tuning run.

.. dropdown:: source/standalone/workflows/ray/tuner.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/tuner.py
    :language: python
    :emphasize-lines: 18-53



To view the logs, simply run ``tensorboard --logdir=<LOCAL_STORAGE_PATH_READ_FROM_OUTPUT>``

Remote Ray Cluster Setup and Use
'''''''''''''''''''''''''''''''''
This guide assumes that one desires to create a cluster on a remote host or server. This
guide includes shared steps, and KubeRay or Ray specific steps. Follow all shared steps (part I and II), and then
only the KubeRay or Ray steps depending on your desired configuration, in order of shared steps part I, then
the configuration specific steps, then shared steps part II.

Shared Steps Between KubeRay and Pure Ray Part I
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.) Build the Isaac Ray image, and upload it to your container registry of choice.

.. code-block:: bash

  # Login with NGC (nvcr.io) registry first, see docker steps in repo.
  ./isaaclab.sh -p docker/container.py start
  # Build the special Isaac Lab Ray Image
  docker build -t <REGISTRY/IMAGE_NAME> -f source/standalone/workflows/ray/cluster_configs/Dockerfile .
  # Push the image to your registry of choice.
  docker push <REGISTRY/IMAGE_NAME>

KubeRay Specific
~~~~~~~~~~~~~~~~
`k9s <https://github.com/derailed/k9s>`_ is a great tool for monitoring your clusters that can
easily be installed with ``snap install k9s --devmode``.

1.) Verify cluster access, and that the correct operators are installed.

.. code-block:: bash

  # Verify cluster access
  kubectl cluster-info
  # If using a manually managed cluster (not Autopilot or the like)
  # verify that there are node pools
  kubectl get nodes
  # Check that the ray operator is installed on the cluster
  # should list rayclusters.ray.io , rayjobs.ray.io , and rayservices.ray.io
  kubectl get crds | grep ray
  # Check that the NVIDIA Driver Operator is installed on the cluster
  # should list clusterpolicies.nvidia.com
  kubectl get crds | grep nvidia

2.) Create the KubeRay cluster and an MLFlow server for receiving logs
that your cluster has access to.
This can be done automatically for Google GKE,
where instructions are included in the following creation file. More than once cluster
can be created at once. Each cluster can have heterogeneous resources if so desired,
although only
For other cloud services, the ``kuberay.yaml.ninja`` will be similar to that of
Google's.

.. dropdown:: source/standalone/workflows/ray/launch.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/launch.py
    :language: python
    :emphasize-lines: 15-37

3.) Fetch the KubeRay cluster IP addresses, and the MLFLow Server IP.
This can be done automatically for KubeRay clusters,
where instructions are included in the following fetching file.
The KubeRay clusters are saved to a file, but the MLFLow Server IP is
printed.

.. dropdown:: source/standalone/workflows/ray/grok_cluster_with_kubectl.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/grok_cluster_with_kubectl.py
    :language: python
    :emphasize-lines: 14-26

Ray Specific
~~~~~~~~~~~~


1.) Verify cluster access.

2.) Create a ``~/.cluster_config`` file, where ``name: <NAME> address: http://<IP>:<PORT>`` is on
a new line for each unique cluster. For one cluster, there should only be one line in this file.

3.) Start an MLFLow Server to receive the logs that the ray cluster has access to,
and determine the server URI.

Shared Steps Between KubeRay and Pure Ray Part II
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1.) Test that your cluster is operational with the following.

.. code-block:: bash

  # Test that NVIDIA GPUs are visible and that Ray is operation with the following command:
  ./isaaclab.sh -p source/standalone/workflows/ray/wrap_resources.py
	--jobs wrap_resources.py --test

2.) Submitting Jobs can be done in the following manner, with the following script.

.. dropdown:: source/standalone/workflows/ray/submit_job.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/submit_job.py
    :language: python
    :emphasize-lines: 12-53

3.) For tuning jobs, specify the hyperparameter sweep similar to :class:`RLGamesCameraJobCfg` in the following file:

.. dropdown:: source/standalone/workflows/ray/tuner.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/tuner.py
    :language: python
    :emphasize-lines: 18-53

For example, see the Cartpole Example configurations.

.. dropdown:: source/standalone/workflows/ray/hyperparameter_tuning/vision_cartpole_cfg.py
  :icon: code

  .. literalinclude:: ../../../source/standalone/workflows/ray/hyperparameter_tuning/vision_cartpole_cfg.py
    :language: python


Tuning jobs can also be submitted via ``submit_job.py``

To view the tuning results, view the MLFlow dashboard of the server that you created.
For KubeRay, this can be done through port forwarding the MLFlow dashboard, with

``kubectl port-forward service/isaacray-mlflow 5000:5000``

and visiting the following address in a browser.

``localhost:5000``

If the MLFlow port is forwarded like above, it can be converted into tensorboard logs with
this following command.

``./isaaclab.sh -p source/standalone/workflows/ray/mlflow_to_local_tensorboard.py \
--uri http://localhost:5000 --experiment-name IsaacRay-<CLASS_JOB_CFG>-tune --download-dir test``


**Cluster Cleanup**
'''''''''''''''''''

For the sake of conserving resources, and potentially freeing precious GPU resources for other people to use
on shared compute platforms, please destroy the Ray cluster after use. They can be easily
recreated! For KubeRay clusters, this can be done as follows.

.. code-block:: bash

  kubectl get raycluster | egrep 'isaacray' | awk '{print $1}' | xargs kubectl delete raycluster &&
  kubectl get deployments | egrep 'mlflow' | awk '{print $1}' | xargs kubectl delete deployment &&
  kubectl get services | egrep 'mlflow' | awk '{print $1}' | xargs kubectl delete service
