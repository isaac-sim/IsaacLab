===========================
Ray Job Dispatch and Tuning
===========================

.. currentmodule:: isaaclab

Isaac Lab supports `Ray <https://docs.ray.io/en/latest/index.html>`_ for streamlining dispatching multiple training jobs (in parallel and in series),
and hyperparameter tuning, both on local and remote configurations.

This `independent community contributed walkthrough video <https://youtu.be/z7MDgSga2Ho?feature=shared>`_
demonstrates some of the core functionality of the Ray integration covered in this overview. Although there may be some
differences in the codebase (such as file names being shortened) since the creation of the video,
the general workflow is the same.

.. attention::

  This functionality is experimental, and has been tested only on Linux.



Overview
--------

The Ray integration is useful for the following.

- Dispatching several training jobs in parallel or sequentially with minimal interaction.
- Tuning hyperparameters; in parallel or sequentially with support for multiple GPUs and/or multiple GPU Nodes.
- Using the same training setup everywhere (on cloud and local) with minimal overhead.
- Resource Isolation for training jobs (resource-wrapped jobs).

The core functionality of the Ray workflow consists of two main scripts that enable the orchestration
of resource-wrapped and tuning aggregate jobs. In resource-wrapped aggregate jobs, each sub-job and its
resource requirements are defined manually, enabling resource isolation.
For tuning aggregate jobs, individual jobs are generated automatically based on a hyperparameter
sweep configuration.

Both resource-wrapped and tuning aggregate jobs dispatch individual jobs to a designated Ray
cluster, which leverages the cluster's resources (e.g., a single workstation node or multiple nodes)
to execute these jobs with workers in parallel and/or sequentially.

By default, jobs use all \
available resources on each available GPU-enabled node for each sub-job worker. This can be changed through
specifying the ``--num_workers`` argument for resource-wrapped jobs, or ``--num_workers_per_node``
for tuning jobs, which is especially critical for parallel aggregate
job processing on local/virtual multi-GPU machines. Tuning jobs assume homogeneous node resource composition for nodes with GPUs.

The two following files contain the core functionality of the Ray integration.

.. dropdown:: scripts/reinforcement_learning/ray/wrap_resources.py
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/wrap_resources.py
    :language: python
    :emphasize-lines: 14-66

.. dropdown:: scripts/reinforcement_learning/ray/tuner.py
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/tuner.py
    :language: python
    :emphasize-lines: 18-53


The following script can be used to submit aggregate
jobs to one or more Ray cluster(s), which can be used for
running jobs on a remote cluster or simultaneous jobs with heterogeneous
resource requirements.

.. dropdown:: scripts/reinforcement_learning/ray/submit_job.py
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/submit_job.py
    :language: python
    :emphasize-lines: 12-53

The following script can be used to extract KubeRay cluster information for aggregate job submission.

.. dropdown:: scripts/reinforcement_learning/ray/grok_cluster_with_kubectl.py
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/grok_cluster_with_kubectl.py
    :language: python
    :emphasize-lines: 14-26

The following script can be used to easily create clusters on Google GKE.

.. dropdown:: scripts/reinforcement_learning/ray/launch.py
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/launch.py
    :language: python
    :emphasize-lines: 16-37

Docker-based Local Quickstart
-----------------------------

First, follow the `Docker Guide <https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html>`_
to set up the NVIDIA Container Toolkit and Docker Compose.

Then, run the following steps to start a tuning run.

.. code-block:: bash

  # Build the base image, but we don't need to run it
  python3 docker/container.py start && python3 docker/container.py stop
  # Build the tuning image with extra deps
  docker build -t isaacray -f scripts/reinforcement_learning/ray/cluster_configs/Dockerfile .
  # Start the tuning image - symlink so that changes in the source folder show up in the container
  docker run -v $(pwd)/source:/workspace/isaaclab/source -it --gpus all --net=host --entrypoint /bin/bash isaacray
  # Start the Ray server within the tuning image
  echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | ./isaaclab.sh -p



In a different terminal, run the following.


.. code-block:: bash

  # In a new terminal (don't close the above) , enter the image with a new shell.
  docker container ps
  docker exec -it <ISAAC_RAY_IMAGE_ID_FROM_CONTAINER_PS> /bin/bash
  # Start a tuning run, with one parallel worker per GPU
  ./isaaclab.sh -p scripts/reinforcement_learning/ray/tuner.py \
    --cfg_file scripts/reinforcement_learning/ray/hyperparameter_tuning/vision_cartpole_cfg.py \
    --cfg_class CartpoleTheiaJobCfg \
    --run_mode local \
    --workflow scripts/reinforcement_learning/rl_games/train.py \
    --num_workers_per_node <NUMBER_OF_GPUS_IN_COMPUTER>


To view the training logs, in a different terminal, run the following and visit ``localhost:6006`` in a browser afterwards.

.. code-block:: bash

  # In a new terminal (don't close the above) , enter the image with a new shell.
  docker container ps
  docker exec -it <ISAAC_RAY_IMAGE_ID_FROM_CONTAINER_PS> /bin/bash
  # Start a tuning run, with one parallel worker per GPU
  tensorboard --logdir=.


Submitting resource-wrapped individual jobs instead of automatic tuning runs is described in the following file.

.. dropdown:: scripts/reinforcement_learning/ray/wrap_resources.py
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/wrap_resources.py
    :language: python
    :emphasize-lines: 14-66

Transferring files from the running container can be done as follows.

.. code-block:: bash

  docker container ps
  docker cp <ISAAC_RAY_IMAGE_ID_FROM_CONTAINER_PS>:</path/in/container/file>  </path/on/host/>


For tuning jobs, specify the tuning job / hyperparameter sweep as child class of :class:`JobCfg` .
The included :class:`JobCfg` only supports the ``rl_games`` workflow due to differences in
environment entrypoints and hydra arguments, although other workflows will work if provided a compatible
:class:`JobCfg`.

.. dropdown:: scripts/reinforcement_learning/ray/tuner.py (JobCfg definition)
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/tuner.py
    :language: python
    :start-at: class JobCfg
    :end-at: self.cfg = cfg

For example, see the following Cartpole Example configurations.

.. dropdown:: scripts/reinforcement_learning/ray/hyperparameter_tuning/vision_cartpole_cfg.py
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/hyperparameter_tuning/vision_cartpole_cfg.py
    :language: python


Remote Clusters
---------------

Select one of the following methods to create a Ray cluster to accept and execute dispatched jobs.

KubeRay Setup
~~~~~~~~~~~~~

If using KubeRay clusters on Google GKE with the batteries-included cluster launch file,
the following dependencies are also needed.

.. code-block:: bash

  python3 -p -m pip install kubernetes Jinja2

For use on Kubernetes clusters with KubeRay,
such as Google Kubernetes Engine or Amazon Elastic Kubernetes Service, ``kubectl`` is required, and can
be installed via the `Kubernetes website <https://kubernetes.io/docs/tasks/tools/>`_ .

Google Cloud is currently the only platform tested, although
any cloud provider should work if one configures the following.

.. attention::
  The ``ray`` command should be modified to use Isaac python, which could be achieved in a fashion similar to
  ``sed -i "1i $(echo "#!/workspace/isaaclab/_isaac_sim/python.sh")" \
  /isaac-sim/kit/python/bin/ray && ln -s /isaac-sim/kit/python/bin/ray /usr/local/bin/ray``.

- An container registry (NGC, GCS artifact registry, AWS ECR, etc) with
  an Isaac Lab image configured to support Ray. See ``cluster_configs/Dockerfile`` to see how to modify the ``isaac-lab-base``
  container for Ray compatibility. Ray should use the isaac sim python shebang, and ``nvidia-smi``
  should work within the container. Be careful with the setup here as
  paths need to be configured correctly for everything to work. It's likely that
  the example dockerfile will work out of the box and can be pushed to the registry, as
  long as the base image has already been built as in the container guide.
- A Kubernetes setup with available NVIDIA RTX (likely ``l4`` or ``l40`` or ``tesla-t4`` or ``a10``) GPU-passthrough node-pool resources,
  that has access to your container registry/storage bucket and has the Ray operator enabled with correct IAM
  permissions. This can be easily achieved with services such as Google GKE or AWS EKS,
  provided that your account or organization has been granted a GPU-budget. It is recommended
  to use manual kubernetes services as opposed to "autopilot" services for cost-effective
  experimentation as this way clusters can be completely shut down when not in use, although
  this may require installing the `Nvidia GPU Operator <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/google-gke.html>`_ .
- An `MLFlow server <https://mlflow.org/docs/latest/getting-started/logging-first-model/step1-tracking-server.html>`_ that your cluster has access to
  (already included for Google Cloud, which can be referenced for the format and MLFlow integration).
- A ``kuberay.yaml.ninja`` file that describes how to allocate resources (already included for
  Google Cloud, which can be referenced for the format and MLFlow integration).

Ray Clusters (Without Kubernetes) Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. attention::
  Modify the Ray command to use Isaac Python like in KubeRay clusters, and follow the same
  steps for creating an image/cluster permissions.

See the `Ray Clusters Overview <https://docs.ray.io/en/latest/cluster/getting-started.html>`_ or
`Anyscale <https://www.anyscale.com/product>`_ for more information.

Also, create an `MLFlow server <https://mlflow.org/docs/latest/getting-started/logging-first-model/step1-tracking-server.html>`_ that your local
host and cluster have access to.

Shared Steps Between KubeRay and Pure Ray Part I
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.) Install Ray on your local machine.

.. code-block:: bash

  python3 -p -m pip install ray[default]==2.31.0

2.) Build the Isaac Ray image, and upload it to your container registry of choice.

.. code-block:: bash

  # Login with NGC (nvcr.io) registry first, see docker steps in repo.
  python3 docker/container.py start
  # Build the special Isaac Lab Ray Image
  docker build -t <REGISTRY/IMAGE_NAME> -f scripts/reinforcement_learning/ray/cluster_configs/Dockerfile .
  # Push the image to your registry of choice.
  docker push <REGISTRY/IMAGE_NAME>

KubeRay Clusters Only
~~~~~~~~~~~~~~~~~~~~~
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
where instructions are included in the following creation file.

.. dropdown:: scripts/reinforcement_learning/ray/launch.py
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/launch.py
    :language: python
    :emphasize-lines: 15-37

For other cloud services, the ``kuberay.yaml.ninja`` will be similar to that of
Google's.


.. dropdown:: scripts/reinforcement_learning/ray/cluster_configs/google_cloud/kuberay.yaml.ninja
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/cluster_configs/google_cloud/kuberay.yaml.jinja
      :language: python



3.) Fetch the KubeRay cluster IP addresses, and the MLFLow Server IP.
This can be done automatically for KubeRay clusters,
where instructions are included in the following fetching file.
The KubeRay clusters are saved to a file, but the MLFLow Server IP is
printed.

.. dropdown:: scripts/reinforcement_learning/ray/grok_cluster_with_kubectl.py
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/grok_cluster_with_kubectl.py
    :language: python
    :emphasize-lines: 14-26

Ray Clusters Only (Without Kubernetes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1.) Verify cluster access.

2.) Create a ``~/.cluster_config`` file, where ``name: <NAME> address: http://<IP>:<PORT>`` is on
a new line for each unique cluster. For one cluster, there should only be one line in this file.

3.) Start an MLFLow Server to receive the logs that the ray cluster has access to,
and determine the server URI.

Dispatching Steps Shared Between KubeRay and Pure Ray Part II
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


1.) Test that your cluster is operational with the following.

.. code-block:: bash

  # Test that NVIDIA GPUs are visible and that Ray is operation with the following command:
  python3 scripts/reinforcement_learning/ray/submit_job.py --aggregate_jobs wrap_resources.py --test

2.) Submitting tuning and/or resource-wrapped jobs is described in the :file:`submit_job.py` file.

.. dropdown:: scripts/reinforcement_learning/ray/submit_job.py
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/submit_job.py
    :language: python
    :emphasize-lines: 12-53

3.) For tuning jobs, specify the tuning job / hyperparameter sweep as a :class:`JobCfg` .
The included :class:`JobCfg` only supports the ``rl_games`` workflow due to differences in
environment entrypoints and hydra arguments, although other workflows will work if provided a compatible
:class:`JobCfg`.

.. dropdown:: scripts/reinforcement_learning/ray/tuner.py (JobCfg definition)
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/tuner.py
    :language: python
    :start-at: class JobCfg
    :end-at: self.cfg = cfg

For example, see the following Cartpole Example configurations.

.. dropdown:: scripts/reinforcement_learning/ray/hyperparameter_tuning/vision_cartpole_cfg.py
  :icon: code

  .. literalinclude:: ../../../scripts/reinforcement_learning/ray/hyperparameter_tuning/vision_cartpole_cfg.py
    :language: python


To view the tuning results, view the MLFlow dashboard of the server that you created.
For KubeRay, this can be done through port forwarding the MLFlow dashboard with the following.

``kubectl port-forward service/isaacray-mlflow 5000:5000``

Then visit the following address in a browser.

``localhost:5000``

If the MLFlow port is forwarded like above, it can be converted into tensorboard logs with
this following command.

``./isaaclab.sh -p scripts/reinforcement_learning/ray/mlflow_to_local_tensorboard.py \
--uri http://localhost:5000 --experiment-name IsaacRay-<CLASS_JOB_CFG>-tune --download-dir test``


Kubernetes Cluster Cleanup
''''''''''''''''''''''''''

For the sake of conserving resources, and potentially freeing precious GPU resources for other people to use
on shared compute platforms, please destroy the Ray cluster after use. They can be easily
recreated! For KubeRay clusters, this can be done as follows.

.. code-block:: bash

  kubectl get raycluster | egrep 'isaacray' | awk '{print $1}' | xargs kubectl delete raycluster &&
  kubectl get deployments | egrep 'mlflow' | awk '{print $1}' | xargs kubectl delete deployment &&
  kubectl get services | egrep 'mlflow' | awk '{print $1}' | xargs kubectl delete service &&
  kubectl get services | egrep 'isaacray' | awk '{print $1}' | xargs kubectl delete service
