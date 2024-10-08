#TODO: Move me into docs ;p

# Ray Integration

Through using Ray, we streamline distributed training runs.

The Ray integration is useful to you if:
- You want to tune models' hyperparameters as fast as possible in parallel on multiple GPUs
	and/or multiple GPU Nodes
- You want to simultaneously tune model hyperparameters for different environments/agents (see
	advanced usage)
- You want to use the same experimental training setup on cloud and local with minimal overhead


# Installation

This guide includes additional dependencies that are not part of the default Isaac Lab install
as this functionality is still largely experimental.

***You likely need to install `kubectl`*** , which can be done from [this link here](https://kubernetes.io/docs/tasks/tools/).

Note that if you are using Ray Clusters without kubernetes, like on a local setup,
this dependency is not needed.

To install all Python dependencies, run

```
./isaaclab.sh -p -m pip install "ray[default, tune]"==2.31.0
```

### Cloud Setup

On your cloud provider of choice, configure the following

- An container registry (NGC, GCS artifact registry, AWS ECR, etc) where you have
	an Isaac Lab image that you can pull with the correct permissions, configured to
	support Ray and nvidia-smi
	- See ```cluster_configs/Dockerfile``` to see how to modify the ```isaac-lab-base```
		container for Ray compatibility. Ray should use the isaac sim python shebang, and nvidia-smi
		should work within the container. Be careful with the setup here as
		paths need to be configured correctly for everything to work. It's likely that
		the example dockerfile will work for you out of the box.
- A Kubernetes Cluster with a GPU-passthrough enabled node-pool with available
	GPU pods that has access to your container registry/storage (likely has to be on same region/VPC),
	and has the Ray operator enabled with correct IAM permissions.
- A ``kuberay.yaml.ninja`` file that describes how to allocate resources (already included for
	google cloud)
- It is highly recommended to create a storage bucket to dump experiment logs/checkpoints to.

An example of what a cloud deploy might look look like is in ``cloud_cluster_configs/google_cloud``


### Local and Other Setups
#### Option A: With Ray Clusters (Recommended)
On the head machine, run ``ray start --head --port 6379``. On any worker machines,
make sure you can connect to the head machine, and then run
```ray start --address='HEAD_NODE_IP:6379'``` . For more info see

https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#on-prem

#### Option B: With Kubernetes / KubeRay
Spin up a kubernetes cluster with GPU passthrough.
For an example, see https://docs.wandb.ai/tutorials/minikube_gpu/

Install ray operator on the local cluster, and make a container where ray is installed.
See ``cluster_configs/Dockerfile`` for an example. Create a ``kuberay.yaml.jinja``
file for your local cluster similar to that of ``cluster_configs/google_cloud/kuberay.yaml.jinja``.
Now, the rest of the steps are the same as for cloud, just make sure to change ``--cluster_host``
to ``local`` when running ``launch.py``

#### Option C: With SLURM

See https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-network-ray
for more information. This guide does not explicitly support SLURM, but it should still be compatible.

# Shared Steps Between Cloud and Local
As a result of using Ray, running experiments in the cloud and locally have very similar steps.

### Kubernetes / KubeRay Specific (You can skip these steps if you've already set up the Ray Cluster)
1. Start your kubernetes server and verify you have access

	``kubectl get nodes`` should list your nodes

2. Ensure Ray Operator is installed on your cluster

	``kubectl get crds | grep ray`` should list rayclusters.ray.io , rayjobs.ray.io , and
	rayservices.ray.io

2. Spin up your KubeRay integration from the template cluster configuration file

	See ``./isaaclab.sh -p source/standalone/workflows/ray/launch.py -h``
	for all possible arguments

	For example, you could invoke with:

	```
	./isaaclab.sh -p source/standalone/workflows/ray/launch.py
	 --cluster_host google_cloud --namespace <NAMESPACE>  --image <CUSTOM_ISAAC_RAY_IMAGE> --min_workers 4 --max_workers 16
	 ```

3. Check that your KubeRay cluster worked with `kubectl get pods` and `kubectl describe pods`.
	It may take a few minutes for the cluster to spin up. If there is an error, or a crash loop backup,
	you can inspect the logs further with ``kubectl logs <POD_NAME>``. When all pods
	say ``Running`` as their status, the cluster is ready to tune hyperparameters.

### Shared Steps for Kubernetes/KubeRay and Ray Clusters

4. Check that you can issue jobs to the cluster, that all GPUs are available,
	that Ray is installed correctly, nvidia-smi works, and that the needed deps
	for Ray/Isaac Lab are found on path.

	If you are using Ray with Kubernetes, you can use the following command
	(after the containers are running, check with ``kubectl get pods``)

	```
	./source/standalone/workflows/ray/grok_cluster_address_with_kubectl.sh  && \
	ray job submit --working-dir source/standalone/workflows/ray \
	--address=$(awk '!/^#/{print $0}' ~/.ray_address | head -n 1) -- /workspace/isaaclab/_isaac_sim/python.sh \
	test_kuberay_config.py --num_jobs 4
	```

	Otherwise, determine the Ray head node address , and run

	```
	ray job submit --working-dir source/standalone/workflows/ray \
	--address=<RAY_HEAD_NODE_ADDRESS> -- /workspace/isaaclab/_isaac_sim/python.sh \
	test_kuberay_config.py --num_jobs 4
	```

5. Define your desired Ray job as a script on your local machine.
	For a hyperparameter tuning example, see ``source/standalone/workflows/ray/hyper_parameter_tuning/config/cartpole_sweep.py`` #TODO: ACTUALLY WRITE THIS BAD BOY

6. Start your distributed Ray job.

	If you are using Ray with Kubernetes, you can use the following command:

	```
	./source/standalone/workflows/ray/grok_ray_address_with_kubectl.sh &&
	ray job submit --working-dir source/standalone/workflows/ray \
	--address=$(awk '!/^#/{print $0}' ~/.ray_address | head -n 1) -- /workspace/isaaclab/_isaac_sim/python.sh \
	<YOUR_JOB_HERE>
	```

	Otherwise, determine the Ray head node address, and run

	```
	./source/standalone/workflows/ray/grok_ray_address.sh &&
	ray job submit --working-dir source/standalone/workflows/ray \
	--address=<RAY_HEAD_NODE_ADDRESS> -- /workspace/isaaclab/_isaac_sim/python.sh \
	<YOUR_JOB_HERE>
	```
7. When you have completed your distributed job, stop the cluster to conserve resources.

	If you are using Kubernetes/KubeRay, this can be done with

	``kubectl delete raycluster <CLUSTER_NAME> -n <NAMESPACE>``

	If you are using Ray clusters, this can be done from the head node with

	``ray stop``


# Advanced Usage

### Multiple Simultaneous Distributed Runs

If you'd like to run several distributed training runs at once, you can
use similar commands to a single distributed training run. This requires
the creation or existence of several Ray clusters
(with unique IP addresses), although this can be done atop a single
kubernetes cluster.

This can be used to compare the performance of separate approaches, tuned for fairness
of comparison.
For example, if you have an MLP agent, and a CNN agent for a task you could tune both of their
hyperparameter simultaneously in parallel.

####  Kubernetes / KubeRay Specific : For Cloud or Kubernetes Local
1. You can create several homogeneous ray clusters at once by specifying the ``--num_clusters`` flag to ``launch.py``.
	For example,

	```
	./isaaclab.sh -p source/standalone/workflows/ray/launch.py
	 --cluster_host google_cloud --namespace <NAMESPACE>  --image <CUSTOM_ISAAC_RAY_IMAGE> --min_workers 4 --max_workers 16 --num_clusters 3
	 ```
	 If you want the clusters to have heterogeneous resource allocations, like different numbers
	 of GPUs, you can call ``launch.py`` several times with the desired parameters,
	 just make sure to change the cluster name each time as otherwise it will reconfigure existing clusters.

2. Get and store all Ray Cluster IPs with the following command

	```
	./source/standalone/workflows/ray/grok_cluster_address_with_kubectl.sh
	```

#### Assuming that you have already set up several Ray Clusters with unique IPs

3. If you used Kubernetes/KubeRay, you can skip this step. Otherwise,
	you should create a ``~/.ray_address`` file, that contains the address
	of each ray cluster with the corresponding port. The file should look something like

	```
	# Cluster: isaac-lab-hyperparameter-tuner-1
	10.21.16.30:6379
	# Cluster: isaac-lab-hyperparameter-tuner-2
	10.21.70.164:6379
	```

4. Check that you can issue a test job to all clusters with
	```
	./isaaclab.sh -p source/standalone/workflows/ray/multicluster_submit.py "test_kuberay_config.py --num_jobs 4"
	```

5. Batch submit your desired jobs. Each desired job is paired with the IP addresses
	in the order that they appear.
	```
	./isaaclab.sh -p source/standalone/workflows/ray/multicluster_submit.py "<JOB_0>" "<JOB_1>" "<JOB_N>"
	```

6. Clean up your cluster to conserve resources (follow single cluster steps for each cluster).
	If you used Kubernetes/KubeRay, and you didn't change the default cluster name,
	this can be done with
	```
	kubectl get raycluster | egrep 'hyperparameter-tuner' | awk '{print $1}' | xargs kubectl delete raycluster
	```
##
Notes
https://discuss.ray.io/t/how-to-define-fcnet-hiddens-size-and-number-of-layers-in-rllib-tune/6504/18
