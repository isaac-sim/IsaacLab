#TODO: Move me into docs ;p

# Ray Integration

Through using Ray, we streamline distributed training runs.

The Ray integration is useful to you if any of the following apply:
- You want to use the same training setup everywhere (on cloud and local) with minimal overhead
- You want to tune models' hyperparameters as fast as possible in parallel on multiple GPUs
	and/or multiple GPU Nodes
- You want to run several training runs at once
- You want to simultaneously tune model hyperparameters for different environments/agents (see
	advanced usage)


Notably, this Ray integration is able to leverage the existing Isaac Lab Hydra support
for changing hyperparameters. See ``hyperparameter_tuning/config/vision_cartpole.py``
for a demonstration of how easy hyperparameter tuning can be when leveraging Isaac-Ray and Hydra.

# Installation

This guide includes additional dependencies that are not part of the default Isaac Lab install
as this functionality is still largely experimental.

***You likely need to install `kubectl`*** , which can be done from [this link here](https://kubernetes.io/docs/tasks/tools/).

Note that if you are using Ray Clusters without kubernetes, like on a local setup,
this dependency is not needed.

To install all Python dependencies, run

```
./isaaclab.sh -p -m pip install ray[default, tune]==2.31.0
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
#### Option A: With Ray Clusters

If you have one machine, you can the following one liner to start a ray server. This
Ray server will run indefinitely until it is stopped with ```CTRL + C```

```
echo "import ray; ray.init(); import time; print('Ray is running...'); [time.sleep(10) for _ in iter(int, 1)]" | ./isaaclab.sh -p
```

Alternatively, if you have more than one machine,

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

# Running Local Experiments
1. Test that your cluster works

```
./isaaclab.sh -p source/standalone/workflows/ray/wrap_isaac_ray_resources.py --cluster_cpu_count CPU_FOR_RAY \
--cluster_gpu_count GPU_FOR_RAY --cluster_ram_gb RAM_FOR_RAY --num_workers 1 --test
```

2. Submit jobs in the following fashion. You can also use this functionality to isolate the amount
of resources that are used for Isaac Lab.
```
./isaaclab.sh -p source/standalone/workflows/ray/wrap_isaac_ray_resources.py --cluster_cpu_count CPU_FOR_RAY \
--cluster_gpu_count GPU_FOR_RAY --cluster_ram_gb RAM_FOR_RAY --num_workers 1 --commands '<DESIRED_JOB>'
```
If you are using any sort of command line arguments, separate them with a ; delimiter. For example;

```
./isaaclab.sh -p source/standalone/workflows/ray/wrap_isaac_ray_resources.py --cluster_cpu_count 8 \
--cluster_gpu_count 1 --cluster_ram_gb 16 --num_workers 1 \
 --commands './isaaclab.sh;-p;source/standalone/workflows/rl_games/train.py;--task;Isaac-Cartpole-v0;--headless'
```



You can also specify more than one job to run in parallel if you have more than one GPU. For example,
```
./isaaclab.sh -p source/standalone/workflows/ray/wrap_isaac_ray_resources.py --cluster_cpu_count CPU_FOR_RAY \
--cluster_gpu_count 2 --cluster_ram_gb RAM_FOR_RAY --num_workers 2 --commands <DESIRED_JOB>
```

# Running Remote Experiments

### Kubernetes / KubeRay Specific (You can skip these steps if you've already set up the Ray Cluster)

1. Start your kubernetes server and verify you have access

	``kubectl get nodes`` should list your nodes

2. Ensure Ray Operator is installed on your cluster

	``kubectl get crds | grep ray`` should list rayclusters.ray.io , rayjobs.ray.io , and
	rayservices.ray.io

3. Spin up your KubeRay integration from the template cluster configuration file

	See ``./isaaclab.sh -p source/standalone/workflows/ray/launch.py -h``
	for all possible arguments

	For example, you could invoke with:

	```
	./isaaclab.sh -p source/standalone/workflows/ray/launch.py
	 --cluster_host google_cloud --namespace <NAMESPACE>  --image <CUSTOM_ISAAC_RAY_IMAGE> --min_workers 4 --max_workers 16
	 ```

4. Check that your KubeRay cluster worked with `kubectl get pods` and `kubectl describe pods`.
	It may take a few minutes for the cluster to spin up. If there is an error, or a crash loop backup,
	you can inspect the logs further with ``kubectl logs <POD_NAME>``. When all pods
	say ``Running`` as their status, the cluster is ready to tune hyperparameters.

### Shared Steps for Kubernetes/KubeRay and Ray Clusters

5. Create a ```~/.cluster_config``` file. If you configured your cluster with Kubernetes/KubeRay,
	you can run the following script and move on to the next step. This command may
	take a moment to run.

	```
	./isaaclab.sh -p source/standalone/workflows/ray/grok_cluster_with_kubectl.py
	```

	If your cluster was created in pure Ray, you must create the file manually with the following contents, one on each
	line for every Ray Cluster. If you are developing locally, you likely only have one ray cluster,
	and only need one line in this file. If you only have 1 gpu, then you can only have 1 worker.

	```
	name: <CLUSTER_NAME> address: http://<RAY_HEAD_IP>.<RAY_DASHBOARD_PORT> num_cpu: <TOTAL_CLUSTER_CPU_COUNT> num_gpu: <TOTAL_CLUSTER_GPU_COUNT> rm_gb: <TOTAL_GIGABYTES_RAM> num_workers: <NUM_WORKERS>
	```

6. Check that you can issue jobs to the cluster, that all GPUs are available,
	that Ray is installed correctly, nvidia-smi works, and that the needed deps
	for Ray/Isaac Lab are found on path.

	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py "wrap_isaac_ray_resources.py --test"
	```

7. Define your desired Ray job as a script on your local machine.
   	For a hyperparameter tuning job, see ```hyper_parameter_tuning/config/vision_cartpole.py```

8. Start your distributed Ray job.

	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py "wrap_isaac_ray_resources.py --commands <YOUR_JOB_HERE>"
	```

	For example,

	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py "isaac_ray_tune.py --cfg=hyper_parameter_tuning/config/vision_cartpole.py"
	```

8. When you have completed your distributed job, stop the cluster to conserve resources.

	If you are using Kubernetes/KubeRay, this can be done with

	``kubectl delete raycluster <CLUSTER_NAME> -n <NAMESPACE>``

	If you are using Ray clusters, this can be done from the head node with

	``ray stop``

# Retrieving Files/Weights From Remote

Generally, it's best practice to store large files or weights in a storage bucket within the cloud from
the training runs.

You can do this by supplying the storage_path option to the ``isaac_ray_tune.py`` job.

However, for the sake of prototyping, if you want to retrieve files from a Kubernetes/KubeRay
cluster, this is possible.

List all pods with ```kubectl get pods```. Use ```kubectl cp``` to fetch information from the GPU enabled pods.

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
	 just make sure to change the cluster name each time as otherwise it will reconfigure existing clusters. Make sure that all pods
         are running before completing the next step with ``kubectl get pods``

2. Get and store all Ray Cluster info with the following command.

	```
	./source/standalone/workflows/ray/grok_cluster_with_kubectl.py
	```

#### Assuming that you have already set up several Ray Clusters with unique IPs

3. If you used Kubernetes/KubeRay, you can skip this step. OtherwiseCreate a ```~/.cluster_config``` file.
	If your cluster was created in pure Ray, you must create the file manually with the following contents, one on each
	line for every Ray Cluster.
	```
	name: <CLUSTER_NAME> address: http://<RAY_HEAD_IP>.<RAY_DASHBOARD_PORT> num_cpu: <TOTAL_CLUSTER_CPU_COUNT> num_gpu: <TOTAL_CLUSTER_GPU_COUNT>
	```

4. Check that you can issue a test job to all clusters with
	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py "wrap_isaac_ray_resources.py --test"
	```

5. Batch submit your desired jobs. Each desired job is paired with the IP addresses
	in the order that they appear.
	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py "wrap_isaac_ray_resources.py --commands <JOB_0>" "wrap_isaac_ray_resources.py --commands <JOB_1>" "wrap_isaac_ray_resources.py --commands <JOB_N>"
	```

	For example if you have three clusters, and would like to tune three cartpole variants at once,

	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py "isaac_ray_tune.py --cfg=hyper_parameter_tuning/config/vision_cartpole.py --tune_type standard" "isaac_ray_tune.py --cfg=hyper_parameter_tuning/config/vision_cartpole.py --tune_type resnet" "isaac_ray_tune.py --cfg=hyper_parameter_tuning/config/vision_cartpole.py --tune_type theia"
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
