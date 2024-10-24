#TODO: Move me into docs ;p

# Welcome to Isaac-Ray

Ray helps streamline running more than one training run and hyperparameter tuning,
both for parallel and in sequence runs, as well as for both local and cloud-based training.

The Ray integration is useful for the following:
- Several training runs at once in parallel or consecutively with minimal interaction
- Using the same training setup everywhere (on cloud and local) with minimal overhead
- Tuning hyperparameters
- Tuning hyperparameters in parallel on multiple GPUs and/or multiple GPU Nodes
- Simultaneously tuning model hyperparameters for different environments/agents (advanced usage)

# Installation

This guide includes additional dependencies that are not part of the default Isaac Lab install
as this functionality is still experimental.

To use Ray Clusters without kubernetes, like on a local setup,
```kubectl``` is not required. Otherwise, `kubectl` needs to be installed, which can be done from [this link here](https://kubernetes.io/docs/tasks/tools/).

To install all Python dependencies, run

```
./isaaclab.sh -p -m pip install ray[default, tune]==2.31.0
```

# Setup / Cluster Configuration

## Local and Other Setups
#### Option A: With Ray Clusters (Recommended for those less familiar with Ray)
For use on a single machine (node), use the following one liner to start a ray server. This
Ray server will run indefinitely until it is stopped with ```CTRL + C```

```
echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | ./isaaclab.sh -p
```

Alternatively, for use on more than one machine (node), see the following;

On the head machine, run ``ray start --head --port 6379``. On any worker machines,
make sure they can connect to the head machine, and then run
```ray start --address='<HEAD_NODE_IP>:6379'``` . For more info follow the [Ray on-premises bare metal guide](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#on-prem)


#### Option B: With Kubernetes / KubeRay
Spin up a kubernetes cluster with GPU passthrough.
For an example, see the [Minikube GPU Guide](https://docs.wandb.ai/tutorials/minikube_gpu/)

Install ray operator on the local cluster, and make a container where ray is installed.
See ``cluster_configs/Dockerfile`` for an example. Create a ``kuberay.yaml.jinja``
file for the local cluster similar to that of ``cluster_configs/google_cloud/kuberay.yaml.jinja``.
Now, the rest of the steps are the same as for cloud, just make sure to change ``--cluster_host``
to ``local`` when running ``launch.py``

#### Option C: With SLURM

See the [Ray community SLURM support](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-network-ray)
for more information. This guide does not explicitly support SLURM, but it should still be compatible.

## Cloud Setup (Not needed for local development)

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

An example of what a cloud deploy might look look like is in ``cloud_cluster_configs/google_cloud``. Google Cloud is currently
the only one supported out of the box.


# Running Local Experiments
1. Test that the cluster works with

	```
	./isaaclab.sh -p source/standalone/workflows/ray/wrap_isaac_ray_resources.py --test
	```

2. The following examples demonstrate how to submit jobs. If there are more jobs than workers, jobs will be queued up for when resources become available
using the Ray functionality. To runs jobs in parallel with only one node (for example, on a single 4 GPU machine),
resources must be isolated so that there is more than
one worker available, as by default, one worker is created with
all available resources for each cluster node.
For several nodes, resource isolation is not needed to run jobs in parallel.

	***To queue up several jobs, separate jobs by the ```+``` delimiter. Ensure jobs is the last
	argument passed to the wrapper***

	```
	./isaaclab.sh -p source/standalone/workflows/ray/wrap_isaac_ray_resources.py
	--jobs <JOB0>+<JOB1>+<JOB2>
	```

	For example, to submit two jobs, see the following example. ***Note the ```+``` delimiter to separate jobs for specifying several jobs.***
	(To run within a container, replace ```./isaaclab.sh -p``` with ```/workspace/isaaclab/isaaclab.sh -p```
	and ```source/standalone/workflows/rl_games/train.py``` with ```/workspace/isaaclab/source/standalone/workflows/rl_games/train.py```)
	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py --jobs wrap_isaac_ray_resources.py --jobs ./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-Cartpole-v0 --headless+./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras agent.params.config.max_epochs=150
	```

	The following is a manual resource isolation example. The number
	of workers is determined by the total resources
	divided by the resources per job.
	```
	./isaaclab.sh -p source/standalone/workflows/ray/wrap_isaac_ray_resources.py --num_cpu_per_job <CPU> \
	--num_gpu_per_job <GPU> --gb_ram_per_job <RAM> --jobs <JOB0>+<JOB1>
	```

	The following is an automatic resource isolation example, where the resources
	for each node are split into ```num_workers_per_node```

	```
	./isaaclab.sh -p source/standalone/workflows/ray/wrap_isaac_ray_resources.py \
	--num_workers_per_node <NUM_TO_DIVIDE_TOTAL_RESOURCES_BY> \
	--jobs <JOB0>+<JOB1>
	```

# Running Remote Experiments

### Kubernetes / KubeRay Specific (You can skip these steps if you've already set up the Ray Cluster)

1. Start the kubernetes server and verify access

	``kubectl get nodes`` should list available GPU nodes

2. Ensure Ray Operator is installed on the cluster

	``kubectl get crds | grep ray`` should list rayclusters.ray.io , rayjobs.ray.io , and
	rayservices.ray.io

3. Spin up the KubeRay integration from the template cluster configuration file

	See ``./isaaclab.sh -p source/standalone/workflows/ray/launch.py -h``
	for all possible arguments

	For example,

	```
	./isaaclab.sh -p source/standalone/workflows/ray/launch.py
	 --cluster_host google_cloud --namespace <NAMESPACE>  --image <CUSTOM_ISAAC_RAY_IMAGE> --min_workers 4 --max_workers 16
	 ```

4. Check that the KubeRay cluster creation worked with `kubectl get pods` and `kubectl describe pods`.
	It may take a few minutes for the cluster to spin up. If there is an error, or a crash loop backup,
	or the pods are stuck on pending, logs can be inspected with ``kubectl logs <POD_NAME>``. When all pods
	have a ``Running`` status, the cluster is ready to tune hyperparameters.

### Shared Steps for Kubernetes/KubeRay and Ray Clusters

5. Create a ```~/.cluster_config``` file. If the cluster was configured with Kubernetes/KubeRay,
	the following script can be run to automate the config creation.

	```
	./isaaclab.sh -p source/standalone/workflows/ray/grok_cluster_with_kubectl.py
	```

	If the cluster was created in pure Ray, then the config should be created manually with the following contents,
	with a description on a new line for every Ray Cluster

	```
	name: <CLUSTER_NAME> address: http://<RAY_HEAD_IP>.<RAY_DASHBOARD_PORT>
	```

6. Check that that it is possible to issue jobs to the cluster, that all GPUs are available,
	that Ray is installed correctly, nvidia-smi works, and that the needed deps
	for Ray/Isaac Lab are found on the path through the following command.

	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py --test
	```

7. Start the distributed Ray job. See the following examples for training and tuning:

	***For several training runs on the same cluster, separate the jobs by the ```+``` delimiter as
	described for the local steps.*** For more information on using ```wrap_isaac_ray_resources.py```
	see the examples in the local experiments above.

	####  Training
	The following is the format expected for submitting training runs to a cluster.
	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py --jobs wrap_isaac_ray_resources.py --jobs <JOB0>+<JOB1>
	```

	For example,
	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py --jobs \
	wrap_isaac_ray_resources.py --jobs /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/source/standalone/workflows/rl_games/train.py --task Isaac-Cartpole-v0 --headless+/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/source/standalone/workflows/rl_games/train.py --task Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras agent.params.config.max_epochs=150
	```


	#### Tuning
	For example,

	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py --jobs isaac_ray_tune.py --cfg=hyperparameter_tuning/vision_cartpole.py
	```

8. When the distributed job is completed, stop the cluster to conserve resources.

	For Kubernetes/KubeRay, this can be done with

	``kubectl get raycluster | egrep 'hyperparameter-tuner' | awk '{print $1}' | xargs kubectl delete raycluster``

	For clusters, this can be done from the head node with

	``ray stop``

	For the recommended single machine setup, this can also be done through ```CTRL + C``` on the
	terminal running the one-liner ray configuration.

## Retrieving Files/Weights From Remote

Generally, it's best practice to store large files or weights in a storage bucket within the cloud from
the training runs.

This can be achieved by supplying the storage_path option to the ``isaac_ray_tune.py`` job,
as long as the ray cluster has access to the storage bucket.

However, for the sake of prototyping, it is possible to manually retrieve files from KubeRay clusters
with ```kubectl cp```.

## Advanced Usage

Prior to following this section, read both the local experiment and cloud experiment sections.
### Multiple Simultaneous Distributed Runs

To run several distributed training runs at once, use similar commands to a single distributed training run. This requires
the creation or existence of several Ray clusters
(with unique IP addresses), although this can be done atop a single
kubernetes cluster.

This can be used to compare the performance of separate approaches, tuned for fairness
of comparison.
For example, for a MLP agent, and a CNN agent for a task , their
hyperparameters could be simultaneously tuned in parallel with heterogeneous resources.

####  Kubernetes / KubeRay Specific : For Cloud or Kubernetes Local
1. Create several homogeneous ray clusters at once by specifying the ``--num_clusters`` flag to ``launch.py``.
	For example,

	```
	./isaaclab.sh -p source/standalone/workflows/ray/launch.py
	 --cluster_host google_cloud --namespace <NAMESPACE>  --image <CUSTOM_ISAAC_RAY_IMAGE> --min_workers 4 --max_workers 16 --num_clusters 3
	 ```
	 Alternatively for heterogeneous resource allocations, like different numbers
	 of GPUs, you call ``launch.py`` several times with the desired parameters,
	 Make sure to change the suffix after the shared cluster prefix of ```isaac-lab-hyperparameter-tuner```
	 to ensure that new clusters are created as opposed to modifying existing clusters.

2. Get and store all Ray Cluster info with the following command. The following assumes that all clusters
	have the same naming prefix.

	```
	./source/standalone/workflows/ray/grok_cluster_with_kubectl.py
	```

#### Assuming that you have already set up several Ray Clusters with unique IPs

3. For Kubernetes/KubeRay, this step can be skipped. Otherwise, create a ```~/.cluster_config``` file.
	The configuration should have the following contents on a new line for each ray cluster.
	```
	name: <CLUSTER_NAME> address: http://<RAY_HEAD_IP>.<RAY_DASHBOARD_PORT>
	```

4. Check that a test job can be submitted to all clusters with
	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py --test
	```

5. Batch submit the desired jobs. Each desired job is paired with the IP addresses
	in the order that they appear in the ```/.cluster_config```. If there are more jobs than
	clusters, jobs will be matched with clusters in modulus order that they appear. (With clusters c1 and c2,
	and jobs j1 j2 j3 j4, jobs j1 and j3 will be submitted to cluster c1, and jobs j2 and j4 will be submitted to cluster c2.

	***Separate cluster jobs by cluster with the ```*``` delimiter. This is compatible with the earlier mentioned
	```+``` delimiter for running several jobs on each cluster***
	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py wrap_isaac_ray_resources.py --jobs <JOB_0_C0>+<JOB_1_C0>*wrap_isaac_ray_resources.py --jobs <JOB_0_C1>+<JOB_1_C1>*wrap_isaac_ray_resources.py --jobs <JOB_N>"
	```

	For example (take special node of the delimiters, where ```*``` separates clusters, and ```+``` separates
	unique jobs on each cluster),
	```
	./isaaclab.sh -p source/standalone/workflows/ray/submit_isaac_ray_job.py --jobs \
	wrap_isaac_ray_resources.py --jobs /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/source/standalone/workflows/rl_games/train.py --task Isaac-Cartpole-v0 --headless+/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/source/standalone/workflows/rl_games/train.py --task Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras agent.params.config.max_epochs=150*wrap_isaac_ray_resources.py --jobs /workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/source/standalone/workflows/rl_games/train.py --task Isaac-Cartpole-v0 --headless+/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab/source/standalone/workflows/rl_games/train.py --task Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras agent.params.config.max_epochs=150
	```

6. Clean up the cluster to conserve resources

	For example,
	```
	kubectl get raycluster | egrep 'hyperparameter-tuner' | awk '{print $1}' | xargs kubectl delete raycluster
	```
##
Notes for Development

https://discuss.ray.io/t/how-to-define-fcnet-hiddens-size-and-number-of-layers-in-rllib-tune/6504/18
