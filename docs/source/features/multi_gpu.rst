Multi-GPU and Multi-Node Training
=================================

.. currentmodule:: isaaclab

Isaac Lab supports multi-GPU and multi-node reinforcement learning. Currently, this feature is only
available for RL-Games and skrl libraries workflows. We are working on extending this feature to
other workflows.

.. attention::

    Multi-GPU and multi-node training is only supported on Linux. Windows support is not available at this time.
    This is due to limitations of the NCCL library on Windows.


Multi-GPU Training
------------------

For complex reinforcement learning environments, it may be desirable to scale up training across multiple GPUs.
This is possible in Isaac Lab through the use of the
`PyTorch distributed <https://pytorch.org/docs/stable/distributed.html>`_ framework or the
`JAX distributed <https://jax.readthedocs.io/en/latest/jax.distributed.html>`_ module respectively.

In PyTorch, the :meth:`torch.distributed` API is used to launch multiple processes of training, where the number of
processes must be equal to or less than the number of GPUs available. Each process runs on
a dedicated GPU and launches its own instance of Isaac Sim and the Isaac Lab environment.
Each process collects its own rollouts during the training process and has its own copy of the policy
network. During training, gradients are aggregated across the processes and broadcasted back to the process
at the end of the epoch.

In JAX, since the ML framework doesn't automatically start multiple processes from a single program invocation,
the skrl library provides a module to start them.

.. image:: ../_static/multi-gpu-rl/a3c-light.svg
    :class: only-light
    :align: center
    :alt: Multi-GPU training paradigm
    :width: 80%

.. image:: ../_static/multi-gpu-rl/a3c-dark.svg
    :class: only-dark
    :align: center
    :width: 80%
    :alt: Multi-GPU training paradigm

|

To train with multiple GPUs, use the following command, where ``--nproc_per_node`` represents the number of available GPUs:

.. tab-set::
    :sync-group: rl-train

    .. tab-item:: rl_games
        :sync: rl_games

        .. code-block:: shell

            python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --distributed

    .. tab-item:: skrl
        :sync: skrl

        .. tab-set::

            .. tab-item:: PyTorch
                :sync: torch

                .. code-block:: shell

                    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/skrl/train.py --task=Isaac-Cartpole-v0 --headless --distributed

            .. tab-item:: JAX
                :sync: jax

                .. code-block:: shell

                    python -m skrl.utils.distributed.jax --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/skrl/train.py --task=Isaac-Cartpole-v0 --headless --distributed --ml_framework jax

Multi-Node Training
-------------------

To scale up training beyond multiple GPUs on a single machine, it is also possible to train across multiple nodes.
To train across multiple nodes/machines, it is required to launch an individual process on each node.

For the master node, use the following command, where ``--nproc_per_node`` represents the number of available GPUs, and
``--nnodes`` represents the number of nodes:

.. tab-set::
    :sync-group: rl-train

    .. tab-item:: rl_games
        :sync: rl_games

        .. code-block:: shell

            python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --distributed

    .. tab-item:: skrl
        :sync: skrl

        .. tab-set::

            .. tab-item:: PyTorch
                :sync: torch

                .. code-block:: shell

                    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 scripts/reinforcement_learning/skrl/train.py --task=Isaac-Cartpole-v0 --headless --distributed

            .. tab-item:: JAX
                :sync: jax

                .. code-block:: shell

                    python -m skrl.utils.distributed.jax --nproc_per_node=2 --nnodes=2 --node_rank=0 --coordinator_address=ip_of_master_machine:5555 scripts/reinforcement_learning/skrl/train.py --task=Isaac-Cartpole-v0 --headless --distributed --ml_framework jax

Note that the port (``5555``) can be replaced with any other available port.

For non-master nodes, use the following command, replacing ``--node_rank`` with the index of each machine:

.. tab-set::
    :sync-group: rl-train

    .. tab-item:: rl_games
        :sync: rl_games

        .. code-block:: shell

            python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=ip_of_master_machine:5555 scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --distributed

    .. tab-item:: skrl
        :sync: skrl

        .. tab-set::

            .. tab-item:: PyTorch
                :sync: torch

                .. code-block:: shell

                    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=ip_of_master_machine:5555 scripts/reinforcement_learning/skrl/train.py --task=Isaac-Cartpole-v0 --headless --distributed

            .. tab-item:: JAX
                :sync: jax

                .. code-block:: shell

                    python -m skrl.utils.distributed.jax --nproc_per_node=2 --nnodes=2 --node_rank=1 --coordinator_address=ip_of_master_machine:5555 scripts/reinforcement_learning/skrl/train.py --task=Isaac-Cartpole-v0 --headless --distributed --ml_framework jax

For more details on multi-node training with PyTorch, please visit the
`PyTorch documentation <https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html>`_.
For more details on multi-node training with JAX, please visit the
`skrl documentation <https://skrl.readthedocs.io/en/latest/api/utils/distributed.html>`_ and the
`JAX documentation <https://jax.readthedocs.io/en/latest/multi_process.html>`_.

.. note::

    As mentioned in the PyTorch documentation, "multi-node training is bottlenecked by inter-node communication
    latencies". When this latency is high, it is possible multi-node training will perform worse than running on
    a single node instance.
