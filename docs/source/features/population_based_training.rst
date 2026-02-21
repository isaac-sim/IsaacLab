Population Based Training
=========================

What PBT Does
-------------

* Trains *N* policies in parallel (a "population") on the **same task**.
* Every ``interval_steps``:

  #. Save each policy's checkpoint and objective.
  #. Score the population and identify **leaders** and **underperformers**.
  #. For underperformers, replace weights from a random leader and **mutate** selected hyperparameters.
  #. Restart that process with the new weights/params automatically.

Leader / Underperformer Selection
---------------------------------

Let ``o_i`` be each initialized policy's objective, with mean ``μ`` and std ``σ``.

Upper and lower performance cuts are::

  upper_cut = max(μ + threshold_std * σ, μ + threshold_abs)
  lower_cut = min(μ - threshold_std * σ, μ - threshold_abs)

* **Leaders**: ``o_i > upper_cut``
* **Underperformers**: ``o_i < lower_cut``

The "Natural-Selection" rules:

1. Only underperformers are acted on (mutated or replaced).
2. If leaders exist, replace an underperformer with a random leader; otherwise, self-mutate.

Mutation (Hyperparameters)
--------------------------

* Each param has a mutation function (e.g., ``mutate_float``, ``mutate_discount``, etc.).
* A param is mutated with probability ``mutation_rate``.
* When mutated, its value is perturbed within ``change_range = (min, max)``.
* Only whitelisted keys (from the PBT config) are considered.

Example Config
--------------

.. code-block:: yaml

   pbt:
     enabled: True
     policy_idx: 0
     num_policies: 8
     directory: .
     workspace: "pbt_workspace"
     objective: episode.Curriculum/difficulty_level
     interval_steps: 50000000
     threshold_std: 0.1
     threshold_abs: 0.025
     mutation_rate: 0.25
     change_range: [1.1, 2.0]
     mutation:
       agent.params.config.learning_rate: "mutate_float"
       agent.params.config.grad_norm: "mutate_float"
       agent.params.config.entropy_coef: "mutate_float"
       agent.params.config.critic_coef: "mutate_float"
       agent.params.config.bounds_loss_coef: "mutate_float"
       agent.params.config.kl_threshold: "mutate_float"
       agent.params.config.gamma: "mutate_discount"
       agent.params.config.tau: "mutate_discount"


``objective: episode.Curriculum/difficulty_level`` is the dotted expression that uses
``infos["episode"]["Curriculum/difficulty_level"]`` as the scalar to **rank policies** (higher is better).
With ``num_policies: 8``, launch eight processes sharing the same ``workspace`` and unique ``policy_idx`` (0-7).


Launching PBT
-------------

You must start **one process per policy** and point them to the **same workspace**. Set a unique
``policy_idx`` for each process and the common ``num_policies``.

Minimal flags you need:

* ``agent.pbt.enabled=True``
* ``agent.pbt.directory=<path/to/shared_folder>``
* ``agent.pbt.policy_idx=<0..num_policies-1>``

.. note::
   All processes must use the same ``agent.pbt.workspace`` so they can see each other's checkpoints.

.. caution::
   PBT is currently supported **only** with the **rl_games** library. Other RL libraries are not supported yet.

Tips
----

* Keep checkpoints reasonable: reduce ``interval_steps`` only if you really need tighter PBT cadence.
* Use larger ``threshold_std`` and ``threshold_abs`` for greater population diversity.
* It is recommended to run 6+ workers to see benefit of pbt.


Training Example
----------------

We provide a reference PPO config here for task:
`Isaac-Dexsuite-Kuka-Allegro-Lift-v0 <https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dexsuite/config/kuka_allegro/agents/rl_games_ppo_cfg.yaml>`_.
For the best logging experience, we recommend using wandb for the logging in the script.

Launch *N* workers, where *n* indicates each worker index:

.. code-block:: bash

   # Run this once per worker (n = 0..N-1), all pointing to the same directory/workspace
   ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
     --seed=<n> \
     --task=Isaac-Dexsuite-Kuka-Allegro-Lift-v0 \
     --num_envs=8192 \
     --headless \
     --track \
     --wandb-name=idx<n> \
     --wandb-entity=<**entity**> \
     --wandb-project-name=<**project**>
     agent.pbt.enabled=True \
     agent.pbt.num_policies=<N> \
     agent.pbt.policy_idx=<n> \
     agent.pbt.workspace=<**pbt_workspace_name**> \
     agent.pbt.directory=<**/path/to/shared_folder**> \


References
----------

This PBT implementation reimplements and is inspired by *Dexpbt: Scaling up dexterous manipulation for hand-arm systems with population based training* (Petrenko et al., 2023).

.. code-block:: bibtex

   @article{petrenko2023dexpbt,
     title={Dexpbt: Scaling up dexterous manipulation for hand-arm systems with population based training},
     author={Petrenko, Aleksei and Allshire, Arthur and State, Gavriel and Handa, Ankur and Makoviychuk, Viktor},
     journal={arXiv preprint arXiv:2305.12127},
     year={2023}
   }
