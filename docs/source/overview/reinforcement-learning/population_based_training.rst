.. _population-based-training:

Population-Based Training
=========================

Population-based training (PBT) is an online optimization loop around regular reinforcement learning (RL)
training. Instead of choosing one set of hyperparameters and training one policy, PBT trains a population of
independent policies on the same task. It periodically ranks them by a task-level objective. An underperforming
learner copies a promising learner's checkpoint, mutates selected hyperparameters, and continues training from
the copied weights.

This cycle combines two ideas:

* **Explore:** different seeds and hyperparameters let the learners discover different behaviors.
* **Exploit:** checkpoint replacement redirects compute from stalled learners to behaviors that are already working.

Unlike a fixed random or grid search, PBT can change hyperparameters during training and reuse the weights from a
successful run. The final artifact is still a normal policy checkpoint; PBT changes how it is trained, not how it
is deployed.

.. seealso::

   If you are new to RL workflows in Isaac Lab, first read :ref:`tutorial-run-rl-training` and
   :ref:`tutorial-configure-rl-training`.

.. caution::

   Isaac Lab currently supports PBT only with the **RL-Games** library.


When to Use PBT
---------------

PBT is most useful when:

* independent seeds frequently converge to very different results or stall in local optima;
* a task has a stable scalar success metric that can rank policies while they train;
* useful hyperparameters may need different values in early and late training; and
* enough compute is available to train several policies concurrently.

Start with ordinary single-policy training when a known configuration converges reliably, when compute only
supports one or two runs, or when policies cannot be compared with a meaningful objective. PBT adds checkpoint
I/O and coordination overhead, and a population of *N* learners consumes approximately the aggregate compute of
*N* ordinary runs. A fair experiment therefore compares PBT with the same number of independent fixed-configuration
runs, not with one run.

PBT is also different from :doc:`multi-GPU training </source/features/multi_gpu>`. Distributed data-parallel
training uses several devices to update one policy. PBT gives each learner its own policy, optimizer state,
experience, and checkpoint, and only exchanges checkpoints during selection.


Example Where PBT Was Enabling
------------------------------

The `DexPBT study`_ tested contact-rich manipulation tasks with large observation and action spaces, sparse task
success, and many locally useful behaviors that did not lead to a complete solution. For its PBT-versus-PPO
comparison, the authors held aggregate compute constant: the best learner in an eight-policy PBT population was
compared with the best of eight independent PPO seeds, with every learner trained for five billion environment
steps.

.. list-table:: Single-arm object reorientation comparison reported in Figure 4 of the `DexPBT study`_
   :header-rows: 1
   :widths: 20 38 42

   * - Condition
     - Training behavior
     - Outcome under the study budget
   * - PPO without PBT
     - Eight policies trained independently with fixed hyperparameters; discoveries were not shared.
     - None of the eight runs reached the final 1 cm success tolerance.
   * - PPO with PBT
     - Eight learners periodically shared promising weights and explored mutated hyperparameters.
     - The population reached the final tolerance and continued improving the number of consecutive successes.

The study reported improvement in every tested scenario and found PBT to be an enabling factor for non-trivial
performance in three of them. In the harder dual-arm reorientation task, its best PBT policy reached almost 40 of
50 consecutive successes. See the paper's `training curves`_ and `supplementary policy videos`_ for the full
comparison and learned behaviors.

No task mathematically requires PBT. Here, "enabling" means that fixed PPO did not solve the task within the tested
budget while PBT did. These experiments used the Isaac Gym DexPBT tasks; they motivate the Isaac Lab workflow below
but are not benchmark results for the included Shadow Hand example.


How Isaac Lab Selects Policies
------------------------------

Every ``interval_steps``, each learner saves its checkpoint and current objective to a shared workspace. For the
initialized policies, let the objective mean and standard deviation be :math:`\mu` and :math:`\sigma`. Isaac Lab
computes the leader and underperformer cuts as:

.. math::

   U = \max(\mu + \mathtt{threshold\_std}\,\sigma,\ \mu + \mathtt{threshold\_abs})

.. math::

   L = \min(\mu - \mathtt{threshold\_std}\,\sigma,\ \mu - \mathtt{threshold\_abs})

Policies with an objective greater than :math:`U` are leaders; policies below :math:`L` are underperformers. Only
an underperformer is changed. It copies a randomly selected leader and mutates each whitelisted hyperparameter with
probability ``mutation_rate``. If there is no leader, it keeps its own checkpoint and mutates its hyperparameters.
All other policies continue unchanged.

The learners coordinate through checkpoint and YAML metadata files rather than a central process. They may progress
asynchronously: a learner compares checkpoints from the latest PBT iteration available at or before its own. Every
learner must still use the same ``num_policies`` and must see the shared directory at the same absolute path because
checkpoint paths stored in the metadata are absolute.


Choose the Ranking Objective
----------------------------

The objective controls selection pressure, so choose a metric that measures the behavior you ultimately want and
is comparable across the population. Higher values are always considered better. A task-success metric is usually
more suitable than a dense shaped reward whose scale changes during training.

The included reorientation configuration uses:

.. code-block:: yaml

   objective: episode.consecutive_successes

This dotted path resolves to ``infos["episode"]["consecutive_successes"]``. The RL-Games wrapper remaps
``extras["log"]`` to ``infos["episode"]``, so use the ``episode.`` prefix even when the environment writes the
metric to ``extras["log"]``.

Before a long run, verify that the objective is present, scalar, and increases when policy behavior improves. A
noisy or non-stationary objective can repeatedly replace useful policies. PBT population members also stop being
independent after checkpoint exchange, so evaluate the selected checkpoint separately across held-out seeds and
conditions.


Configure PBT
-------------

The ``Isaac-Reorient-Cube-Shadow-Direct`` RL-Games configuration provides a complete PBT example. PBT is disabled
by default; the launch command in the next section enables it.

.. literalinclude:: ../../../../source/isaaclab_tasks/isaaclab_tasks/core/reorient/config/shadow_hand/agents/rl_games_ppo_cfg.yaml
   :language: yaml
   :start-at: pbt:

The main controls are:

``directory`` and ``workspace``
   Identify the shared storage root and the PBT run within that root. Use a new workspace for each population.

``num_policies`` and ``policy_idx``
   Define the population size and this learner's unique zero-based index.

``objective``
   Selects the scalar from the environment info dictionary used to rank learners.

``interval_steps``
   Sets the number of environment steps between checkpoint comparisons. Shorter intervals react sooner but cause
   more checkpoint I/O and give a mutated learner less time to adapt.

``threshold_std`` and ``threshold_abs``
   Control how far a policy must be from the population mean to be classified as a leader or underperformer. If no
   policy crosses both effective margins, the population continues without replacement.

``mutation_rate``, ``change_range``, and ``mutation``
   Control which hyperparameters may change and how often. ``mutate_float`` multiplies or divides by a random factor
   in ``change_range``; ``mutate_discount`` conservatively changes the distance from 1.0 for discount-like values.
   Parameters absent from ``mutation`` remain fixed.


Launch the Population
---------------------

Start one process per policy. The example below launches policy 0 of the default eight-policy population. Run it in
eight terminals or scheduler jobs, changing only ``PBT_POLICY_IDX`` from 0 through 7. Keep ``PBT_NUM_POLICIES``,
``PBT_DIRECTORY``, and ``agent.pbt.workspace`` identical for every process.

.. code-block:: bash

   export PBT_POLICY_IDX=0
   export PBT_NUM_POLICIES=8
   export PBT_DIRECTORY=/absolute/path/to/shared_folder

   uv run --extra rl-games isaaclab train --rl_library rl_games \
     --task Isaac-Reorient-Cube-Shadow-Direct \
     --num_envs 8192 \
     --seed="${PBT_POLICY_IDX}" \
     agent.pbt.enabled=true \
     agent.pbt.num_policies="${PBT_NUM_POLICIES}" \
     agent.pbt.policy_idx="${PBT_POLICY_IDX}" \
     agent.pbt.directory="${PBT_DIRECTORY}" \
     agent.pbt.workspace=shadow_reorient_pbt

Assign each process a GPU with your scheduler or ``CUDA_VISIBLE_DEVICES`` when running several workers on one
host. Weights & Biases tracking is optional; if enabled, give each worker a unique run name while keeping the same
project.

During training, each policy writes numbered ``.pth`` checkpoints and matching ``.yaml`` metadata under its policy
directory in the shared workspace. The metadata records the objective and mutated parameter values. At the end of
the run, select the checkpoint with the best validated objective, then evaluate it as a normal RL-Games policy.


Troubleshooting
---------------

* **No policies are replaced:** the population may not yet have crossed the configured margins. Inspect the
  objectives before reducing the thresholds; identical low scores can also mean the ranking metric is uninformative.
* **An objective lookup fails:** verify the dotted path against the keys emitted in ``infos["episode"]``.
* **Workers cannot load one another's checkpoints:** confirm that every worker uses the same absolute shared path,
  workspace name, and population size.
* **Training spends too much time checkpointing:** increase ``interval_steps`` or place the workspace on faster
  shared storage.


References
----------

* Jaderberg et al., `Population Based Training of Neural Networks`_ (2017).
* Petrenko et al., `DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training`_
  (2023).

.. _Population Based Training of Neural Networks: https://arxiv.org/abs/1711.09846
.. _DexPBT study: https://arxiv.org/abs/2305.12127
.. _DexPBT\: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training: https://arxiv.org/abs/2305.12127
.. _training curves: https://arxiv.org/pdf/2305.12127#page=7
.. _supplementary policy videos: https://sites.google.com/view/dexpbt
