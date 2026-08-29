.. _tutorials:
.. _tutorial-so101-vial-placement:

Build and Train a Vial-Placement Task
======================================

This tutorial builds a complete robot-learning workflow around an `SO-101 arm
<https://github.com/TheRobotStudio/SO-ARM100>`__ that picks up a vial and places it in a rack.
Instead of assembling disconnected examples, you will work with a production-shaped downstream project:
`IsaacLabTutorial <https://github.com/isaac-sim/IsaacLabTutorial>`__. The project includes the robot and
workshop assets, a manager-based reinforcement-learning environment, state and wrist-camera observations,
RSL-RL agent configurations, tests, and a trained state-policy checkpoint.

.. raw:: html

   <div style="text-align: center; margin: 1.5rem 0;">
     <video controls autoplay loop muted playsinline style="width: 100%; max-width: 960px;">
       <source src="../_static/tutorials/so101_vial_placement.mp4" type="video/mp4">
       Your browser does not support embedded videos.
     </video>
   </div>

By the end, you will know how a downstream Isaac Lab task is packaged, discovered, validated, trained,
and evaluated. You will also know where to change the robot, scene, MDP, and learning configuration for
your own task.

Navigate the tutorial
---------------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: **1. Set up**
      :link: tutorial-so101-setup
      :link-type: ref

      Install the project and verify task discovery.

   .. grid-item-card:: **2. Tour the project**
      :link: tutorial-so101-project
      :link-type: ref

      Find the assets, task terms, configurations, and tests.

   .. grid-item-card:: **3. Understand the task**
      :link: tutorial-so101-task
      :link-type: ref

      Connect the scene, actions, observations, resets, rewards, and physics.

   .. grid-item-card:: **4. Validate**
      :link: tutorial-so101-validate
      :link-type: ref

      Run tests, a zero-agent smoke test, and a benchmark.

   .. grid-item-card:: **5. Train and evaluate**
      :link: tutorial-so101-train
      :link-type: ref

      Train with RSL-RL, play a checkpoint, and measure success.

   .. grid-item-card:: **6. Make it yours**
      :link: tutorial-so101-extend
      :link-type: ref

      Adapt the task without tangling reusable and robot-specific code.

.. _tutorial-so101-setup:

1. Set up the project
---------------------

You need Python 3.12, `uv <https://docs.astral.sh/uv/>`__, Git, and an NVIDIA GPU. The tutorial uses
Newton with the MJWarp solver, so the primary workflow does not require the Isaac Sim application.
Review the :ref:`system requirements <installation-system-requirements>` before starting.

The tutorial's ``pyproject.toml`` expects ``IsaacLab`` and ``IsaacLabTutorial`` to be sibling directories.
From the directory where you keep repositories, run:

.. code-block:: bash

   git clone https://github.com/isaac-sim/IsaacLab.git
   git clone https://github.com/isaac-sim/IsaacLabTutorial.git
   cd IsaacLabTutorial
   uv sync

If you already have an Isaac Lab source checkout, clone only ``IsaacLabTutorial`` beside it and run ``uv sync``
there.

``uv sync`` creates the project environment and installs both the tutorial and the adjacent Isaac Lab source
packages in editable mode. Confirm that the shared CLI discovers the downstream package:

.. code-block:: bash

   uv run isaaclab --help
   uv run python -c 'import importlib.metadata; print([e.name for e in importlib.metadata.entry_points(group="isaaclab.tasks")])'

The second command should include ``so101-vial-place``. Task discovery works because the project publishes an
``isaaclab.tasks`` entry point:

.. code-block:: toml
   :caption: pyproject.toml

   [project.entry-points."isaaclab.tasks"]
   so101-vial-place = "so101_vial_place.tasks"

Importing that package registers two Gymnasium task IDs:

.. list-table::
   :header-rows: 1
   :widths: 43 57

   * - Task ID
     - Policy observations
   * - ``IsaacTutorial-Place-Vial-SO101``
     - Robot, vial, rack, placement, and progress state
   * - ``IsaacTutorial-Place-Vial-SO101-Camera``
     - 64 by 64 wrist RGB plus proprioception, with privileged state for the critic

.. tip::

   Run all remaining commands from the ``IsaacLabTutorial`` directory. Use the state task first: it trains
   faster, is easier to inspect, and has a reference checkpoint in ``checkpoints/model.pt``.

.. _tutorial-so101-project:

2. Tour the project
-------------------

The project keeps reusable task logic separate from SO-101-specific configuration:

.. code-block:: text

   IsaacLabTutorial/
   |-- pyproject.toml                     package metadata and task-discovery entry point
   |-- checkpoints/model.pt               trained state-policy checkpoint
   |-- src/so101_vial_place/
   |   |-- assets/                        robot, vial, rack, mat, and reset-pose data
   |   `-- tasks/place_vial/
   |       |-- mdp/                       actions, observations, rewards, and progress logic
   |       |-- reset/                     reset dataset, curriculum, and generation logic
   |       `-- config/so101/
   |           |-- __init__.py            Gymnasium task registration
   |           |-- state_env_cfg.py       scene and state-task configuration
   |           |-- camera_env_cfg.py      wrist-camera task variant
   |           |-- control.py             robot command conventions
   |           |-- physics.py             contact-material configuration
   |           `-- agents/                RSL-RL models and PPO settings
   `-- tests/                              behavioral and configuration contracts

This boundary is intentional. A new robot gets a sibling of ``config/so101`` and reuses the vial-placement
MDP. A different manipulation problem gets a sibling of ``tasks/place_vial`` and owns its task terms.

The registration connects a task ID to the generic manager-based environment, the task configuration, and its
default RSL-RL agent:

.. code-block:: python
   :caption: src/so101_vial_place/tasks/place_vial/config/so101/__init__.py

   gym.register(
       id="IsaacTutorial-Place-Vial-SO101",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": f"{_PACKAGE}.state_env_cfg:SO101VialEnvCfg",
           "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101StatePPORunnerCfg",
           "default_agent": "rsl_rl",
       },
   )

This is why the normal ``isaaclab train``, ``play``, and ``benchmark`` commands work without project-local
launcher scripts.

.. _tutorial-so101-task:

3. Understand the task
----------------------

The environment is assembled declaratively with :class:`~isaaclab.envs.ManagerBasedRLEnvCfg`. The main
configuration connects one scene configuration and one configuration for each MDP manager:

.. code-block:: python
   :caption: src/so101_vial_place/tasks/place_vial/config/so101/state_env_cfg.py

   @configclass
   class SO101VialEnvCfg(ManagerBasedRLEnvCfg):
       scene = SO101SceneCfg(num_envs=4096, env_spacing=0.9, replicate_physics=True)
       actions = ActionsCfg()
       observations = ObservationsCfg()
       events = DatasetEventsCfg()
       rewards = RewardsCfg()
       terminations = TerminationsCfg()

       def __post_init__(self):
           self.decimation = 4
           self.episode_length_s = 20.0
           self.sim.dt = 1.0 / 120.0
           self.sim.physics = PhysicsCfg()

The 120 Hz simulation and decimation of four give the policy a 30 Hz control rate. ``replicate_physics=True``
allows the same scene to scale across thousands of environments.

Scene and control
~~~~~~~~~~~~~~~~~

``SO101SceneCfg`` contains a fixed-base SO-101, a free 20 g vial, the rack, a collision mat, contact sensors on
both jaws and the vial, and a dome light. Identified actuator dynamics and limits come from the robot USD; the
task deliberately does not replace them in Python.

The six-dimensional policy action contains five bounded, relative arm-joint position increments and one bounded
relative gripper increment. The task never attaches the vial to the gripper and never writes its pose after reset.
Grasping, transport, insertion, and release must therefore succeed through simulated contact.

MDP at a glance
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Component
     - Contract
   * - Actions
     - Five arm-joint increments, scaled to 0.033 rad per control step, and one jaw increment.
   * - State actor
     - Joint state and targets, previous action, end-effector state, vial state, rack-relative target,
       placement features, and irreversible progress flags.
   * - Camera actor
     - No object state: randomized 64 by 64 wrist RGB plus noisy joint state, targets, and previous action.
   * - Critic
     - Privileged state observations plus physical contact state.
   * - Events
     - Reset-pose sampling and modest vial mass and friction randomization.
   * - Rewards
     - Compact object-to-goal shaping, physical milestone rewards, a success bonus, loss penalty, and small
       action-rate and joint-velocity costs.
   * - Terminations
     - Successful placement, a lost vial, unstable robot state, or the 20 s time limit.

Physical progress and resets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Long-horizon manipulation is difficult to learn from the initial tabletop pose alone. The project therefore
replays physics-validated reset poses across eight phases:

.. code-block:: text

   home -> pregrasp -> grasp -> lift -> reorient -> transport -> insert -> release

Training samples these phases as a reset curriculum. Each phase remains part of the same task: observations,
actions, rewards, and termination criteria do not change. Progress flags advance only when physical evidence is
present—for example, bilateral jaw contact near the intended grasp point and a measured load-bearing lift.
Interactive play returns to phase-zero starts so the final policy must complete the entire task.

The active reward is intentionally small. Most shaping candidates remain configured with zero weight so experiments
can be reproduced without hiding behavior in ad hoc code. Read ``mdp/terms.py`` for the physical predicates and
``reset/curriculum.py`` for the reset distribution.

State and camera policies
~~~~~~~~~~~~~~~~~~~~~~~~~

``SO101VialEnvCfg`` is the fully observed baseline. ``SO101VialCameraEnvCfg`` subclasses it, adds the wrist camera,
and replaces only the actor observations. Its asymmetric critic retains full state during training. The associated
RSL-RL configuration replaces the state MLP actor with a small CNN while preserving the shared environment and MDP.

.. _tutorial-so101-validate:

4. Validate before training
----------------------------

Start with fast checks. They catch broken assets, registration, geometry, rewards, reset data, and configuration
contracts before a long GPU run:

.. code-block:: bash

   uv run pytest -q
   uv run ruff check .

Next, construct and step a small vectorized environment with zero actions:

.. code-block:: bash

   uv run isaaclab zero_agent \
     --task IsaacTutorial-Place-Vial-SO101 \
     --num_envs 8 --visualizer none presets=newton_mjwarp

If this fails, fix task discovery, asset loading, or simulation stability before tuning RL. When it passes, measure
the environment separately from the learning algorithm:

.. code-block:: bash

   uv run isaaclab benchmark runtime \
     --task IsaacTutorial-Place-Vial-SO101 \
     --num_envs 4096 --num_steps 1000 --warmup_steps 50 \
     --visualizer none presets=newton_mjwarp

Reduce ``--num_envs`` if the batch does not fit your GPU. For the camera task, add the Newton renderer preset and
start with a smaller batch:

.. code-block:: bash

   uv run isaaclab benchmark runtime \
     --task IsaacTutorial-Place-Vial-SO101-Camera \
     --num_envs 1024 --num_steps 1000 --warmup_steps 50 \
     --visualizer none presets=newton_mjwarp,newton_renderer

.. _tutorial-so101-train:

5. Train and evaluate
---------------------

Train the state policy with RSL-RL. The command below uses the tutorial's default agent configuration but caps the
run at 800 iterations for the guided workflow:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 uv run isaaclab train --rl_library rsl_rl \
     --task IsaacTutorial-Place-Vial-SO101 \
     --num_envs 4096 --max_iterations 800 --seed 42 \
     --run_name so101_vial_seed42 --device cuda:0 \
     --visualizer none presets=newton_mjwarp

Checkpoints and TensorBoard events are written below ``logs/rsl_rl/so101_vial_state``. While training runs, inspect
the learning curves in another terminal:

.. code-block:: bash

   uv run tensorboard --logdir logs/rsl_rl/so101_vial_state

Look for rising episode completion metrics, not reward alone. A larger return can come from shaping without a
corresponding increase in successful releases.

Play a policy
~~~~~~~~~~~~~

Use the included checkpoint first to verify the complete pipeline, then replace the checkpoint path with one from
your run:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl \
     --task IsaacTutorial-Place-Vial-SO101 \
     --num_envs 1 --checkpoint checkpoints/model.pt --deterministic \
     --visualizer newton presets=newton_mjwarp

The rollout should start from the canonical home phase, pick up the horizontal vial, turn it upright, move it over
the target hole, insert it, and open the jaw so gravity seats it in the rack.

Measure the policy
~~~~~~~~~~~~~~~~~~

A visually convincing rollout is not an evaluation. Run the tracked phase-zero start set headlessly with the
project's external callback:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl \
     --task IsaacTutorial-Place-Vial-SO101 \
     --num_envs 1024 --checkpoint checkpoints/model.pt --deterministic \
     --external_callback so101_vial_place.utils.evaluation.install_episode_counter \
     --visualizer none presets=newton_mjwarp

The callback runs each tracked start once and prints one ``SO101_EVAL_RESULT`` JSON record. Preserve that record,
the checkpoint, seed, task ID, and resolved configuration when comparing experiments.

.. _tutorial-so101-extend:

6. Make it yours
----------------

Change one boundary at a time and rerun the tests, zero-agent smoke test, and benchmark before training:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Goal
     - Start here
   * - Change the scene or SO-101
     - ``config/so101/state_env_cfg.py``, ``control.py``, ``physics.py``, and ``assets/``
   * - Change actor inputs
     - Observation groups in ``state_env_cfg.py`` or ``camera_env_cfg.py``
   * - Change task behavior
     - Reusable functions and manager terms in ``mdp/``
   * - Change reset coverage
     - ``reset/curriculum.py`` and the validated reset dataset
   * - Change the network or PPO
     - ``config/so101/agents/rsl_rl_ppo_cfg.py``
   * - Add another robot
     - Add ``config/<robot_name>/`` and register new task IDs there; keep the shared MDP robot-agnostic.
   * - Add another manipulation task
     - Add a sibling of ``tasks/place_vial`` with its own MDP and robot configurations.

Keep these invariants as the task evolves:

* Task IDs resolve through the package entry point—do not copy Isaac Lab's launcher scripts.
* Policy action order and scale match the robot command contract.
* Physical quantities and quaternions use Isaac Lab's current conventions; Isaac Lab 3 uses XYZW quaternion order.
* Resets create valid physical states, and post-reset task code does not teleport the manipulated object.
* Success requires a released, mechanically seated vial rather than proximity to the rack.
* Tests cover observable task contracts, while a benchmark checks that changes did not move expensive work into the
  per-environment step loop.

From here, use :ref:`Task Design Workflows <feature-workflows>` for the manager/direct architecture,
:ref:`Reinforcement Learning <rl-frameworks>` for framework details, and
:doc:`Hydra Configuration System </source/features/hydra>` for command-line overrides.
