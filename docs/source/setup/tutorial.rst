.. _tutorial-so101-vial-placement:

Tutorial
========

.. figure:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/tutorial_so101_vialplace_play.gif
   :alt: The SO-101 arm picking up a vial and placing it in a rack.
   :align: center
   :width: 85%

.. note::

   This tutorial ties in with the `NVIDIA Sim-to-Real SO-101 learning course
   <https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html>`__ and its
   `workshop repository <https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop>`__, which present a broader
   end-to-end physical AI workflow with the SO-101, Isaac Lab, and NVIDIA Isaac GR00T.

This tutorial builds a complete robot-learning workflow around an `SO-101 arm
<https://github.com/TheRobotStudio/SO-ARM100>`__ that picks up a vial and places it in a rack.
Instead of assembling disconnected examples, you will work with a production-shaped downstream project:
`IsaacLabTutorial main branch <https://github.com/isaac-sim/IsaacLabTutorial/tree/main>`__. The project includes the
robot and workshop assets, a manager-based reinforcement-learning environment, state and wrist-camera observations,
RSL-RL agent configurations, tests, state-policy training, and state-to-vision policy distillation.

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

      Train a state teacher, distill a wrist-camera policy, and measure success.

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

Clone the tutorial's ``main`` branch and create its environment:

.. code-block:: bash

   git clone https://github.com/isaac-sim/IsaacLabTutorial.git
   cd IsaacLabTutorial
   uv sync

The project pins Isaac Lab's ``develop`` branch through ``pyproject.toml``, so it does not require a sibling Isaac Lab
checkout. ``uv sync`` creates the environment and installs the tutorial, Isaac Lab, and their runtime dependencies.

This repository is also a complete example of the downstream project produced by the
:doc:`Isaac Lab template generator </source/developer-tools/template_generator>`. Use the generator when starting your
own project, then follow the same package layout and task-discovery pattern shown here.

Confirm that the shared CLI discovers the downstream package:

.. code-block:: bash

   uv run isaaclab --help
   uv run python -c 'import importlib.metadata; print([e.name for e in importlib.metadata.entry_points(group="isaaclab.tasks")])'

The second command should include ``so101-vial-place``. Task discovery works because the project publishes an
``isaaclab.tasks`` entry point:

.. code-block:: toml
   :caption: pyproject.toml

   [project.entry-points."isaaclab.tasks"]
   so101-vial-place = "isaaclab_tutorial.tasks"

Importing that package registers three Gymnasium task IDs:

.. list-table::
   :header-rows: 1
   :widths: 43 57

   * - Task ID
     - Policy observations
   * - ``IsaacTutorial-Place-Vial-SO101``
     - Robot, vial, rack, placement, and progress state
   * - ``IsaacTutorial-Place-Vial-SO101-Camera``
     - Direct-from-scratch PPO with 64 by 48 wrist RGB plus proprioception
   * - ``IsaacTutorial-Place-Vial-SO101-Camera-Distillation``
     - State-to-vision distillation with the same deployed camera observations

.. tip::

   Run all remaining commands from the ``IsaacLabTutorial`` directory. Train the state task first: it is the
   baseline and supplies the teacher checkpoint used by the distillation task.

.. _tutorial-so101-project:

2. Tour the project
-------------------

The project keeps reusable task logic separate from SO-101-specific configuration:

.. code-block:: text

   IsaacLabTutorial/
   |-- pyproject.toml                     package metadata and task-discovery entry point
   |-- media/                              reference rollout
   |-- src/isaaclab_tutorial/
   |   |-- assets/                        workshop and reset-pose data
   |   |-- tasks/place_vial/
   |   |   |-- mdp/                       actions, events, observations, rewards, and progress logic
   |   |   |-- reset/                     reset dataset, curriculum, and generation logic
   |   |   `-- config/so101/
   |   |       |-- __init__.py            Gymnasium task registration
   |   |       |-- env_cfg.py             scene, MDP, and physics configuration
   |   |       |-- camera_env_cfg.py      wrist-camera task variant
   |   |       `-- agents/                PPO and distillation configurations and models
   |   `-- utils/                         exact rollout evaluation helpers
   `-- tests/                              behavioral and configuration contracts

This boundary is intentional. A new robot gets a sibling of ``config/so101`` and reuses the vial-placement
MDP. A different manipulation problem gets a sibling of ``tasks/place_vial`` and owns its task terms.

The registration connects a task ID to the generic manager-based environment, the task configuration, and its
default RSL-RL agent:

.. code-block:: python
   :caption: src/isaaclab_tutorial/tasks/place_vial/config/so101/__init__.py

   gym.register(
       id="IsaacTutorial-Place-Vial-SO101",
       entry_point="isaaclab.envs:ManagerBasedRLEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": f"{_PACKAGE}.env_cfg:SO101VialEnvCfg",
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
   :caption: src/isaaclab_tutorial/tasks/place_vial/config/so101/env_cfg.py

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
           self.is_finite_horizon = False
           self.sim.dt = 1.0 / 120.0
           self.sim.physics = PhysicsCfg()

The 120 Hz simulation and decimation of four give the policy a 30 Hz control rate. ``replicate_physics=True``
allows the same scene to scale across thousands of environments.

Scene and control
~~~~~~~~~~~~~~~~~

``SO101SceneCfg`` contains a fixed-base SO-101, a free 20 g vial, the rack, a collision mat, contact sensors on
both jaws and the vial, and a dome light. The robot configuration, identified actuator dynamics, and limits come
from Isaac Lab's ``SO101_CFG``.

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
     - No object state: randomized 64 by 48 wrist RGB plus noisy joint state, targets, and previous action.
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
and replaces only the actor observations. Its asymmetric critic retains full state during direct PPO training.
``SO101VialCameraDistillationEnvCfg`` additionally exposes the state teacher and a training-only geometry target;
the deployed student still consumes only wrist RGB and proprioception.

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

Train the state teacher with RSL-RL. The task's default agent configuration is an 800-iteration run:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 uv run isaaclab train --rl_library rsl_rl \
     --task IsaacTutorial-Place-Vial-SO101 \
     --num_envs 4096 --max_iterations 800 --seed 42 \
     --run_name so101_vial_seed42 --device cuda:0 \
     --visualizer none presets=newton_mjwarp

Checkpoints and TensorBoard events are written below ``logs/rsl_rl/so101_vial_state/<run>``; the final checkpoint is
``model_799.pt``. While training runs, inspect the learning curves in another terminal:

.. code-block:: bash

   uv run tensorboard --logdir logs/rsl_rl/so101_vial_state

Look for rising episode completion metrics, not reward alone. A larger return can come from shaping without a
corresponding increase in successful releases.

Play the state teacher
~~~~~~~~~~~~~~~~~~~~~~

Play the trained teacher from the canonical home phase:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl \
     --task IsaacTutorial-Place-Vial-SO101 \
     --num_envs 1 --checkpoint /path/to/state_model.pt --deterministic \
     --visualizer newton presets=newton_mjwarp

The rollout should start from the canonical home phase, pick up the horizontal vial, turn it upright, move it over
the target hole, insert it, and open the jaw so gravity seats it in the rack.

Distill the wrist-camera policy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the finished teacher to the dedicated distillation task. Add the renderer preset because the student observes
wrist RGB:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 uv run isaaclab train --rl_library rsl_rl \
     --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
     --num_envs 1024 --max_iterations 800 --seed 42 \
     --checkpoint /path/to/state_teacher.pt \
     --run_name wrist_distillation_seed42 --device cuda:0 \
     --visualizer none presets=newton_mjwarp,newton_renderer

Distillation is single-GPU. Its bounded replay DAgger runner begins with teacher trajectories, gradually adds student
recovery states, and retains a 25 percent teacher-action floor. A training-only geometry head supplies a dense
localization target, and sparse stochastic weight averaging stabilizes the final checkpoint. Outputs are written below
``logs/rsl_rl/so101_vial_camera_distillation/<run>``.

Play the distilled student with the same task ID used for training:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl \
     --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
     --num_envs 1 --checkpoint /path/to/distilled_model.pt --deterministic \
     --visualizer newton presets=newton_mjwarp,newton_renderer

Measure the policies
~~~~~~~~~~~~~~~~~~~~

A visually convincing rollout is not an evaluation. Run the tracked phase-zero start set headlessly with the
project's external callback:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl \
     --task IsaacTutorial-Place-Vial-SO101 \
     --num_envs 1024 --checkpoint /path/to/state_model.pt --deterministic \
     --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
     --visualizer none presets=newton_mjwarp

Use the same 1,024-start contract for the distilled policy:

.. code-block:: bash

   uv run isaaclab play --rl_library rsl_rl \
     --task IsaacTutorial-Place-Vial-SO101-Camera-Distillation \
     --num_envs 1024 --checkpoint /path/to/distilled_model.pt --deterministic \
     --external_callback isaaclab_tutorial.utils.evaluation.install_episode_counter \
     --visualizer none presets=newton_mjwarp,newton_renderer

The callback runs each tracked start once and prints one ``SO101_EVAL_RESULT`` JSON record. Preserve that record,
the checkpoint, seed, task ID, and resolved configuration when comparing experiments. For reference, the results
reported on the tutorial's ``main`` branch are 99.4 percent success for the state teacher and 67.6--72.9 percent
success across repeated distilled-policy audits, with no unsafe rack contacts.

Maintain the reset dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~

The checked-in reset dataset is ready for training. Generate or inspect a separate candidate without overwriting it:

.. code-block:: bash

   uv run generate-so101-resets \
     --output checkpoints/reset_poses.pt --device cuda:0 \
     --visualizer none presets=newton_mjwarp

   uv run view-so101-resets \
     --dataset checkpoints/reset_poses.pt --device cuda:0 \
     --visualizer newton presets=newton_mjwarp

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
     - ``config/so101/env_cfg.py`` and ``assets/``
   * - Change actor inputs
     - Observation groups in ``env_cfg.py`` or ``camera_env_cfg.py``
   * - Change task behavior
     - Reusable functions and manager terms in ``mdp/``
   * - Change reset coverage
     - ``reset/curriculum.py`` and the validated reset dataset
   * - Change PPO or distillation
     - ``config/so101/agents/rsl_rl_ppo_cfg.py``, ``rsl_rl_distillation_cfg.py``, and ``distillation.py``
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
