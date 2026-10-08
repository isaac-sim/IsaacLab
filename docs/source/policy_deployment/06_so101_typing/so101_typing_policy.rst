.. _so101_typing_policy:

Training and Deploying an SO-101 Keyboard Typing Policy
=======================================================

This walkthrough uses the `SO-101 typing task repository <https://github.com/NVIDIA/so101-typing-task>`_
to train a reinforcement learning policy in Isaac Lab and deploy it to an SO-101
follower through LeRobot. The robot presses letters on a fixed Logitech MX Keys
keyboard with its typing jaw. Training uses PPO with RSL-RL and Newton's
MJWarp solver; hardware inference runs in a separate CPU environment.

The workflow has two entry points: play a released checkpoint in simulation, or
calibrate your own scene and train a policy for it. Physical deployment then
requires alignment evidence for your robot and the policy's trained scene.

.. note::

   **Runtime compatibility.** The commands below use the task repository's pinned
   Docker runtime, which fetches an Isaac Lab source ancestor and applies
   ``runtime/benchmark-runtime.patch``. They do not establish compatibility with
   an unmodified Isaac Lab 3.0.0 installation. See :ref:`so101_typing_ga_integration`
   for the checks needed before documenting a native GA installation.

Policy and deployment contract
------------------------------

The policy uses a frozen A–Z key-position map in the robot-base frame. It has no
camera input and does not estimate a moved keyboard. Each letter passes through
three phases: seek the press, release the key, and lift clear before advancing.
Simulation detects key actuation from the keyboard model; the hardware runner
uses Linux evdev key-down/key-up events and computes tip clearance from measured
joint angles using forward kinematics.

The actor receives 22 values in this exact order:

.. list-table:: Actor observations
   :header-rows: 1
   :widths: 35 10 55

   * - Observation
     - Size
     - Deployment source
   * - Joint position relative to reset
     - 5
     - Encoder readings converted to model coordinates, minus the saved reset
   * - Joint velocity
     - 5
     - Estimated from measured joint motion
   * - Persistent joint command relative to reset
     - 5
     - Controller state, minus the saved reset
   * - Normalized target XYZ
     - 3
     - Active letter's saved position and target-map mean/std
   * - Phase one-hot encoding
     - 3
     - Seek, release, or clearance state
   * - Phase elapsed time
     - 1
     - Time in the active phase, in seconds

The checkpoint's observation normalizer is applied to this vector. The runner
uses the deterministic mean of the native bounded Beta actor, producing five
actions in ``[-1, 1]``. At 25 Hz, it integrates them into a persistent position
command and clips the result to the joint limits:

.. code-block:: text

   joint order = shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
   rate limits = [0.30, 1.10, 0.75, 0.20, 0.10] rad/s
   q_command_next = clip_to_joint_limits(q_command + action * rate_limits * 0.04)

The five outputs control the arm; the moving jaw stays closed. Preserve the
observation order, normalization, action integration, phase transitions and
checkpoint-specific reset/clearance settings when changing the runtime. The
``transit15`` stage uses a 15 mm clearance condition. Hardware clearance is
model-derived rather than independently measured by a camera.

Requirements and installation
-----------------------------

For simulation and training, use Linux x86-64 with an NVIDIA GPU, Docker with
NVIDIA Container Toolkit, host Python 3.10 or later, and ``curl`` or ``wget``.
The repository reports testing on an RTX 5090 with 32 GB VRAM. Its built image is
approximately 32.6 GB; allow at least 70 GB for build layers, caches and runs.
Reduce the number of parallel environments to fit your GPU and record that
change, since it changes the number of samples per PPO update.

For physical use, you also need the SO-101 follower with the supported fixed
typing jaw, Logitech MX Keys keyboard, serial and evdev device access, and
``uv``. The hardware setup script creates a separate Python 3.12 environment with
CPU PyTorch, LeRobot/Feetech support and evdev. Hardware inference does not require
Isaac Sim or a GPU.

The task repository, its linked documentation, runtime dependencies and checkpoint
downloads are public. Clone over HTTPS without a GitHub account or authentication.

Clone the repository, then run all following commands from its root:

.. code-block:: bash

   git clone https://github.com/NVIDIA/so101-typing-task.git
   cd so101-typing-task
   ./so101 build

The first build downloads the pinned dependencies; the first simulation launch
compiles Warp kernels. The launcher retains a user-owned cache for later runs.
Rebuild after changing runtime source. See the repository's
`runtime provenance <https://github.com/NVIDIA/so101-typing-task/blob/main/runtime/README.md>`_
and ``docker/typing-public.Dockerfile`` for exact source and dependency pins.

Try the released policy in simulation
-------------------------------------

This path needs no physical robot. Download the September calibrated-scene
checkpoint and keep its complete bundle, including ``params/``:

.. code-block:: bash

   ./so101 download
   bundle=.artifacts/so101-keyboard-typing-benchmark/releases/calibrated-20260917
   ./so101 video --actuator anchorbench --stage transit15 \
     --checkpoint "$bundle/model_2498.pt" \
     --env-config "$bundle/params/env.yaml" --target NEWTON

Downloads verify pinned checksums. Playback and evaluation use the saved
environment paired with the checkpoint. The released offsets describe the
original setup; they do not calibrate another robot.

For a short installation check that starts fresh training:

.. code-block:: bash

   mkdir -p output/sim-example
   cp configs/tasks/replay-calibrated-20260917.json output/sim-example/task.json
   ./so101 probe --task-config output/sim-example/task.json
   ./so101 train --actuator anchorbench --stage p1a \
     --task-config output/sim-example/task.json --num-envs 64 --iterations 10

Ten updates check startup and output generation; they are too few to expect
successful typing. Add ``--plan`` to a simulation command to inspect its launch
without starting simulation or writing outputs.

Calibrate a scene for your robot
--------------------------------

Complete LeRobot motor calibration before securing the keyboard in its typing
position. Use the same robot ID throughout calibration, recording and deployment:

.. code-block:: bash

   ./so101 setup-hardware
   source .venv-hardware/bin/activate
   lerobot-find-port
   lerobot-calibrate --robot.type=so101_follower \
     --robot.port=/dev/serial/by-id/YOUR_CONTROLLER --robot.id=my_typing_robot

Clamp the base and secure the keyboard on the same flat tabletop, with no mat or
riser. Follow the repository's
`fixture setup <https://github.com/NVIDIA/so101-typing-task/blob/main/FIXTURE_SETUP.md>`_
and `measurement guide <https://github.com/NVIDIA/so101-typing-task/blob/main/docs/release/TASK_CONFIGURATION.md>`_
to save your measured starting scene as ``output/my-fixture/task.json``. Keep the
base, keyboard, typing jaw and motor calibration fixed afterward.

Record hand-guided presses, then jointly fit encoder-to-model offsets and the
effective simulated keyboard pose:

.. code-block:: bash

   ./scripts/real2sim.sh record \
     --robot-id my_typing_robot --port /dev/serial/by-id/YOUR_CONTROLLER \
     --keyboard-device /dev/input/by-id/YOUR_KEYBOARD-event-kbd \
     --task-config output/my-fixture/task.json \
     --seconds 300 --output-root output/my-fixture/recordings

   # Repeat recording to cover A-Z, then fit into a new output directory.
   ./scripts/real2sim.sh fit \
     --recordings output/my-fixture/recordings --out-dir output/my-scene

The recorder disables torque and leaves it off; support the arm while guiding
it. Record complete presses, releases and lifts with the moving jaw closed.
The fit must pass its reset-per-key simulated contact checks before physical-use
training. Keep ``task.json``, ``mapping-report.json`` and the contact evidence
together. See the
`record-and-fit guide <https://github.com/NVIDIA/so101-typing-task/blob/main/docs/release/REAL2SIM.md>`_
for the full procedure and interpretation of failures.

These recordings calibrate the scene; PPO learns from simulation rather than
from the recorded trajectories. The fitted pose is an effective model correction,
so do not move the physical keyboard to match its coordinates. Deployment applies
the fitted offsets in opposite directions for reads and commands, after converting
LeRobot readings into the joint-angle convention used by the model:

.. code-block:: text

   q_model = q_lerobot + offset
   q_lerobot_target = q_model_target - offset

Simulation already uses model coordinates; do not add the offsets again inside
training. Collect a fresh, separate recording for later alignment validation and
keep it out of the fit.

Train and evaluate
------------------

Use the fitted task explicitly. ``p1a`` trains two-letter sequences;
``transit15`` continues to six-letter sequences with the 15 mm transit-clearance
contract. Both use the same 22-input, five-output actor interface.

.. code-block:: bash

   ./so101 train --actuator anchorbench --stage p1a \
     --task-config output/my-scene/task.json \
     --num-envs 4096 --iterations 500 --seed 1307

   # Substitute the checkpoint and matching environment paths printed by training.
   ./so101 evaluate --actuator anchorbench --stage p1a \
     --checkpoint /absolute/path/to/P1A/model_N.pt \
     --env-config /absolute/path/to/P1A/params/env.yaml --num-envs 1024 --seed 2307

   ./so101 train --actuator anchorbench --stage transit15 \
     --checkpoint /absolute/path/to/P1A/model_N.pt \
     --num-envs 4096 --iterations 2000 --seed 1307

Resumed training inherits the checkpoint's task and physics configuration from
``params/``. Keep that entire directory and the same actuator profile through
continuation, evaluation and export. Checkpoint names inherit RSL-RL iteration
indices on resume; use the saved iteration metadata to interpret the budget.

Evaluate the final checkpoint and inspect its presses, releases and clearance:

.. code-block:: bash

   ./so101 evaluate --actuator anchorbench --stage transit15 \
     --checkpoint /absolute/path/to/Transit15/model_N.pt \
     --env-config /absolute/path/to/Transit15/params/env.yaml \
     --num-envs 1024 --seed 2307
   ./so101 video --actuator anchorbench --stage transit15 \
     --checkpoint /absolute/path/to/Transit15/model_N.pt \
     --env-config /absolute/path/to/Transit15/params/env.yaml --target NEWTON

Repeat evaluation with ``--seed 3307``. Review strict press/release/clearance
success, wrong-key events, termination reasons and rollout videos. Simulation
success alone does not establish physical alignment. The repository also offers
``--actuator usd``; both actuator choices use Newton MJWarp. Compare independently
trained profiles with the same task, sample budget and evaluation criteria.

See the `training guide <https://github.com/NVIDIA/so101-typing-task/blob/main/docs/release/TRAINING.md>`_
for the staged supervisor, reports and configuration options.

Qualify alignment, export and deploy
------------------------------------

Follow the repository's
`alignment guide <https://github.com/NVIDIA/so101-typing-task/blob/main/docs/release/ALIGNMENT.md>`_
using the frozen scene and mapping from calibration. The standard path requires:

1. A prepared alignment candidate bound to your robot's calibration and scene.
2. Passing simulated replay of a reserved recording that did not influence fitting.
3. Separate simulated and physical Q/G/P press, release and clearance checks.
4. A ``qualification.json`` produced from the matching replay and probe evidence.

The alignment guide supplies the commands and evidence paths for these steps.
Qualification covers representative alignment; it does not establish full-keyboard
policy reliability. Recalibration, fixture movement or a jaw change requires
requalification. File hashes cannot detect a physically moved keyboard.

Export your evaluated checkpoint with its matching environment and qualification:

.. code-block:: bash

   ./so101 prepare-deployment \
     --checkpoint /absolute/path/to/Transit15/model_N.pt \
     --env-config /absolute/path/to/Transit15/params/env.yaml \
     --robot-id my_typing_robot \
     --alignment output/my-fixture/qualified/qualification.json \
     --out-dir output/my_typing_robot
   ./so101 deploy output/my_typing_robot NEWTON

The second command is a software dry run; it does not connect to the robot.
Export and dry run require the local calibration for the selected robot ID.
The bundle includes the checkpoint, environment, geometry, normalization, reset
target, joint mapping and copied qualification evidence. The runner rechecks
their identities before motor access.

For an inference consistency check, generate a simulation video with
``--trace-policy-actions 400`` and compare the CPU actor to its recorded actions:

.. code-block:: bash

   .venv-hardware/bin/python -m so101_typing.hardware.check_actor_parity \
     --checkpoint output/my_typing_robot/checkpoint.pt \
     --report /absolute/path/to/video/evaluation.json

After reviewing the dry run, an operator can execute the qualified bundle:

.. code-block:: bash

   ./so101 deploy output/my_typing_robot NEWTON --execute \
     --port /dev/serial/by-id/YOUR_CONTROLLER \
     --keyboard-device /dev/input/by-id/YOUR_KEYBOARD-event-kbd

This command moves hardware. Follow the
`operator procedure <https://github.com/NVIDIA/so101-typing-task/blob/main/docs/release/HARDWARE_PREPARATION.md#operator-run-deployment>`_
for reset arrival, confirmation and stopping. Keep the sweep clear and the power
stop accessible. Preserve the hardware logs for review.

.. _so101_typing_ga_integration:

Integration with a native Isaac Lab 3.0 GA runtime
--------------------------------------------------

The pinned runtime provides the reproducible route for the current task. Moving
it into a stock Isaac Lab 3.0.0 checkout is a separate integration task. Before
replacing the installation instructions above with a native GA recipe:

* Port the task package, Gym registrations, assets and launchers to the GA APIs.
  Identify which shared DexSuite, actuator and Newton changes in
  ``benchmark-runtime.patch`` are still required instead of applying the patch
  blindly to a different source revision.
* Validate the Newton MJWarp solver, joint ordering, actuator profiles, contact
  behavior and 25 Hz observation/action contract under the selected GA versions.
* Check startup, a short P1A-to-Transit15 continuation, released-checkpoint
  evaluation and CPU actor parity. Record the runtime versions and resulting
  reports; building successfully does not establish checkpoint equivalence.
* Repeat measured-motion replay and physical alignment qualification before
  reporting deployment equivalence for the new runtime.

This repository exports a task-specific PyTorch bundle and runs its own LeRobot
controller. The commands above do not export a LEAPP bundle or use a ROS inference
node. Either deployment route would require an adapter preserving the same
observation, action, geometry and phase-state contract.

Reported results and their scope
--------------------------------

The repository's September example used 500 P1A updates plus 2,000 Transit15
updates with 4,096 environments. Its recorded result reports 2,048/2,048 strict
simulation successes across two evaluation seeds and one supervised physical
``NEWTON`` sequence completed in approximately 7.10 seconds. That physical trial
used an experimental, unqualified mapping; it was not an execution through the
standard qualification path described above.

The 93/93 per-key calibration checks used fitting recordings, and continuous
measured-motion replay still failed. The result therefore does not establish a
physical success rate, held-out alignment, placement robustness, transfer between
robots, or stock Isaac Lab 3.0 GA compatibility. See the
`recorded evidence and limitations <https://github.com/NVIDIA/so101-typing-task/blob/main/docs/release/calibrated_policy_result.json>`_
when interpreting this example.
