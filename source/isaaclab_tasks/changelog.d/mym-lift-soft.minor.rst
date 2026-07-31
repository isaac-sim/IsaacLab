Added
^^^^^

* Added a task-local ``mdp`` package for the Franka deformable lift environments, providing
  deformable-aware rewards, observations, terminations, events, curricula, and a pose command that
  tracks the deformable center of mass.
* Added an ``ik`` action preset to the Franka deformable lift environments that drives the arm with
  an absolute end-effector pose through a differential inverse-kinematics controller. Select it
  with ``presets=ik``.
* Added a gravity curriculum to the Franka deformable lift environments that linearly ramps the
  vertical gravity up to -9.81 m/s^2 over the first environment steps, so the policy learns to
  grasp before it has to hold the object up. The soft-beam tasks ramp from near zero over 10000
  steps, the cloth tasks from -1.0 m/s^2 over 20000 steps.
* Added terminations for a diverged solve (non-finite deformable or robot state) and for joint
  velocities beyond the simulation limits, so unrecoverable environments reset instead of poisoning
  the rollout.

Changed
^^^^^^^

* **Breaking:** Changed the Franka deformable lift environments to select their action space
  through a :class:`~isaaclab_tasks.utils.PresetCfg`, with ``joint`` (relative joint-position arm
  targets plus a limit-rescaled gripper) as the new default. ``Isaac-Lift-Soft-Franka`` and
  ``Isaac-Lift-Soft-Franka-Camera`` previously used absolute task-space differential inverse
  kinematics, and ``Isaac-Lift-Cloth-Franka`` and ``Isaac-Lift-Cloth-Franka-Camera`` previously
  used absolute joint-position targets. Append ``presets=ik`` to the run command to get the
  task-space inverse-kinematics action space back.
* **Breaking:** Changed the rsl_rl ``experiment_name`` of ``Isaac-Lift-Soft-Franka`` and
  ``Isaac-Lift-Cloth-Franka`` from ``franka_deformable`` to ``franka_soft``. New runs are written to
  ``logs/rsl_rl/franka_soft``; move existing ``logs/rsl_rl/franka_deformable`` run directories there
  to resume from an older checkpoint.
* Changed the Franka deformable lift environments to simulate under real gravity: gravity is no
  longer disabled on the robot, and the vertical gravity of ``Isaac-Lift-Soft-Franka`` and
  ``Isaac-Lift-Soft-Franka-Camera`` is no longer zeroed.
* Changed the simulation step of the Franka deformable lift environments from 1/60 s to 1/120 s
  with a decimation of 4, which halves the policy control rate from 60 Hz to 30 Hz, and raised the
  default number of environments of ``Isaac-Lift-Soft-Franka`` from 128 to 2048.
* Changed the table of the Franka deformable lift environments from the ``SeattleLabTable`` USD
  asset to an invisible cuboid collider whose top surface sits at ``z = 0``. The goal command's
  success visualizer draws the table instead, tinted by whether the goal is reached.
* Changed the rsl_rl PPO configuration of ``Isaac-Lift-Soft-Franka`` and
  ``Isaac-Lift-Cloth-Franka`` to the actor/critic model configurations, using
  :class:`~isaaclab_rl.rsl_rl.RslRlMLPModelCfg` with observation normalization and explicit
  ``obs_groups``, a learning rate of 1e-3, and ``max_iterations`` lowered from 50000 to 5000.
* Retuned ``Isaac-Lift-Soft-Franka`` and ``Isaac-Lift-Soft-Franka-Camera`` for stable grasping: a
  stiffer and denser beam, a smaller particle radius, explicit collider contact and rest offsets,
  and full-surface rigid-soft contact with signed-distance fields on the gripper.
* Retuned the Franka arm and hand actuator gains of the Franka deformable lift environments, adding
  realistic armature and a slower, weaker gripper so it settles on the object instead of crushing
  it. This replaces the previous per-task gripper overrides, so the cloth tasks now use the same
  gains as the soft-beam tasks.
* Changed ``Isaac-Lift-Cloth-Franka`` to place the cloth over a kinematic rigid support that is
  registered with the coupler, so the cloth no longer passes through it.
