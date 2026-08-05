Changed
^^^^^^^

* **Breaking:** Aligned the direct-workflow ant and humanoid environments with their manager-based
  twins, so ``Isaac-Ant-Direct``/``Isaac-Ant`` and ``Isaac-Humanoid-Direct``/``Isaac-Humanoid`` now
  define the same MDP and converge to the same reward. The direct environments gained feet joint
  wrench observations, reset joint randomization, and ``step_dt``-scaled rewards, and adopted the
  gear-weighted energy and joint-limit penalties. Existing direct-workflow checkpoints are not
  compatible: ``observation_space`` grew from 36 to 60 (ant) and from 75 to 87 (humanoid), and
  ``episode_length_s`` changed from 15.0 to 16.0.
* **Breaking:** Changed ``joint_gears`` on the ant and humanoid direct environment configurations
  from a backend-ordered list (or a ``{"physx": [...], "newton": [...]}`` dict) to a dict keyed by
  joint name expression. Migrate by replacing the ordered list with regex entries, e.g.
  ``joint_gears = [15, 15, 15, 15, 15, 15, 15, 15]`` becomes ``joint_gears = {".*": 15.0}``. The
  gears are now resolved by joint name, so a single table is correct for every physics backend.
* **Breaking:** Removed the unused ``contact_force_scale`` field from the ant and humanoid direct
  environment configurations, then reintroduced it as the scale applied to the new feet wrench
  observations.
* Changed the walk target used by the manager-based locomotion terms to be relative to each
  environment origin. Previously the target was an absolute world position shared by every
  environment, so robots in different environments aimed along slightly different directions.

Added
^^^^^

* Added :class:`~isaaclab_tasks.core.locomotion.mdp.rewards.survival_success_rate` and registered it
  as a zero-weight reward term on the manager-based ant and humanoid tasks, so the
  ``Metrics/success_rate`` log is no longer emitted as a side effect of
  :class:`~isaaclab_tasks.core.locomotion.mdp.rewards.progress_reward`.
* Added a terminal ``terminating`` penalty to the manager-based ant and humanoid tasks, matching the
  death cost the direct environments already applied.

Fixed
^^^^^

* Fixed the manager-based ant and humanoid tasks rejecting ``presets=ovphysx`` with
  ``Unknown preset(s): ovphysx``. Their physics preset configurations were missing the ``ovphysx``
  entry that the direct-workflow configurations already declared. The direct configurations now also
  set ``bounce_threshold_velocity=0.2`` on the Isaac Sim PhysX preset, matching the manager-based
  tasks instead of leaving the 0.5 default.
* Fixed the manager-based ant and humanoid tasks applying unbounded joint efforts, which drove the
  MJWarp solver to produce ``NaN`` articulation states and aborted humanoid training partway through
  a run. The joint effort action is now clipped at the gear magnitude, matching the clamp the direct
  environments already applied to their actions.
* Fixed :class:`~isaaclab_tasks.core.locomotion.humanoid.agents.rsl_rl_ppo_cfg.HumanoidDirectPPORunnerCfg`
  overriding the learning rate, target KL, and value loss coefficient of the manager-based humanoid
  runner config, so the two workflows trained at different rates despite sharing an MDP. It now
  inherits the algorithm settings and only overrides the experiment name, as the ant config does.
* Fixed :class:`~isaaclab_tasks.core.locomotion.mdp.rewards.progress_reward` computing the distance
  to the target without zeroing the vertical component on reset, which made the potential recorded
  at reset inconsistent with every subsequent step.
