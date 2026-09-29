Added
^^^^^

* Added :func:`~isaaclab.utils.math.sample_uniform_from_ranges` to sample named components with
  shared, bounded caching of device bounds. Core and Lift event terms now use this sampler instead
  of maintaining separate tensor-cache helpers.

Changed
^^^^^^^

* Changed episode logging in :class:`~isaaclab.managers.CommandTerm`,
  :class:`~isaaclab.managers.TerminationManager`, :class:`~isaaclab.managers.CurriculumManager`,
  :class:`~isaaclab.envs.mdp.UniformPoseCommand`, :class:`~isaaclab.envs.mdp.UniformPose2dCommand` and
  :class:`~isaaclab.envs.mdp.survival_success_rate` to report 0-d device tensors instead of Python floats,
  matching :class:`~isaaclab.managers.RewardManager`, so resets no longer synchronize the stream. Code that
  reads these ``extras["log"]`` entries as floats should call ``float()`` or ``.item()`` on them.
* Removed per-step host synchronizations from the heading control of
  :class:`~isaaclab.envs.mdp.UniformVelocityCommand` and from
  :class:`~isaaclab.envs.mdp.actions.task_space_actions.DifferentialInverseKinematicsAction`.
* Cached the device range tensors of the root-state, nodal-state, push and center-of-mass event terms by
  value instead of uploading them on every call.
* Batched the per-term updates of :class:`~isaaclab.managers.RewardManager`, and skipped the defensive copy
  before the built-in noise functions and after delay buffers in :class:`~isaaclab.managers.ObservationManager`.
* Shifted :class:`~isaaclab.utils.buffers.CircularBuffer` histories of small frames in two kernels
  regardless of the history length.

Fixed
^^^^^

* Fixed :class:`~isaaclab.envs.mdp.reset_root_state_uniform` ignoring the ``pose_range`` and ``velocity_range``
  passed at call time, including curriculum updates, in favor of the ranges present at construction.
