* **Breaking:** Required noise functions and models configured through
  :class:`~isaaclab.utils.noise.NoiseCfg` and :class:`~isaaclab.utils.noise.NoiseModelCfg`
  to leave their input unchanged. :class:`~isaaclab.managers.ObservationManager` now calls
  them without a defensive input copy. Custom callbacks using in-place operations must
  clone their input first (for example, ``data.clone().add_(bias)``) or use out-of-place
  operations (``data + bias``). Returning the unchanged input or a view remains supported;
  the manager still protects borrowed outputs during subsequent processing.

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
  before noise callbacks and after delay buffers in :class:`~isaaclab.managers.ObservationManager`.
* Shifted :class:`~isaaclab.utils.buffers.CircularBuffer` histories of small frames in two kernels
  regardless of the history length.
