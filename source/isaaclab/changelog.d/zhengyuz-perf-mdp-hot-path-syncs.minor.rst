Changed
^^^^^^^

* **Breaking:** Converted :func:`~isaaclab.envs.mdp.reset_root_state_uniform` and
  :func:`~isaaclab.envs.mdp.joint_vel_out_of_limit` to class-based manager terms that bake
  their range and joint-index tensors at construction, removing a synchronizing
  host-to-device copy per call. Configurations referencing the terms through
  ``func=mdp.<term>`` are unaffected; direct function calls must construct the term first.
* Changed :meth:`~isaaclab.managers.TerminationManager.reset` to move episodic termination
  statistics to the host in one transfer instead of one synchronizing ``.item()`` per term.
* Changed :class:`~isaaclab.envs.mdp.randomize_physics_scene_gravity` to rewrite its baked
  distribution tensors only when the requested ranges change, removing six small
  host-to-device copies per reset batch.
