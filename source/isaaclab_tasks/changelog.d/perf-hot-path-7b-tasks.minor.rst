Changed
^^^^^^^

* Stored the joint, body, and tendon indices of the Reorient, Cabinet, and locomotion direct tasks as
  device tensors, and precomputed their static joint limits and default arm pose, removing host-to-device
  index uploads and CUDA synchronizations from every physics substep. The public
  ``actuated_dof_indices``, ``finger_bodies``, ``finger_wrench_bodies``, ``arm_joint_ids``, and
  ``finger_joint_ids`` attributes, and the indices returned by
  :func:`~isaaclab_tasks.core.reorient.utils.resolve_actuated_tendons`, are now ``torch.long`` tensors
  instead of lists. Call ``.tolist()`` where a list is needed.
* Switched the Reorient, Cabinet, and locomotion direct tasks from the deprecated articulation target
  setters to ``articulation.actuators.target_command``.
* Logged the Cabinet, locomotion direct, Lift deformable and cable, and Fourbar pole success metrics as
  0-d device tensors instead of synchronizing with ``.item()`` on every reset. Call ``.item()`` where a
  float is needed.
* Changed :meth:`~isaaclab_tasks.core.reorient.utils.EpisodeErrorRecorder.reset` to compute its statistics
  without a host synchronization. Once any error has been recorded, it returns NaN statistics instead of an
  empty dictionary when none of the selected environments has a sample. Check the values with
  :func:`torch.isnan` instead of testing for missing keys.
* Stacked the Lift contact sensor forces so :func:`~isaaclab_tasks.core.lift.mdp.contacts` and
  :func:`~isaaclab_tasks.core.lift.mdp.contact_count` take one norm per call instead of one per sensor.
* Precomputed the ANYmal symmetry augmentation as one cached column permutation and sign tensor per
  transform, removing the per-minibatch index and sign uploads (about 1.5 ms to 0.15 ms per call).
* Sent only the marker indices to Lift success markers on static assets each step, cached the locomotion
  walk-target offset and the Lift reset position bounds on the device, and read the Reorient goal directly
  from the command buffers instead of concatenating the command on every access.
