Fixed
^^^^^

* Fixed a per-step performance regression in :func:`~isaaclab.envs.mdp.events.apply_external_force_torque`
  when the event was configured with all-zero ``force_range`` and ``torque_range`` (a common default
  for tasks that declare the event term but apply no disturbance). The event was unconditionally
  sampling zero wrenches and routing them through the dual-buffer ``WrenchComposer`` introduced in
  PR #5265, paying the full per-step compose-and-apply cost in
  :meth:`~isaaclab.assets.Articulation.write_data_to_sim` for what is semantically a no-op. The
  function now returns early when both ranges are exactly zero. This restores the H1, G1, and
  Anymal-C ``Velocity-Rough`` throughput observed prior to PR #5265. Behaviour for non-zero ranges
  is unchanged.
