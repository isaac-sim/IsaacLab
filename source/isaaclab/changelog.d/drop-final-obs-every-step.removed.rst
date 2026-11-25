* Removed the unreleased ``final_obs_every_step`` option of manager-based, direct and multi-agent environments.
  Learned dynamics in ``rsl-rl-lib-extras`` 6.0.0.2 takes successors from ``extras["final_obs"]`` of finished
  episodes, so environments only capture final observations before resets.
