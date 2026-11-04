* Added a Policy Ghosts mode to ``isaaclab play`` that is switched on and off from the Newton visualizer's HUD
  without a restart. It uses the first ``--policy_ghost_envs`` (default 4) environments of each of
  ``--policy_ghost_slots`` (default 3) slots from the running simulation, hides the other environments and gives them
  zero actions, and resets every environment when it is switched off. A run with fewer environments than the slots need
  still offers the restart.
