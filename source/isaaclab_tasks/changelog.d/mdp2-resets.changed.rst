* Changed ``Isaac-Lift-KukaAllegro-MDP2`` to reset through ``isaaclab_experimental.mdp2.mdp.reset_from_bank`` with the
  MDP1 lift sampler settings and, as MDP1, a bank of 1024 states per object shared by the worlds that spawned it,
  replacing its task-local reset bank.
