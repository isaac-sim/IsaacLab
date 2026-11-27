* Changed ``Isaac-Lift-KukaAllegro-MDP2`` to reset through ``isaaclab_experimental.mdp2.mdp.reset_to_states`` with the
  MDP1 lift sampler settings and, as MDP1, 1024 reset states per object shared by the worlds that spawned it,
  replacing its task-local reset state bank.
