* Moved world-model reset record sampling and state updates to the simulation device, deferring host metadata reads
  until chunk generation. The reset event now passes a device boolean mask to ``WorldModelManager.reset_envs``.
  Training uses a seeded Torch generator, so the sampled sequence differs from the previous NumPy sampler.
