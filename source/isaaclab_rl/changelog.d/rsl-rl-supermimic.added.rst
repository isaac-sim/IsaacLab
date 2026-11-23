* Added typed SuperMimic models, task and algorithm configurations using the standard RSL-RL runner and environment wrapper.
* Added an agent-configured every-step physical observation capture mode before resets and interval events.
* Added an optional deployment-only actor EMA half-life to the SuperMimic algorithm configuration.
* Added the optional SuperMimic recipe settings to the actor, world-model and algorithm configurations, including the
  BlinkMimic settings: a reference-action actor, mirrored world-model batches, Muon and a raised actor learning rate
  with a decay trigger.
