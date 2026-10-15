* Changed ``Isaac-Shadow-Handover-Direct`` to start each episode with the object in a random hand
  (``randomize_start_hand``) and to penalize changes of each hand's actions between steps
  (``action_rate_penalty_scale``). Set ``randomize_start_hand=False`` and ``action_rate_penalty_scale=0.0`` for
  the previous behavior.
