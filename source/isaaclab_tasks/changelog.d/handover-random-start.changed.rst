* Changed ``Isaac-Shadow-Handover-Direct`` to start each episode with the object in a random hand
  (``randomize_start_hand``). Set ``randomize_start_hand=False`` for the previous fixed right-to-left start.
* Changed ``Isaac-Shadow-Handover-Direct`` and ``Isaac-Shadow-Handover`` to start the object at the holding hand's
  goal point, 3 cm closer to the palm than before, and renamed the Direct ``goal_position_offset`` to
  ``hold_position_offset`` since it now sets both points.
