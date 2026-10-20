* Fixed the assemble trocar task never advancing its stage. The ``update_stage`` reward term had weight 0, which
  the reward manager skips, so no stage reward or success termination was ever triggered.
