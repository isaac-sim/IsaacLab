* Added :attr:`~isaaclab.managers.TerminationTermCfg.invalid_transition` for physics faults and other
  corrupted transitions. Such terms terminate their environments, zero every reward contribution before
  episodic accumulation, and zero the affected terminal-observation rows. Other transitions retain their values.
