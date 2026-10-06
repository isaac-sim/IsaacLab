* Moved SO-101 cube resets and commanded lift positions from a region centered 16 cm
  from the base to one centered 27 cm away, preserving the 5 cm sampling width.
  Existing checkpoints retained their action and observation dimensions but required
  retraining for the new workspace.
* Reduced the SO-101 PPO training budget from 2,400 to 800 updates after observing
  convergence within 400-800 updates in the outward workspace. Retained intermediate
  checkpoints for policy selection because further training did not consistently improve quality.
