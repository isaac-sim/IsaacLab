* **Breaking:** Changed Franka Reach and Reach-OSC to continuous pose tracking: success remained a
  reported metric but no longer ended the episode or awarded the terminal success bonus. Episodes
  ran until timeout. Requalify existing checkpoints because reward totals and episode lengths changed.
