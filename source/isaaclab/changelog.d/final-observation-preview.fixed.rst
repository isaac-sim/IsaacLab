* Corrected ``compute_final_obs`` terminal observations to preview the current successor before
  automatic reset, command updates and interval events. Captures own their storage, clear stale keys
  and preserve built-in observation history, delays, processors and camera frame stacks. This corrects
  the terminal values used by SB3 and TorchRL. Additional stateful reads by custom recorders remain
  outside this guarantee.
* Preserved ordinary camera reads in environment adapters that do not expose the preview flag.
