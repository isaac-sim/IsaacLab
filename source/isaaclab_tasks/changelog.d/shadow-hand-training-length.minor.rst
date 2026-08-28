Changed
^^^^^^^

* Reduced the default RSL-RL training length for the Shadow Hand tasks: reorientation from 10000 to
  3000 iterations and handover from 5000 to 3500. Success rate flattens well before the previous
  budgets, so a default run reaches the same success rate in roughly a third of the wall time. Pass
  ``agent.max_iterations=<n>`` to train longer.
