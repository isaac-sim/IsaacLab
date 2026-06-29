Fixed
^^^^^

* Fixed :class:`~isaaclab_tasks.core.dexsuite.mdp.terminations.out_of_bound` so its world-space bounds
  track runtime updates to ``in_bound_range`` (e.g. from a curriculum term), rebuilding per-axis only
  when a bound changes instead of caching them once at construction.
* Fixed :class:`~isaaclab_tasks.core.dexsuite.mdp.commands.ObjectUniformPoseCommand` to skip the
  ``orientation_error`` metric when ``position_only`` is enabled.

Changed
^^^^^^^

* Tuned the Dexsuite reorient task (reach/tracking reward weights and standard deviations, Newton
  substeps) and added an out-of-bound z-range ADR curriculum term that widens the drop tolerance as
  difficulty increases.
