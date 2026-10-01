Added
^^^^^

* Allowed manager-based environment subclasses to record completed episodes without starting
  replacements. Inactive environments continued simulating, but produced no rewards, completion
  signals, or trajectory records. The default behavior retained automatic resets.
* Added optional environment IDs to recorder manager step methods so callers could record only
  selected environments while recorder terms continued returning data for the full batch.
