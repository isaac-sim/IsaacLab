Changed
^^^^^^^

* **Breaking:** Changed RLinf ``--checkpoint`` to accept a path rather than the ``latest`` or ``best``
  selectors. Pass a ``global_step_<N>`` directory, a directory below it, or its ``full_weights.pt`` file.
* Documented pinned RLinf and GR00T installation commands for N1.5 and N1.7.

Fixed
^^^^^

* Fixed RLinf evaluation ignoring ``--checkpoint``. N1.5 loaded the weight overlay through the
  extension; N1.7 used RLinf's native ``runner.ckpt_path`` hook.
* Fixed RLinf training resume by resolving ``--checkpoint`` to its enclosing ``global_step_<N>`` directory.
* Fixed relative model paths resolving against Ray worker directories instead of the launcher directory.
* Fixed RLinf 0.3 rollout workers missing model settings by merging the actor model configuration
  with explicit rollout overrides.
