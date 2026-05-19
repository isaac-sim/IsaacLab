Fixed
^^^^^

* Removed the stale file-level ``@pytest.mark.xfail`` decorator on
  ``test_environments_newton`` (the cited Hydra deep-nesting issue was already
  resolved by PR #5029 and follow-ups #5130 / #5177).
