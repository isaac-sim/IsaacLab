Fixed
^^^^^

* Fixed physics-manager shutdown to notify all STOP listeners and clear shared
  state even when one listener fails.
* Fixed dead weakly referenced listeners to become no-ops instead of raising
  during lifecycle event dispatch.
* Fixed simulation-context teardown to use the resolved physics-manager class
  and finish cleanup before reporting STOP-listener failures.
