* Consolidated MicroDuck walking, recovery and skating terms into one ``mdp`` package. Existing walking
  APIs retain their behavior; distinct encoder storage, command sampling, grouped contact, recovery-gate
  and finite-value contracts use descriptive helper names within the shared modules. Updated task and
  agent configurations, lazy exports and tests to the canonical package without an alternate namespace.
